"""Bounded HTTP and DNS-pinned public URL fetching for external tool data."""

import asyncio
import ipaddress
import json
import socket
import ssl
from urllib.parse import urlencode, urljoin, urlsplit

import httpx
import truststore

MAX_BYTES = 2 * 1024 * 1024
DNS_RESOLVERS = {
    "google": "https://dns.google/resolve",
    "cloudflare": "https://cloudflare-dns.com/dns-query",
}


class ToolError(Exception):
    """Only deliberate, credential-free error messages may reach the model."""


def client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        trust_env=False,
        follow_redirects=False,
        headers={"User-Agent": "PersonalAssistant/0.1", "Accept": "*/*"},
    )


async def read_response(response: httpx.Response) -> bytes:
    response.raise_for_status()
    data = bytearray()
    async for chunk in response.aiter_bytes():
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise ToolError("上游响应超过 2 MiB 限制。")
    return bytes(data)


async def request_json(timeout: float, method: str, url: str, **kwargs):
    async with asyncio.timeout(timeout):
        async with client(timeout) as session:
            async with session.stream(method, url, **kwargs) as response:
                import json

                return json.loads(await read_response(response))


async def public_targets(
    url: str, *, resolver: str = "", timeout: float = 15
) -> tuple[list[httpx.URL], str, str]:
    if len(url) > 4096 or any(ord(c) < 32 or c == "\\" for c in url):
        raise ToolError("URL 无效。")
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 80, 443}
        ):
            raise ValueError
        host = parsed.hostname.encode("idna").decode("ascii")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if resolver:
            if resolver not in DNS_RESOLVERS:
                raise ToolError("备用 DNS 仅支持 google 或 cloudflare。")
            query = DNS_RESOLVERS[resolver] + "?" + urlencode({"name": host, "type": "A"})
            # The fixed HTTPS resolver is itself fetched through the public-IP
            # checks, with fallback disabled so recursion cannot continue.
            raw, _, _ = await fetch_public(
                query, min(timeout, 4), dns_fallback="", accept="application/dns-json"
            )
            records = json.loads(raw)
            if not isinstance(records, dict) or str(records.get("Status")) != "0":
                raise ToolError("备用 DNS 解析失败。")
            ips = [
                ipaddress.ip_address(item["data"])
                for item in records.get("Answer", [])
                if item.get("type") in {1, 28}
            ]
        else:
            addresses = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
            ips = [ipaddress.ip_address(item[4][0]) for item in addresses]
        if not ips or any(not ip.is_global for ip in ips):
            raise ToolError("仅允许公网地址，拒绝本机、内网、保留及元数据服务地址。")
        original = httpx.URL(url)
        # Connect to the checked IP, retaining the origin Host and TLS SNI.
        # No second DNS lookup can redirect the connection into a private network.
        targets = [
            original.copy_with(host=str(ip))
            for ip in sorted(dict.fromkeys(ips), key=lambda ip: ip.version)[:16]
        ]
        return targets, original.netloc.decode(), host
    except (ValueError, UnicodeError, OSError):
        raise ToolError("URL 或域名无效，无法访问。") from None


async def public_target(url: str) -> tuple[httpx.URL, str, str]:
    targets, header, sni = await public_targets(url)
    return targets[0], header, sni


async def fetch_public(
    url: str, timeout: float, *, dns_fallback: str = "", accept: str = "*/*"
) -> tuple[bytes, str, str]:
    if dns_fallback and dns_fallback not in DNS_RESOLVERS:
        raise ToolError("备用 DNS 仅支持 google 或 cloudflare。")
    async with asyncio.timeout(timeout):
        async with client(timeout) as session:
            for _ in range(5):
                targets, host_header, sni = await public_targets(url)
                redirect = None
                last_error = None
                for phase in range(2 if dns_fallback else 1):
                    if phase:
                        try:
                            ipaddress.ip_address(sni)
                        except ValueError:
                            targets, host_header, sni = await public_targets(
                                url, resolver=dns_fallback, timeout=timeout
                            )
                        else:
                            break  # A user-specified IP must never be replaced.
                    for target in targets:
                        try:
                            async with session.stream(
                                "GET",
                                target,
                                headers={"Host": host_header, "Accept": accept},
                                extensions={"sni_hostname": sni},
                                timeout=httpx.Timeout(timeout, connect=min(3, timeout / 4)),
                            ) as response:
                                if response.is_redirect:
                                    location = response.headers.get("location")
                                    if not location:
                                        raise ToolError("重定向缺少目标地址。")
                                    redirect = urljoin(url, location)
                                else:
                                    return (
                                        await read_response(response),
                                        response.headers.get("content-type", ""),
                                        url,
                                    )
                        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                            last_error = error
                            continue
                        break
                    if redirect:
                        break
                if redirect:
                    url = redirect
                    continue
                if last_error:
                    raise last_error
                raise ToolError("无法连接上游公网地址。")
            raise ToolError("重定向次数超过限制。")
