"""Bounded HTTP and DNS-pinned public URL fetching for external tool data."""

import asyncio
import ipaddress
import socket
import ssl
from urllib.parse import urljoin, urlsplit

import httpx
import truststore

MAX_BYTES = 2 * 1024 * 1024


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


async def public_target(url: str) -> tuple[httpx.URL, str, str]:
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
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
        ips = [ipaddress.ip_address(item[4][0]) for item in addresses]
        if not ips or any(not ip.is_global for ip in ips):
            raise ToolError("仅允许公网地址，拒绝本机、内网、保留及元数据服务地址。")
        original = httpx.URL(url)
        # Connect to the checked IP, retaining the origin Host and TLS SNI.
        # No second DNS lookup can redirect the connection into a private network.
        return original.copy_with(host=str(ips[0])), original.netloc.decode(), host
    except (ValueError, UnicodeError, OSError):
        raise ToolError("URL 或域名无效，无法访问。") from None


async def fetch_public(url: str, timeout: float) -> tuple[bytes, str, str]:
    async with asyncio.timeout(timeout):
        async with client(timeout) as session:
            for _ in range(5):
                target, host_header, sni = await public_target(url)
                async with session.stream(
                    "GET",
                    target,
                    headers={"Host": host_header},
                    extensions={"sni_hostname": sni},
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ToolError("重定向缺少目标地址。")
                        url = urljoin(url, location)
                        continue
                    return (
                        await read_response(response),
                        response.headers.get("content-type", ""),
                        url,
                    )
            raise ToolError("重定向次数超过限制。")
