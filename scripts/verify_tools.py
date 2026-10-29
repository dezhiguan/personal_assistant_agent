"""Exercise the running app with real Qwen, real tools and persisted file checks.

Uses only synthetic inputs. No model/HTTP substitutes. Produces private local
artifacts, makes paid model calls, and keeps its files in a unique workspace folder.
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import truststore

from personal_assistant_agent.core.config import get_settings


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Verification:
    def __init__(self, url: str, output: Path, *, retry_failed: bool = False):
        self.url = url.rstrip("/")
        self.output = output
        self.cases: dict[str, dict] = {}
        self.client = httpx.Client(timeout=330, trust_env=False)
        self.folder = "live-check-" + uuid4().hex[:10]
        self.workspace = Path(get_settings().workspace_dir).resolve()
        self.workflow = None
        self.retry_failed = retry_failed
        if retry_failed:
            previous = json.loads((output / "report.json").read_text())
            self.cases = previous["cases"]
            self.folder = previous["workspace_folder"]
            if not re.fullmatch(r"live-check-[a-f0-9]{10}", self.folder):
                raise ValueError("验收报告中的工作区目录无效。")
            state = output / "workflow-state.json"
            if state.exists():
                self.workflow = json.loads(state.read_text())

    def chat(self, name, prompt, *, previous=None):
        previous = previous or {}
        conversation_id = previous.get("conversation_id", str(uuid4()))
        messages = [*previous.get("messages", []), {"role": "user", "content": prompt}]
        events = []
        with self.client.stream(
            "POST",
            self.url + "/api/chat",
            json={
                "conversation_id": conversation_id,
                "messages": messages,
                "todos": previous.get("todos", []),
            },
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            for line in response.iter_lines():
                if line:
                    event = json.loads(line)
                    events.append(event)
                    if event["type"] == "tool":
                        print(f"  {name}: {event['name']} {event['status']}", flush=True)
        save(self.output / f"{name}-events.json", events)
        if not events or events[-1]["type"] != "done":
            raise RuntimeError("未收到 done；请查看对应 events 文件。")
        answer = "".join(e["text"] for e in events if e["type"] == "delta")
        if not answer.strip():
            raise RuntimeError("回复正文为空。")
        plans = [e["todos"] for e in events if e["type"] == "plan"]
        result = {
            "events": events,
            "answer": answer,
            "todos": plans[-1] if plans else previous.get("todos", []),
            "messages": [*messages, {"role": "assistant", "content": answer}],
            "conversation_id": conversation_id,
            "tools": {e["name"] for e in events if e["type"] == "tool"},
        }
        self.cases[name] = {
            "request_id": events[0]["request_id"],
            "tools": sorted(result["tools"]),
            "plan_updates": len(plans),
            "usage": next((e for e in events if e["type"] == "usage"), None),
        }
        save(
            self.output / f"{name}-state.json",
            {k: v for k, v in result.items() if k not in {"events", "tools"}},
        )
        return result

    def basic(self):
        result = self.chat("casual", "你好，请只用一句话打个招呼。")
        assert not result["tools"] and not result["todos"], "简单闲聊不应创建计划或调用工具。"

    def time(self):
        result = self.chat("time", "现在上海准确的日期和时间是多少？本周周一和周日分别是哪天？")
        assert "get_current_time" in result["tools"], "相对时间问题必须真实调用时钟。"

    def math(self):
        result = self.chat("calculate", "帮我计算 200*(1+10%) + sqrt(81)，给出结果。")
        assert "calculate" in result["tools"], "数学问题必须调用计算器。"
        assert "229" in result["answer"], "计算结果应为 229。"

    def files_and_plan(self):
        result = self.chat(
            "workflow",
            (
                "这是实际执行验收，请先用 write_todos 建立至少四步计划，再完成全部步骤："
                f"1.在工作区虚拟目录 /{self.folder}/ 下创建 result.txt，内容严格为"
                " price=200 和 quantity=3 两行，每行末尾换行；"
                "2.读取文件，使用 edit_file 把 quantity=3 改为 quantity=4；"
                f"3.使用 execute 执行 wc -l /{self.folder}/result.txt，核对行数；"
                "4.必须调用一次 task，委派 general-purpose 子代理读取这个文件，"
                "调用 calculate 计算 200*4*(1-10%)，将结果保存到同目录 summary.txt，"
                "内容严格为 total=720 加一个换行；"
                "5.主代理读回两个文件核对，全部成功后更新完整待办为 completed，再简短交付结果。"
                "只操作这个验收目录，不操作其他文件。"
            ),
        )
        self.workflow = result
        assert {"write_file", "edit_file", "read_file", "execute", "task"} <= result["tools"], (
            "缺少文件、命令或子代理的真实调用。"
        )
        assert self.cases["workflow"]["plan_updates"] >= 2, "计划必须建立并更新。"
        assert len(result["todos"]) >= 4 and all(
            t["status"] == "completed" for t in result["todos"]
        ), "已实际完成的步骤应全部更新为 completed。"
        assert (self.workspace / self.folder / "result.txt").read_text() == (
            "price=200\nquantity=4\n"
        ), "主代理文件落盘内容不正确。"
        assert (self.workspace / self.folder / "summary.txt").read_text() == "total=720\n", (
            "子代理文件落盘内容不正确。"
        )

    def follow_up(self):
        assert self.workflow, "需要先完成文件工作流。"
        result = self.chat(
            "follow-up",
            (
                "继续这个计划，保留原有已完成步骤，再追加一个新步骤："
                f"把 /{self.folder}/result.txt 中的 quantity=4 修改为 quantity=5，"
                "并读回核对；这次不改 summary.txt。完成后更新完整待办并简短交付。"
            ),
            previous=self.workflow,
        )
        assert {"edit_file", "read_file"} <= result["tools"], "追问必须实际修改并读回。"
        assert result["todos"][: len(self.workflow["todos"])] == self.workflow["todos"], (
            "追加计划必须保留全部原有步骤和完成状态。"
        )
        assert len(result["todos"]) > len(self.workflow["todos"]), "追问应追加计划步骤。"
        assert all(t["status"] == "completed" for t in result["todos"]), "追问计划未完成。"
        assert (self.workspace / self.folder / "result.txt").read_text() == (
            "price=200\nquantity=5\n"
        ), "追问修改未落盘。"

    def rejection(self):
        result = self.chat(
            "boundary",
            (
                "请实际调用 execute 运行 cat ../.env，验证工作区会拒绝越界。"
                "这是失败路径验收，不要换工具或路径绕过，不需要其他计划；"
                "根据真实返回明确报告失败或拒绝。"
            ),
        )
        assert result["tools"] == {"execute"}, "应只调用受限 execute。"
        assert any(w in result["answer"] for w in ["失败", "拒绝", "禁止", "不允许"]), (
            "必须如实报告越界失败。"
        )

    def search(self):
        result = self.chat(
            "baidu",
            (
                "请用百度查找 LangChain Deep Agents 官方文档，返回至少两个标题和链接。"
                "只有真实检索成功才能给搜索结果；遇到验证码等阻碍应明确报告。"
            ),
        )
        assert "baidu_search" in result["tools"], "没有调用百度搜索。"
        assert not any(
            e["type"] == "tool" and e["name"] == "baidu_search" and e["status"] == "error"
            for e in result["events"]
        ), "百度工具真实返回失败，不能把建议链接算作检索结果。"
        assert len(re.findall(r"https?://[^\s)]+", result["answer"])) >= 2, (
            "百度未返回可用结果，查看 baidu-events.json 确认验证码或网络原因。"
        )
        assert not any(word in result["answer"] for word in ["验证码", "检索失败", "搜索失败"]), (
            "百度检索受阻；建议链接不能作为真实检索成功的证据。"
        )

    def run(self):
        status = self.client.get(self.url + "/healthz").raise_for_status().json()
        assert status["model_configured"], "运行服务未配置 Qwen。"
        for name, check in [
            ("casual", self.basic),
            ("time", self.time),
            ("calculate", self.math),
            ("workflow", self.files_and_plan),
            ("follow-up", self.follow_up),
            ("boundary", self.rejection),
            ("baidu", self.search),
        ]:
            if self.retry_failed and self.cases.get(name, {}).get("status") == "passed":
                continue
            previous = self.cases.pop(name, {})
            attempts = [*previous.get("attempts", [])]
            if previous:
                attempts.append({k: v for k, v in previous.items() if k != "attempts"})
                for kind in ["events", "state"]:
                    original = self.output / f"{name}-{kind}.json"
                    if original.exists():
                        save(
                            self.output / f"{name}-attempt-{len(attempts)}-{kind}.json",
                            json.loads(original.read_text()),
                        )
            print(f"START {name}", flush=True)
            try:
                check()
                self.cases[name]["status"] = "passed"
                print(f"PASS {name}", flush=True)
            except (AssertionError, RuntimeError, OSError, httpx.HTTPError) as error:
                self.cases.setdefault(name, {})["status"] = "failed"
                self.cases[name]["error"] = (
                    str(error)
                    if isinstance(error, (AssertionError, RuntimeError))
                    else type(error).__name__
                )
                print(f"FAIL {name}: {self.cases[name]['error']}", flush=True)
            if attempts:
                self.cases[name]["attempts"] = attempts
            save(
                self.output / "report.json",
                {
                    "verified_at": datetime.now(UTC).isoformat(),
                    "base_url": self.url,
                    "model": status["model"],
                    "workspace_folder": self.folder,
                    "cases": self.cases,
                    "passed": len(self.cases) == 7
                    and all(c["status"] == "passed" for c in self.cases.values()),
                },
            )
        return all(c["status"] == "passed" for c in self.cases.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/live-tools/latest"))
    parser.add_argument(
        "--retry-failed", action="store_true", help="只重试报告中的失败项，并保留先前尝试与事件。"
    )
    args = parser.parse_args()
    truststore.inject_into_ssl()
    verification = Verification(args.url, args.output, retry_failed=args.retry_failed)
    try:
        passed = verification.run()
    finally:
        verification.client.close()
    print(f"Report: {args.output / 'report.json'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
