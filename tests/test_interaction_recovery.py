"""UC10 / A6: kill real processes; inspect SQLite and the synthetic supplier ledger."""

import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from personal_assistant_agent.interaction.store import TaskStore

FIXTURE = Path(__file__).with_name("interaction_process.py")


def wait_until(check, process=None):
    deadline = time.monotonic() + 15
    while not check():
        if process and process.poll() is not None:
            _, stderr = process.communicate()
            pytest.fail(f"Fixture exited {process.returncode}: {stderr}")
        if time.monotonic() > deadline:
            pytest.fail("Subprocess acceptance barrier timed out")
        time.sleep(0.02)


def launch(root, mode):
    return subprocess.Popen(
        [sys.executable, str(FIXTURE), mode],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def kill(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


@pytest.mark.parametrize("mode", ["safe", "effect", "effect-gap", "pause", "cancel"])
def test_disk_restart_at_execution_boundaries(tmp_path, mode):
    first = launch(tmp_path, mode)
    store = None
    second = None
    try:
        wait_until(lambda: (tmp_path / "tool-started").exists(), first)
        store = TaskStore(str(tmp_path / "private"))
        task_id = store.tasks()[0]["id"]
        if mode == "pause":
            store.control(task_id, "synthetic-conversation", str(uuid4()), "PAUSE")
            (tmp_path / "release").touch()
            wait_until(lambda: store.task(task_id)["status"] == "PAUSED", first)
        elif mode == "cancel":
            store.control(task_id, "synthetic-conversation", str(uuid4()), "CANCEL")
        elif mode == "effect-gap":
            (tmp_path / "release").touch()
            first.wait(timeout=15)
            assert first.returncode == 37
            assert (tmp_path / "effect-settled-gap").exists()
        kill(first)
        if mode == "effect-gap":
            store.control(
                task_id, "synthetic-conversation", str(uuid4()), "STEER", changes={"格式": "简短"}
            )
        (tmp_path / "ready").unlink(missing_ok=True)
        (tmp_path / "release").touch()
        second = launch(tmp_path, "effect" if mode == "effect-gap" else mode)
        wait_until(lambda: (tmp_path / "ready").exists(), second)
        expected = {
            "safe": "SUCCEEDED",
            "effect": "RECOVERY_REQUIRED",
            "effect-gap": "SUCCEEDED",
            "pause": "PAUSED",
            "cancel": "CANCELLED",
        }[mode]
        wait_until(lambda: store.task(task_id)["status"] == expected, second)
        if mode == "effect-gap":
            assert "格式=简短" in store.task(task_id)["result"]
        if "effect" in mode:
            assert (tmp_path / "supplier-ledger").read_text().splitlines() == ["effect"]
        if mode == "safe":
            assert (tmp_path / "reads").read_text().splitlines() == ["read", "read"]
        if mode == "pause":
            store.control(task_id, "synthetic-conversation", str(uuid4()), "RESUME")
            wait_until(lambda: store.task(task_id)["status"] == "SUCCEEDED", second)
        if expected == "SUCCEEDED" or mode == "pause":
            current = store.task(task_id)
            assert current["trace_summary"]["model_calls"] == current["model_calls"]
            assert current["trace_summary"]["tool_calls"] == current["tool_calls"]
            result_events = [
                e
                for e in store.events("synthetic-conversation", limit=10000)
                if e["type"] == "task.result"
            ]
            assert len(result_events) == 1
            assert (
                json.loads(
                    store.db.execute(
                        "SELECT data FROM tasks WHERE id=?",
                        (task_id,),
                    ).fetchone()[0]
                )["attempt"]
                >= 2
            )
    finally:
        kill(first)
        if second:
            kill(second)
        if store:
            store.db.close()


@pytest.mark.parametrize(
    "mode,exit_code", [("queued-gap", 36), ("applied-gap", 38), ("result-gap", 39)]
)
def test_restart_repairs_accept_checkpoint_and_notification_gaps(tmp_path, mode, exit_code):
    first = launch(tmp_path, mode)
    second = None
    store = None
    try:
        if mode == "result-gap":
            wait_until(lambda: (tmp_path / "tool-started").exists(), first)
            (tmp_path / "release").touch()
        first.wait(timeout=15)
        assert first.returncode == exit_code, first.communicate()[1]
        assert (tmp_path / mode).exists()
        store = TaskStore(str(tmp_path / "private"))
        tid = store.tasks()[0]["id"]
        (tmp_path / "ready").unlink(missing_ok=True)
        (tmp_path / "release").touch()
        second = launch(tmp_path, "safe")
        wait_until(lambda: store.task(tid)["status"] == "SUCCEEDED", second)
        events = store.events("synthetic-conversation", limit=10000)
        assert len([e for e in events if e["type"] == "task.result"]) == 1
        assert len([e for e in events if e["type"] == "control.applied"]) == (mode != "result-gap")
        if mode != "result-gap":
            assert "中国大陆" in store.task(tid)["result"]
        assert (tmp_path / "reads").read_text().splitlines() == ["read"]
    finally:
        kill(first)
        if second:
            kill(second)
        if store:
            store.db.close()


def test_http_shutdown_with_open_sse_drains_graph_and_restarts(tmp_path):
    """A real ASGI server must reach lifespan cleanup despite an open subscription."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    fixture_dir = str(Path(__file__).parent)
    script = (
        f"import sys; sys.path.insert(0, {fixture_dir!r}); "
        "import interaction_browser as fixture; original=fixture.uvicorn.run; "
        f"fixture.uvicorn.run=lambda app, **kw: original(app, **{{**kw, 'port': {port}, "
        "'timeout_graceful_shutdown': 0.2}); fixture.main()"
    )

    def serve():
        return subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    first, second, store = serve(), None, None
    base = f"http://127.0.0.1:{port}"
    cid = str(uuid4())
    try:
        with httpx.Client(base_url=base, timeout=2) as client:

            def ready():
                try:
                    return client.get("/healthz").status_code == 200
                except httpx.TransportError:
                    return False

            wait_until(ready, first)
            receipt = client.post(
                f"/api/conversations/{cid}/messages",
                json={
                    "client_message_id": str(uuid4()),
                    "content": "搜索全球新闻",
                    "intent": "NEW_TASK",
                },
            )
            assert receipt.status_code == 202
            tid = receipt.json()["task_id"]
            wait_until(lambda: (tmp_path / "tool-started").exists(), first)
            with client.stream("GET", f"/api/conversations/{cid}/events") as stream:
                assert stream.status_code == 200
                first.send_signal(signal.SIGTERM)
                time.sleep(0.4)
                # HTTP draining elapsed; the slow tool still owns the graph.
                assert first.poll() is None
                (tmp_path / "release").touch()
                first.wait(timeout=15)  # Keep the SSE socket open until the process exits.
                assert first.returncode in {0, -signal.SIGTERM}
            store = TaskStore(str(tmp_path / ".assistant-private/browser-fixture"))
            second = serve()
            wait_until(ready, second)
            wait_until(lambda: store.task(tid)["status"] == "SUCCEEDED", second)
            assert store.task(tid)["model_calls"] == 2
            assert (tmp_path / "reads").read_text().splitlines() == ["read"]
            assert len([e for e in store.events(cid) if e["type"] == "task.result"]) == 1
    finally:
        kill(first)
        if second:
            kill(second)
        if store:
            store.db.close()
