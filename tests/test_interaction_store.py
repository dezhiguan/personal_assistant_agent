"""Control transactions, truthful acknowledgements, ownership and bounded acceptance."""

import json
from uuid import uuid4

import pytest

from personal_assistant_agent.interaction.store import Conflict, TaskStore


def new(store, cid="conversation"):
    return store.receive(
        cid,
        str(uuid4()),
        "搜索全球新闻",
        {
            "intent": "NEW_TASK",
            "goal": "搜索新闻",
            "constraints": {"范围": "全球"},
        },
        {},
    )["task_id"]


def test_controls_idempotency_conflicts_revision_and_terminal_race(tmp_path):
    store = TaskStore(str(tmp_path))
    try:
        tid = new(store)
        cid = str(uuid4())
        first = store.control(
            tid, "conversation", cid, "STEER", "只搜国内", {"范围": "中国大陆"}, 0
        )
        assert first == store.control(
            tid, "conversation", cid, "STEER", "只搜国内", {"范围": "中国大陆"}, 0
        )
        with pytest.raises(Conflict):
            store.control(tid, "conversation", cid, "STEER", "只搜国际", {"范围": "国际"}, 0)
        with pytest.raises(Conflict):
            store.control(tid, "conversation", str(uuid4()), "PAUSE", expected_revision=0)
        assert not store.finish(tid, 0, "过期全球结果")
        assert store.finish(tid, 1, "国内结果")
        assert store.finish(tid, 1, "不得重复通知")
        with pytest.raises(Conflict):
            store.control(tid, "conversation", str(uuid4()), "STEER", changes={"时间": "三天"})
        assert len([e for e in store.events("conversation") if e["type"] == "task.result"]) == 1
        with pytest.raises(KeyError):
            store.control(tid, "other-conversation", str(uuid4()), "CANCEL")
    finally:
        store.db.close()


def test_control_failure_rolls_back_without_ack(tmp_path, monkeypatch):
    store = TaskStore(str(tmp_path))
    try:
        tid = new(store)
        original_event = store.event

        def fail_event(*args, **kwargs):
            raise OSError("synthetic disk full")

        monkeypatch.setattr(store, "event", fail_event)
        with pytest.raises(OSError):
            store.control(tid, "conversation", str(uuid4()), "STEER", changes={"范围": "中国大陆"})
        assert store.task(tid)["revision"] == 0
        assert not store.controls(tid)
        monkeypatch.setattr(store, "event", original_event)
        assert "全球" in store.task(tid)["effective_task_request"]
    finally:
        store.db.close()


def test_fifo_revisions_and_checkpoint_ack_snapshot(tmp_path):
    store = TaskStore(str(tmp_path))
    try:
        tid = new(store)
        for changes in [
            {"范围": "中国大陆"},
            {"时间": "最近三天"},
            {"主题": "科技"},
            {"时间": "最近一周"},
        ]:
            store.control(tid, "conversation", str(uuid4()), "STEER", changes=changes)
        assert store.task(tid)["constraints"] == {
            "范围": "中国大陆",
            "时间": "最近一周",
            "主题": "科技",
        }
        assert all(not c["applied"] for c in store.controls(tid))
        store.applied(tid, 2, "checkpoint-2", "搜索最近三天的国内新闻")
        store.applied(tid, 2, "checkpoint-2", "搜索最近三天的国内新闻")
        applied = [e for e in store.events("conversation") if e["type"] == "control.applied"]
        assert len(applied) == 2
        assert all(
            e["payload"]["effective_task_request"] == "搜索最近三天的国内新闻" for e in applied
        )
        snapshot = store.snapshot("conversation")
        store.control(tid, "conversation", str(uuid4()), "CANCEL")
        subsequent = store.events("conversation", snapshot["cursor"])
        assert subsequent and all(e["sequence"] > snapshot["cursor"] for e in subsequent)
        assert not store.events("other-conversation")
    finally:
        store.db.close()


def test_message_idempotency_and_queue_limits(tmp_path):
    store = TaskStore(str(tmp_path))
    try:
        mid = str(uuid4())
        request = {"content": "你好"}
        decision = {"intent": "CHAT"}
        first = store.receive("c", mid, "你好", decision, request, queue_limit=1)
        assert store.receive("c", mid, "你好", decision, request, queue_limit=1) == first
        with pytest.raises(Conflict):
            store.receive("c", mid, "改变", decision, {"content": "改变"})
        with pytest.raises(Conflict):
            store.receive("c", str(uuid4()), "你好2", decision, {}, queue_limit=1)
        assert len(store.messages("c")) == 1
        assert json.loads(store.messages("c")[0]["response"])["pending"] is True
    finally:
        store.db.close()


def test_real_sqlite_busy_and_full_never_acknowledge_uncommitted_control(tmp_path):
    import sqlite3

    store = TaskStore(str(tmp_path))
    locker = sqlite3.connect(store.path, isolation_level=None)
    try:
        tid = new(store)
        store.db.execute("PRAGMA busy_timeout=20")
        locker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError) as busy:
            store.control(tid, "conversation", str(uuid4()), "STEER", changes={"范围": "中国大陆"})
        assert busy.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        locker.execute("ROLLBACK")
        assert store.task(tid)["revision"] == 0 and not store.controls(tid)
        pages = store.db.execute("PRAGMA page_count").fetchone()[0]
        store.db.execute(f"PRAGMA max_page_count={pages}")
        with pytest.raises(sqlite3.Error) as full:
            store.control(
                tid,
                "conversation",
                str(uuid4()),
                "STEER",
                "合成数据" * 2000,
                changes={"范围": "中国大陆"},
            )
        assert full.value.sqlite_errorcode == sqlite3.SQLITE_FULL
        assert store.task(tid)["revision"] == 0 and not store.controls(tid)
    finally:
        locker.close()
        store.db.close()


def test_full_steering_queue_preserves_cancel_priority(tmp_path):
    store = TaskStore(str(tmp_path), control_limit=1)
    try:
        tid = new(store)
        store.control(tid, "conversation", str(uuid4()), "STEER", changes={"范围": "中国大陆"})
        with pytest.raises(Conflict, match="修正队列已满"):
            store.control(tid, "conversation", str(uuid4()), "STEER", changes={"时间": "三天"})
        assert store.task(tid)["revision"] == 1
        store.control(tid, "conversation", str(uuid4()), "CANCEL")
        assert store.task(tid)["status"] == "CANCELLED"
    finally:
        store.db.close()
