"""Single-process control plane. All acknowledgements follow a durable transaction."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
ACTIVE = {"RUNNING", "PAUSING", "CANCELLING"}


class Conflict(ValueError):
    pass


class TaskStore:
    def __init__(self, directory: str, *, control_limit: int = 128):
        self.control_limit = control_limit
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        self.path = root / "tasks.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        self.path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA synchronous=FULL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in {0, 1}:
            self.db.close()
            raise RuntimeError("不兼容的任务数据库版本。")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, conversation TEXT, data TEXT);
            CREATE TABLE IF NOT EXISTS messages(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT, id TEXT UNIQUE,
                content TEXT, role TEXT, kind TEXT, response TEXT, request TEXT);
            CREATE TABLE IF NOT EXISTS controls(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, task TEXT,
                kind TEXT, content TEXT, revision INTEGER, applied INTEGER DEFAULT 0,
                request TEXT, receipt TEXT);
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT, data TEXT);
            CREATE TABLE IF NOT EXISTS operations(
                id TEXT PRIMARY KEY, task TEXT, name TEXT, status TEXT, result TEXT);
            CREATE INDEX IF NOT EXISTS events_conversation ON events(conversation, seq);
            CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation, seq);
            CREATE INDEX IF NOT EXISTS controls_task ON controls(task, seq);
            PRAGMA user_version=1;
        """)
        # SQLite creates WAL/SHM next to the database; the enclosing directory is private.

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def task(self, task_id, conversation=None):
        row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row or (conversation is not None and row["conversation"] != conversation):
            raise KeyError("任务不属于当前会话或不存在。")
        return json.loads(row["data"])

    def tasks(self, conversation=None):
        rows = self.db.execute(
            "SELECT data FROM tasks" + (" WHERE conversation=?" if conversation else ""),
            (conversation,) if conversation else (),
        )
        return [json.loads(row[0]) for row in rows]

    def save(self, task):
        self.db.execute(
            "INSERT OR REPLACE INTO tasks VALUES(?,?,?)",
            (
                task["id"],
                task["conversation_id"],
                json.dumps(task, ensure_ascii=False),
            ),
        )

    def event(self, conversation, kind, payload, task=None, chat=None):
        event = {
            "conversation_id": conversation,
            "type": kind,
            "payload": payload,
            "task_id": task,
            "chat_message_id": chat,
            "occurred_at": time.time(),
        }
        if task:
            execution = self.task(task)
            event.update(
                task_revision=execution["revision"], attempt_id=execution.get("attempt_id")
            )
        cursor = self.db.execute(
            "INSERT INTO events(conversation,data) VALUES(?,?)",
            (
                conversation,
                json.dumps(event, ensure_ascii=False),
            ),
        )
        return cursor.lastrowid

    def events(self, conversation, after=0, limit=200):
        rows = self.db.execute(
            "SELECT seq,data FROM events WHERE conversation=? AND seq>? ORDER BY seq LIMIT ?",
            (conversation, after, limit),
        )
        return [
            {
                **json.loads(r["data"]),
                "sequence": r["seq"],
                "event_id": f"{conversation}:{r['seq']}",
            }
            for r in rows
        ]

    def snapshot(self, conversation):
        with self.transaction():
            high = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
            return {
                "tasks": self.tasks(conversation),
                "messages": self.messages(conversation),
                "cursor": high,
            }

    def messages(self, conversation, kind=None):
        rows = self.db.execute(
            "SELECT * FROM messages WHERE conversation=?"
            + (" AND kind=?" if kind else "")
            + " ORDER BY seq",
            (conversation, kind) if kind else (conversation,),
        )
        return [dict(r) for r in rows]

    def duplicate(self, message_id, request):
        row = self.db.execute(
            "SELECT request,response FROM messages WHERE id=?", (message_id,)
        ).fetchone()
        if row:
            if json.loads(row["request"] or "null") != request:
                raise Conflict("同一消息 ID 的内容不同。")
            return json.loads(row["response"])

    def receive(self, conversation, message_id, content, decision, request, *, queue_limit=32):
        with self.transaction():
            if duplicate := self.duplicate(message_id, request):
                return duplicate
            kind = decision["intent"]
            task_id = decision.get("target_task_id")
            response = {"message_id": message_id, "intent": kind}
            if kind == "STEER" and task_id:
                target = self.task(task_id, conversation)
                expected = decision.get("expected_revision")
                if expected is not None and target["revision"] != expected:
                    raise Conflict("路由期间任务要求已更新；本条修正尚未接受，请核对后重试。")
                if target["status"] in TERMINAL:
                    kind = "NEW_TASK"
                    decision = {
                        **decision,
                        "source_task_id": task_id,
                        "goal": decision.get("changes", {}).get("目标") or target["goal"],
                        "constraints": {
                            **target["constraints"],
                            **decision.get("changes", {}),
                        },
                    }
                    response.update(intent=kind, revision_of=task_id)
            if kind == "NEW_TASK":
                if source_id := decision.get("source_task_id"):
                    source = self.task(source_id, conversation)
                    decision = {
                        **decision,
                        "constraints": {
                            **source["constraints"],
                            **decision.get("constraints", {}),
                        },
                    }
                    response["revision_of"] = source_id
                pending = [t for t in self.tasks() if t["status"] not in TERMINAL]
                if len(pending) >= queue_limit:
                    raise Conflict("任务队列已满，请稍后提交。")
                task_id = str(uuid4())
                task = {
                    "id": task_id,
                    "conversation_id": conversation,
                    "status": "QUEUED",
                    "original_request": content,
                    "authorization_message_id": message_id,
                    "goal": decision.get("goal") or content,
                    "constraints": decision.get("constraints", {}),
                    "revision": 0,
                    "applied_revision": -1,
                    "desired_action": "RUN",
                    "result": None,
                    "source_task_id": decision.get("source_task_id"),
                    "created_at": time.time(),
                    "attempt": 0,
                    "active_seconds": 0,
                    "model_calls": 0,
                    "tool_calls": 0,
                    "progress": {},
                    "fingerprint": None,
                }
                task["effective_task_request"] = effective(task)
                self.save(task)
                self.event(conversation, "task.created", task, task_id)
                response["task_id"] = task_id
            elif kind in {"STEER", "PAUSE", "RESUME", "CANCEL"}:
                if not task_id:
                    raise Conflict("请指定要控制的任务。")
                response.update(
                    self._control(
                        task_id,
                        conversation,
                        message_id,
                        kind,
                        content,
                        decision.get("changes", {}),
                        decision.get("expected_revision"),
                        request,
                    )
                )
            elif kind == "CHAT":
                waiting = self.db.execute(
                    "SELECT count(*) FROM messages WHERE conversation=? AND kind='CHAT' "
                    "AND role='user' AND response LIKE '%\"pending\": true%'",
                    (conversation,),
                ).fetchone()[0]
                if waiting >= queue_limit:
                    raise Conflict("聊天回复队列已满，请稍后发送。")
                response["pending"] = True
            elif kind == "STATUS":
                response["tasks"] = (
                    [self.task(task_id, conversation)] if task_id else self.tasks(conversation)
                )
            elif kind == "CLARIFY":
                response["question"] = decision.get("question") or "请指定任务目标。"
                response["candidates"] = decision.get("candidates", [])
            self.db.execute(
                "INSERT INTO messages(conversation,id,content,role,kind,response,request) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    conversation,
                    message_id,
                    content,
                    "user",
                    kind,
                    json.dumps(response, ensure_ascii=False),
                    json.dumps(request),
                ),
            )
            self.event(
                conversation, "message.received", response, response.get("task_id"), message_id
            )
            return response

    def control(
        self,
        task_id,
        conversation,
        control_id,
        kind,
        content="",
        changes=None,
        expected_revision=None,
    ):
        request = {
            "conversation": conversation,
            "task_id": task_id,
            "kind": kind,
            "content": content,
            "changes": changes or {},
            "expected": expected_revision,
        }
        with self.transaction():
            return self._control(
                task_id,
                conversation,
                control_id,
                kind,
                content,
                changes or {},
                expected_revision,
                request,
            )

    def _control(
        self, task_id, conversation, control_id, kind, content, changes, expected, request
    ):
        row = self.db.execute(
            "SELECT request,receipt FROM controls WHERE id=?", (control_id,)
        ).fetchone()
        if row:
            if json.loads(row["request"]) != request:
                raise Conflict("同一控制 ID 的内容不同。")
            return json.loads(row["receipt"])
        task = self.task(task_id, conversation)
        if expected is not None and task["revision"] != expected:
            raise Conflict("任务版本已改变，请刷新后重试。")
        if task["status"] in TERMINAL or task["desired_action"] == "CANCEL":
            raise Conflict("TASK_ALREADY_TERMINAL：请创建关联原成果的新任务。")
        safe_recovery = (
            task["status"] == "RECOVERY_REQUIRED" and task.get("recoverable") and kind == "RESUME"
        )
        if (
            task["status"] in {"RECOVERY_REQUIRED", "WAITING_INPUT"}
            and kind != "CANCEL"
            and not safe_recovery
        ):
            raise Conflict("任务需要核对或匹配的业务输入，不能通过普通控制恢复。")
        if kind == "RESUME" and task["status"] not in {"PAUSED", "PAUSING"} and not safe_recovery:
            raise Conflict("仅可恢复已暂停或正在暂停的任务；未知操作须先核对。")
        if kind == "STEER":
            pending = self.db.execute(
                "SELECT count(*) FROM controls WHERE task=? AND kind='STEER' AND applied=0",
                (task_id,),
            ).fetchone()[0]
            if pending >= self.control_limit:
                raise Conflict("任务修正队列已满；暂停、取消和状态查询仍可使用。")
            task.setdefault("constraint_sources", {}).update({k: control_id for k in changes})
            if changes.get("目标"):
                task["goal"] = changes["目标"]
            task["constraints"].update({k: v for k, v in changes.items() if k != "目标"})
            task["effective_task_request"] = effective(task)
        elif kind == "PAUSE":
            if task["status"] in {"QUEUED", "PAUSED"}:
                task["status"] = "PAUSED"
            else:
                task["status"] = "PAUSING"
            task["desired_action"] = "PAUSE"
        elif kind == "CANCEL":
            task["desired_action"] = "CANCEL"
            from personal_assistant_agent.interaction.middleware import READ_ONLY

            unknown = [
                r[0]
                for r in self.db.execute(
                    "SELECT name FROM operations WHERE task=? AND status='STARTED'",
                    (task_id,),
                )
                if r[0] not in READ_ONLY
            ]
            if task["status"] == "RECOVERY_REQUIRED" and unknown:
                task["status"] = "RECOVERY_REQUIRED"
                task["reason"] = "取消已收到；未知写操作或同步线程须先核对是否结束。"
            else:
                task["status"] = "CANCELLING" if task["status"] in ACTIVE else "CANCELLED"
        elif kind == "RESUME":
            task["desired_action"] = "RUN"
            if task["status"] == "PAUSED" or safe_recovery:
                task["status"] = "QUEUED"
        else:
            raise Conflict("未知控制类型。")
        task["revision"] += 1
        self.save(task)
        cursor = self.db.execute(
            "INSERT INTO controls(id,task,kind,content,revision,request) VALUES(?,?,?,?,?,?)",
            (control_id, task_id, kind, content, task["revision"], json.dumps(request)),
        )
        receipt = {
            "task_id": task_id,
            "control_id": control_id,
            "control_seq": cursor.lastrowid,
            "status": "RECEIVED",
            "task_revision": task["revision"],
            "effective_task_request": task["effective_task_request"],
            "changes": changes,
        }
        self.db.execute(
            "UPDATE controls SET receipt=? WHERE id=?",
            (
                json.dumps(receipt, ensure_ascii=False),
                control_id,
            ),
        )
        self.event(conversation, "control.received", receipt, task_id)
        self.event(conversation, "task.state_changed", task, task_id)
        self._settled(task)
        return receipt

    def controls(self, task_id):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM controls WHERE task=? ORDER BY seq",
                (task_id,),
            )
        ]

    def applied(self, task_id, revision, checkpoint, effective_request=None):
        with self.transaction():
            task = self.task(task_id)
            task["applied_revision"] = max(task["applied_revision"], revision)
            self.save(task)
            for control in self.controls(task_id):
                if (
                    not control["applied"]
                    and control["kind"] == "STEER"
                    and control["revision"] <= revision
                ):
                    self.db.execute("UPDATE controls SET applied=1 WHERE id=?", (control["id"],))
                    self.event(
                        task["conversation_id"],
                        "control.applied",
                        {
                            "control_id": control["id"],
                            "directive_revision": revision,
                            "checkpoint_id": checkpoint,
                            "scope": "root",
                            "effective_task_request": effective_request,
                        },
                        task_id,
                    )

    def state(self, task_id, status, **fields):
        with self.transaction():
            task = self.task(task_id)
            task.update(status=status, **fields)
            if status in {"CANCELLED", "RECOVERY_REQUIRED"}:
                task["effects"] = [
                    dict(r)
                    for r in self.db.execute(
                        "SELECT name,status FROM operations WHERE task=?",
                        (task_id,),
                    )
                ]
            self.save(task)
            self.event(task["conversation_id"], "task.state_changed", task, task_id)
            self._settled(task)
            return task

    def _settled(self, task):
        matches = {"PAUSED": "PAUSE", "CANCELLED": "CANCEL", "RUNNING": "RESUME"}
        for item in self.controls(task["id"]):
            if item["kind"] == matches.get(task["status"]) and not item["applied"]:
                self.db.execute("UPDATE controls SET applied=1 WHERE id=?", (item["id"],))
                self.event(
                    task["conversation_id"],
                    "control.settled",
                    {
                        "control_id": item["id"],
                        "status": task["status"],
                    },
                    task["id"],
                )

    def finish(self, task_id, revision, result):
        with self.transaction():
            task = self.task(task_id)
            if task["status"] == "SUCCEEDED":
                return True
            if task["desired_action"] != "RUN" or task["revision"] != revision:
                return False
            task.update(status="SUCCEEDED", result=result)
            self.save(task)
            # A transaction commits result, notification and event together, once.
            self.db.execute(
                "INSERT OR IGNORE INTO messages"
                "(conversation,id,content,role,kind) VALUES(?,?,?,?,?)",
                (
                    task["conversation_id"],
                    f"result:{task_id}",
                    result,
                    "assistant",
                    "RESULT",
                ),
            )
            self.event(task["conversation_id"], "task.result", task, task_id)
            return True


def effective(task):
    return task["goal"] + (
        "\n当前要求："
        + "；".join(f"{k}={v}" for k, v in task["constraints"].items() if v is not None)
        if task["constraints"]
        else ""
    )
