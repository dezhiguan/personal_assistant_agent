"""Public contracts remain incremental, scoped and independent of the task model."""

from uuid import uuid4


def test_persisted_messages_status_controls_and_scope(client):
    cid, other, mid = str(uuid4()), str(uuid4()), str(uuid4())
    runtime = client.app.state.interaction

    # Queue-paused fixture avoids model traffic while exercising public APIs.
    def seed():
        task = runtime.store.receive(
            cid, str(uuid4()), "搜索新闻", {"intent": "NEW_TASK", "goal": "搜索新闻"}, {}
        )["task_id"]
        runtime.store.control(task, cid, str(uuid4()), "PAUSE")
        return task

    task = client.portal.call(seed)
    ack = client.post(
        f"/api/conversations/{cid}/messages",
        json={
            "client_message_id": mid,
            "content": "所有任务现在怎么样",
            "intent": "STATUS",
        },
    )
    assert ack.status_code == 202
    assert ack.json()["tasks"][0]["status"] == "PAUSED"
    assert (
        client.post(
            f"/api/conversations/{cid}/messages",
            json={
                "client_message_id": mid,
                "content": "所有任务现在怎么样",
                "intent": "STATUS",
            },
        ).json()
        == ack.json()
    )
    assert (
        client.post(
            f"/api/conversations/{cid}/messages",
            json={
                "client_message_id": mid,
                "content": "改成不同内容",
                "intent": "STATUS",
            },
        ).status_code
        == 409
    )
    snapshot = client.get(f"/api/conversations/{cid}/tasks").json()
    assert snapshot["cursor"] and snapshot["tasks"][0]["id"] == task
    assert client.get(f"/api/tasks/{task}", params={"conversation_id": other}).status_code == 404
    assert client.get(f"/api/tasks/{task}", params={"conversation_id": cid}).status_code == 200
    body = {"conversation_id": cid, "control_id": str(uuid4()), "type": "CANCEL"}
    assert client.post(f"/api/tasks/{task}/controls", json=body).status_code == 202
    assert client.post(f"/api/tasks/{task}/controls", json=body).status_code == 202
    body["type"] = "RESUME"
    assert client.post(f"/api/tasks/{task}/controls", json=body).status_code == 409
    assert client.get(f"/api/conversations/{cid}/events", params={"after": -1}).status_code == 400
    assert (
        client.get(
            f"/api/conversations/{cid}/events",
            headers={
                "Last-Event-ID": f"{other}:1",
            },
        ).status_code
        == 400
    )


def test_untrusted_input_and_ambiguous_task_target(client):
    runtime = client.app.state.interaction
    cid = str(uuid4())

    def seed():
        task_ids = []
        for _ in range(2):
            task = runtime.store.receive(
                cid,
                str(uuid4()),
                "搜索全球新闻",
                {"intent": "NEW_TASK", "goal": "搜索新闻", "constraints": {"范围": "全球"}},
                {},
            )["task_id"]
            runtime.store.control(task, cid, str(uuid4()), "PAUSE")
            task_ids.append(task)
        return task_ids

    task_ids = client.portal.call(seed)
    result = client.post(
        f"/api/conversations/{cid}/messages",
        json={
            "client_message_id": str(uuid4()),
            "content": "只看科技",
        },
    )
    assert result.status_code == 202
    assert result.json()["intent"] == "CLARIFY"
    assert set(result.json()["candidates"]) == set(task_ids)
    assert all(
        client.portal.call(runtime.store.task, t)["constraints"] == {"范围": "全球"}
        for t in task_ids
    )
    for bad in [
        {"content": " "},
        {"content": "x", "system": "untrusted"},
        {"content": "x", "target_task_id": "not-a-uuid"},
    ]:
        assert (
            client.post(
                f"/api/conversations/{cid}/messages",
                json={
                    "client_message_id": str(uuid4()),
                    **bad,
                },
            ).status_code
            == 422
        )


def test_control_and_status_latency_without_model_or_task_slots(client):
    import time

    runtime = client.app.state.interaction
    cid = str(uuid4())

    def seed():
        tid = runtime.store.receive(
            cid, str(uuid4()), "搜索新闻", {"intent": "NEW_TASK", "goal": "搜索新闻"}, {}
        )["task_id"]
        runtime.store.control(tid, cid, str(uuid4()), "PAUSE")
        return tid

    tid = client.portal.call(seed)
    durations = {"control": [], "status": []}
    for _ in range(20):
        start = time.perf_counter()
        response = client.post(
            f"/api/tasks/{tid}/controls",
            json={
                "conversation_id": cid,
                "control_id": str(uuid4()),
                "type": "STEER",
                "content": "只看科技",
            },
        )
        durations["control"].append((time.perf_counter() - start) * 1000)
        assert response.status_code == 202
        start = time.perf_counter()
        assert client.get(f"/api/tasks/{tid}", params={"conversation_id": cid}).status_code == 200
        durations["status"].append((time.perf_counter() - start) * 1000)
    p95 = {name: sorted(samples)[18] for name, samples in durations.items()}
    assert all(value < 300 for value in p95.values()), p95
