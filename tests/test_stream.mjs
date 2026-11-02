import test from 'node:test';
import assert from 'node:assert/strict';
import { parseEvents, streamChat } from '../src/personal_assistant_agent/static/js/api.js';
import { latestTodos, normalizeTodos } from '../src/personal_assistant_agent/static/js/planning.js';
import { applyTraceEvent, beginTrace, costLabel, duration, endTrace, summarize } from '../src/personal_assistant_agent/static/js/timeline.js';

test('NDJSON parser handles Chinese split across byte boundaries and incomplete lines', async () => {
  const bytes = new TextEncoder().encode('{"type":"delta","text":"你好"}\n{"type":"done"}');
  const stream = new ReadableStream({ start(controller) {
    for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
    controller.close();
  } });
  const events = [];
  for await (const event of parseEvents(stream)) events.push(event);
  assert.deepEqual(events, [{ type: 'delta', text: '你好' }, { type: 'done' }]);
});

test('NDJSON parser rejects malformed stream instead of showing it as a model answer', async () => {
  const stream = new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode('invalid json\n'));
    controller.close();
  } });
  await assert.rejects(async () => { for await (const event of parseEvents(stream)) void event; });
});

test('old server rejecting todos prompts a restart instead of blaming message format', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    detail: [{ type: 'extra_forbidden', loc: ['body', 'todos'], input: [] }],
  }), { status: 422 }));
  await assert.rejects(async () => {
    for await (const event of streamChat({ messages: [], todos: [] })) void event;
  }, /旧版聊天接口.*重启 Python 服务/);
});

test('plan events preserve revisions and clearing across byte boundaries', async () => {
  const events = [
    { type: 'plan', todos: [{ content: '整理需求', status: 'in_progress' }] },
    { type: 'plan', todos: [{ content: '整理需求', status: 'completed' }] },
    { type: 'plan', todos: [] },
    { type: 'done' },
  ];
  const bytes = new TextEncoder().encode(events.map(e => JSON.stringify(e)).join('\n'));
  const stream = new ReadableStream({ start(controller) {
    for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
    controller.close();
  } });
  const actual = [];
  for await (const event of parseEvents(stream)) actual.push(event);
  assert.deepEqual(actual, events);
});

test('tool progress coexists with plan, delta and done events', async () => {
  const events = [
    { type: 'plan', todos: [{ content: '创建文件', status: 'in_progress' }] },
    { type: 'tool', name: 'write_file', call_id: 'create', status: 'running' },
    { type: 'tool', name: 'write_file', call_id: 'create', status: 'finished' },
    { type: 'delta', text: '文件已写入。' },
    { type: 'done' },
  ];
  const stream = new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode(events.map(e => JSON.stringify(e)).join('\n')));
    controller.close();
  } });
  const actual = [];
  for await (const event of parseEvents(stream)) actual.push(event);
  assert.deepEqual(actual, events);
});

test('follow-ups restore interrupted plans; retry and empty plans do not revive stale state', () => {
  const todos = [{ content: '整理需求', status: 'in_progress' }];
  const stopped = { role: 'assistant', state: 'stopped', todos };
  const failed = { role: 'assistant', state: 'error', todos: [{ content: '失败的计划', status: 'pending' }] };
  assert.deepEqual(latestTodos([stopped, failed], failed), todos);
  const restored = latestTodos([stopped, failed]);
  restored[0].status = 'completed';
  assert.equal(todos[0].status, 'in_progress');
  assert.deepEqual(latestTodos([stopped, { role: 'assistant', todos: [] }]), []);
  assert.deepEqual(latestTodos([]), []);
  assert.deepEqual(normalizeTodos([null, { content: 'x', status: 'unknown' }, { content: ' ', status: 'pending' }]), []);
});

test('timeline upserts completed spans without duplicates and keeps start order', () => {
  const message = {};
  beginTrace(message, '测试问题');
  applyTraceEvent(message, { type: 'meta', timeline_version: 1, started_at: '2026-11-01T00:00:00Z' });
  const first = { id: 'model', kind: 'model', sequence: 1, status: 'running' };
  const second = { id: 'tool', kind: 'tool', sequence: 2, status: 'finished',
    tool_cost: { status: 'unknown', amount: null } };
  applyTraceEvent(message, { type: 'trace', span: first });
  applyTraceEvent(message, { type: 'trace', span: second });
  applyTraceEvent(message, { type: 'trace', span: { ...first, status: 'finished',
    usage: { input_tokens: 100, output_tokens: 20, total_tokens: 120 },
    cost_estimate: { total_cost_cny: 0.00012 } } });
  assert.equal(message.trace.spans.length, 2);
  assert.deepEqual(message.trace.spans.map(s => s.id), ['model', 'tool']);
  const summary = summarize(message.trace);
  assert.equal(summary.input_tokens, 100);
  assert.equal(summary.output_tokens, 20);
  assert.equal(summary.total_tokens, 120);
  assert.equal(summary.model_cost_estimate_cny, 0.00012);
  assert.equal(summary.cost_complete, false);
  assert.equal(summary.unknown_tool_costs, 1);
});

test('stopping and restoring interrupted traces never fabricates a measured duration', () => {
  const message = {};
  beginTrace(message, '停止测试');
  applyTraceEvent(message, { type: 'trace', span: { id: 'a', kind: 'model', sequence: 1,
    status: 'running', duration_ms: null } });
  endTrace(message, 'stopped');
  assert.equal(message.trace.spans[0].duration_ms, null);
  assert.equal(message.trace.spans[0].status, 'stopped');
  assert.equal(summarize(message.trace).usage_complete, false);
  assert.equal(summarize(message.trace).input_tokens, null);
  assert.equal(summarize(message.trace).output_tokens, null);
  assert.equal(summarize(message.trace).model_cost_complete, false);
  const restored = JSON.parse(JSON.stringify(message));
  assert.equal(costLabel(restored.trace.spans[0]), '未知');
});

test('server wall duration takes precedence over browser receipt or overlapping span durations', () => {
  const message = {};
  beginTrace(message, '并行测试');
  applyTraceEvent(message, { type: 'done', trace_summary: { duration_ms: 1234,
    total_tokens: 321, model_calls: 2 } });
  endTrace(message, 'complete');
  assert.equal(message.trace.elapsed, 1234);
  assert.equal(summarize(message.trace).total_tokens, 321);
  assert.equal(summarize(message.trace).input_tokens, null);
  assert.equal(summarize(message.trace).output_tokens, null);
  assert.equal(duration(null), '未知');
  assert.equal(duration(1234), '1.23 秒');
});

test('billing labels distinguish unknown, local API-free and nested costs', () => {
  assert.match(costLabel({ kind: 'tool', tool_cost: { status: 'unknown', amount: null } }), /未知/);
  assert.match(costLabel({ kind: 'tool', tool_cost: { status: 'no_external_fee', amount: 0 } }), /无外部 API 费用/);
  assert.match(costLabel({ kind: 'tool', tool_cost: { status: 'included' } }), /不重复计费/);
  assert.match(costLabel({ kind: 'model', cost_estimate: { total_cost_cny: 0.000001 } }), /¥0.000001（估算）/);
});

test('duplex events keep task results and ordinary replies separate and deduplicate replay', async () => {
  const { reduceEvent } = await import('../src/personal_assistant_agent/static/js/duplex.js');
  const session = { messages: [] };
  const event = (sequence, type, payload, extra = {}) => ({ sequence, type, payload, ...extra });
  reduceEvent(session, event(1, 'task.created', { id: 'A', status: 'RUNNING', revision: 1 }, { task_id: 'A' }));
  reduceEvent(session, event(2, 'chat.delta', { text: '普通聊天' }, { chat_message_id: 'chat-1' }));
  reduceEvent(session, event(3, 'control.received', {}, { task_id: 'A' }));
  assert.match(session.tasks.A.controlNote, /已收到/);
  reduceEvent(session, event(4, 'control.applied', { directive_revision: 1 }, { task_id: 'A' }));
  assert.match(session.tasks.A.controlNote, /已应用/);
  const final = event(5, 'task.result', { id: 'A', revision: 1, status: 'SUCCEEDED', result: '国内科技新闻' }, { task_id: 'A' });
  reduceEvent(session, final);
  assert.equal(reduceEvent(session, final), false);
  assert.equal(reduceEvent(session, event(1, 'task.state_changed', { status: 'RUNNING' }, { task_id: 'A' })), false);
  assert.equal(session.messages.find(m => m.replyTo === 'chat-1').text, '普通聊天');
  assert.equal(session.messages.find(m => m.replyTo === 'task:A').text, '国内科技新闻');
  assert.equal(session.tasks.A.status, 'SUCCEEDED');
  reduceEvent(session, event(6, 'chat.settled', { text: '独立回答', error: null }, { chat_message_id: 'chat-1' }));
  assert.equal(session.messages.find(m => m.replyTo === 'chat-1').state, 'complete');
});

test('an earlier applied revision cannot acknowledge newer pending requirements', async () => {
  const { reduceEvent } = await import('../src/personal_assistant_agent/static/js/duplex.js');
  const session = { messages: [], tasks: { A: { revision: 3, applied_revision: 0 } } };
  reduceEvent(session, { sequence: 1, type: 'control.applied', task_id: 'A', payload: { directive_revision: 1 } });
  assert.equal(session.tasks.A.applied_revision, 1);
  assert.match(session.tasks.A.controlNote, /等待应用/);
});

test('reconnecting restores active task and chat states without cancelling either', async t => {
  const { Duplex } = await import('../src/personal_assistant_agent/static/js/duplex.js');
  const session = { id: 'conversation', messages: [
    { id: 'task-answer', role: 'assistant', replyTo: 'task:A', text: '', state: 'stopped', trace: { state: 'stopped' } },
    { id: 'chat-answer', role: 'assistant', replyTo: 'chat-A', text: '', state: 'stopped', trace: { state: 'stopped' } },
  ] };
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    tasks: [{ id: 'A', status: 'RUNNING' }], cursor: 42,
    messages: [{ id: 'chat-A', role: 'user', kind: 'CHAT', content: '什么是科技', response: '{"pending":true}' }],
  })));
  const previous = globalThis.EventSource;
  globalThis.EventSource = class { close() {} };
  try {
    const duplex = new Duplex(() => {}, () => {});
    await duplex.connect(session);
    assert.equal(session.messages[0].state, 'streaming');
    assert.equal(session.messages[0].trace.state, 'running');
    assert.equal(session.messages[1].state, 'streaming');
    assert.equal(session.messages[1].trace.state, 'running');
    assert.equal(session.eventCursor, 42);
    duplex.disconnect();
  } finally { globalThis.EventSource = previous; }
});

test('task timeline retains chronological order when callbacks restart on resume', async () => {
  const { Duplex } = await import('../src/personal_assistant_agent/static/js/duplex.js');
  const duplex = new Duplex(() => {}, applyTraceEvent);
  const answer = { taskId: 'A', replyTo: 'task:A' };
  for (const span of [
    { id: 'first-model', sequence: 1, started_at: '2030-01-01T00:00:00Z' },
    { id: 'tool', sequence: 2, started_at: '2030-01-01T00:00:01Z' },
    { id: 'resumed-model', sequence: 1, started_at: '2030-01-01T00:05:00Z' },
  ]) duplex.trace(answer, { type: 'trace', span });
  assert.deepEqual(answer.trace.spans.map(s => s.id), ['first-model', 'tool', 'resumed-model']);
  assert.deepEqual(answer.trace.spans.map(s => s.sequence), [1, 2, 3]);
});

test('duplex and the application share the same timeline clock module', async () => {
  const { Duplex } = await import('../src/personal_assistant_agent/static/js/duplex.js');
  const { elapsed } = await import('../src/personal_assistant_agent/static/js/timeline.js?v=2');
  const answer = { replyTo: 'chat-A' };
  const duplex = new Duplex(() => {}, applyTraceEvent);
  duplex.trace(answer, { type: 'meta', timeline_version: 1 });
  assert.equal(answer.trace.state, 'running');
  assert.ok(Number.isFinite(elapsed(answer.trace)));
});

test('task completion appears after intervening steering messages, once', async () => {
  const { reduceEvent } = await import('../src/personal_assistant_agent/static/js/duplex.js');
  const placeholder = { id: 'task-answer', replyTo: 'task:A', role: 'assistant', state: 'streaming', text: '' };
  const session = { messages: [placeholder, { role: 'user', text: '只看科技' }, { role: 'assistant', text: '已收到' }] };
  const event = { sequence: 1, type: 'task.result', task_id: 'A', payload: { result: '最新国内科技新闻', status: 'SUCCEEDED' } };
  assert.equal(reduceEvent(session, event), true);
  assert.equal(session.messages.at(-1), placeholder);
  assert.equal(placeholder.taskResult, true);
  assert.equal(reduceEvent(session, event), false);
  assert.equal(session.messages.filter(m => m.replyTo === 'task:A').length, 1);
});

test('an empty browser restores durable message and result order from the server', async t => {
  const { Duplex } = await import('../src/personal_assistant_agent/static/js/duplex.js');
  const session = { id: 'conversation', messages: [] };
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    tasks: [{ id: 'A', status: 'SUCCEEDED', result: '国内新闻' }], cursor: 42,
    messages: [
      { id: 'new', role: 'user', kind: 'NEW_TASK', content: '搜索全球新闻' },
      { id: 'steer', role: 'user', kind: 'STEER', content: '只看国内' },
      { id: 'result:A', role: 'assistant', kind: 'RESULT', content: '国内新闻' },
      { id: 'chat', role: 'user', kind: 'CHAT', content: '谢谢', response: '{"pending":false}' },
      { id: 'chat:chat', role: 'assistant', kind: 'CHAT', content: '不用谢' },
    ],
  })));
  const previous = globalThis.EventSource;
  globalThis.EventSource = class { close() {} };
  try {
    const duplex = new Duplex(() => {}, () => {});
    await duplex.connect(session);
    assert.deepEqual(session.messages.map(m => m.text), ['搜索全球新闻', '只看国内', '国内新闻', '谢谢', '不用谢']);
    await duplex.connect(session);
    assert.equal(session.messages.filter(m => m.replyTo === 'task:A').length, 1);
    assert.equal(session.messages.at(-1).text, '不用谢');
    duplex.disconnect();
  } finally { globalThis.EventSource = previous; }
});
