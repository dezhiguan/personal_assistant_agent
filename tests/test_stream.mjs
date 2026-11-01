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
  assert.equal(duration(null), '未知');
  assert.equal(duration(1234), '1.23 秒');
});

test('billing labels distinguish unknown, local API-free and nested costs', () => {
  assert.match(costLabel({ kind: 'tool', tool_cost: { status: 'unknown', amount: null } }), /未知/);
  assert.match(costLabel({ kind: 'tool', tool_cost: { status: 'no_external_fee', amount: 0 } }), /无外部 API 费用/);
  assert.match(costLabel({ kind: 'tool', tool_cost: { status: 'included' } }), /不重复计费/);
  assert.match(costLabel({ kind: 'model', cost_estimate: { total_cost_cny: 0.000001 } }), /¥0.000001（估算）/);
});
