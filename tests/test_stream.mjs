import test from 'node:test';
import assert from 'node:assert/strict';
import { parseEvents, streamChat } from '../src/personal_assistant_agent/static/js/api.js';
import { latestTodos, normalizeTodos } from '../src/personal_assistant_agent/static/js/planning.js';

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
