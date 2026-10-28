import test from 'node:test';
import assert from 'node:assert/strict';
import { parseEvents } from '../src/personal_assistant_agent/static/js/api.js';

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
