/** HTTP transport and incremental NDJSON parsing. No provider credentials live here. */
export async function getStatus() {
  const response = await fetch('/healthz', { cache: 'no-store', signal: AbortSignal.timeout(8000) });
  if (!response.ok) throw new Error('服务暂时无法连接');
  return response.json();
}

export async function* parseEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.trim()) yield JSON.parse(line);
      }
      if (done) {
        if (buffer.trim()) yield JSON.parse(buffer);
        break;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function* streamChat(payload, signal) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    if (response.status === 422) throw new Error('消息格式不正确或对话过长，请开启新对话。');
    throw new Error(typeof data.detail === 'string' ? data.detail : '服务暂时不可用，请稍后重试。');
  }
  if (!response.body) throw new Error('浏览器无法读取流式回复，请更换新版浏览器。');
  yield* parseEvents(response.body);
}
