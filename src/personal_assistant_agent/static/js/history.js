// Preserve live object references; task bodies are authoritative on the server.
export function mergeSessions(local, stored, hidden = []) {
  const deleted = new Set(hidden);
  const known = new Set(local.map(s => s.id));
  return [...local, ...stored.filter(s => s && typeof s.id === 'string'
    && typeof s.title === 'string' && Array.isArray(s.messages) && !known.has(s.id))]
    .filter(s => !deleted.has(s.id));
}

export function discoverSessions(local, directory, hidden = []) {
  const known = new Set(local.map(s => s.id));
  const deleted = new Set(hidden);
  return [...directory.filter(s => !known.has(s.id) && !deleted.has(s.id))
    .map(s => ({ ...s, messages: [] })), ...local];
}
