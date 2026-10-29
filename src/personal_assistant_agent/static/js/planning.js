/** Plans are conversation data, stored alongside each assistant response. */
const labels = { pending: '待办', in_progress: '进行中', completed: '已完成' };

export function normalizeTodos(value) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 32)
    .filter(item => item && typeof item.content === 'string' && item.content.trim()
      && item.content.length <= 500 && Object.hasOwn(labels, item.status))
    .map(({ content, status }) => ({ content, status }));
}

export function latestTodos(messages, currentMessage) {
  const previous = messages.findLast(message => message !== currentMessage
    && message.role === 'assistant' && message.state !== 'error' && Array.isArray(message.todos));
  return normalizeTodos(previous?.todos);
}

export function renderPlan(container, message) {
  const todos = normalizeTodos(message.todos);
  container.replaceChildren();
  container.hidden = !todos.length;
  if (!todos.length) return;
  const complete = todos.filter(item => item.status === 'completed').length;
  const heading = document.createElement('div');
  heading.className = 'plan-heading';
  const title = document.createElement('span');
  title.textContent = '任务计划';
  const count = document.createElement('span');
  count.className = 'plan-count';
  count.textContent = `${complete} / ${todos.length} 已完成`;
  heading.append(title, count);
  const progress = document.createElement('progress');
  progress.max = todos.length;
  progress.value = complete;
  progress.setAttribute('aria-label', '任务完成进度');
  const list = document.createElement('ol');
  for (const item of todos) {
    const row = document.createElement('li');
    row.dataset.status = item.status;
    const text = document.createElement('span');
    text.textContent = item.content;
    const status = document.createElement('span');
    status.className = 'plan-status';
    status.textContent = item.status === 'in_progress' && message.state !== 'streaming'
      ? '待继续' : labels[item.status];
    row.append(text, status);
    list.append(row);
  }
  container.append(heading, progress, list);
}
