import { getStatus, streamChat } from './api.js';
import { renderMarkdown } from './markdown.js';
import { latestTodos, normalizeTodos, renderPlan } from './planning.js';
import { TimelinePanel, applyTraceEvent, beginTrace, elapsed, endTrace } from './timeline.js';

const $ = id => document.getElementById(id);
const STORAGE_KEY = 'yiyu.assistant.v2';
let sessions = [];
let currentId = null;
let job = null;
let toastTimer;
let tracingEnabled = false;
const timeline = new TimelinePanel();

function showTimeline() {
  $('app').classList.remove('trace-collapsed');
  $('app').classList.add('trace-open');
  $('trace-toggle').setAttribute('aria-expanded', 'true');
}

function hideTimeline() {
  $('app').classList.add('trace-collapsed');
  $('app').classList.remove('trace-open');
  $('trace-toggle').setAttribute('aria-expanded', 'false');
}

const wideTimeline = window.matchMedia('(min-width: 1200px)');
function updateTimelineExpanded() {
  const open = wideTimeline.matches ? !$('app').classList.contains('trace-collapsed')
    : $('app').classList.contains('trace-open');
  $('trace-toggle').setAttribute('aria-expanded', String(open));
}
wideTimeline.addEventListener('change', updateTimelineExpanded);
updateTimelineExpanded();

function toast(text) {
  clearTimeout(toastTimer);
  $('toast').textContent = text;
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2500);
}

function loadHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    let renamed = false;
    sessions = (Array.isArray(saved.sessions) ? saved.sessions : [])
      .filter(s => s && typeof s.id === 'string' && typeof s.title === 'string' && Array.isArray(s.messages))
      .map(s => ({ ...s, messages: s.messages.filter(m => m && ['user', 'assistant'].includes(m.role) && typeof m.text === 'string').map(m => {
        const text = m.role === 'assistant' ? m.text.replaceAll('一隅', '个人智能助理') : m.text;
        renamed ||= text !== m.text;
        if (m.trace?.state === 'running') endTrace(m, 'stopped');
        return { ...m, text, ...(Array.isArray(m.todos) ? { todos: normalizeTodos(m.todos) } : {}), state: m.state === 'streaming' ? 'stopped' : m.state };
      }) }));
    currentId = sessions.some(s => s.id === saved.currentId) ? saved.currentId : null;
    if (renamed) persist();
  } catch {
    $('storage-note').textContent = '浏览器存储不可用，本次仅临时保留';
  }
}

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessions, currentId }));
    $('storage-note').textContent = '记录保存在此浏览器';
  } catch {
    $('storage-note').textContent = '记录保存失败，本次对话暂存于页面';
  }
}

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS(svg.namespaceURI, 'use');
  svg.setAttribute('aria-hidden', 'true');
  use.setAttribute('href', '#i-' + name);
  svg.append(use);
  return svg;
}

function activeSession() { return sessions.find(s => s.id === currentId); }
function scrollBottom() { $('chat-scroll').scrollTop = $('chat-scroll').scrollHeight; }

function closeMenu() {
  $('app').classList.remove('menu-open');
  $('menu-toggle').setAttribute('aria-expanded', 'false');
}

function renderHistory() {
  $('history').replaceChildren();
  if (!sessions.length) {
    const empty = document.createElement('p');
    empty.className = 'history-empty';
    empty.textContent = '暂无对话';
    $('history').append(empty);
  }
  for (const session of sessions) {
    const button = document.createElement('button');
    button.className = 'history-item' + (session.id === currentId ? ' selected' : '');
    button.title = session.title;
    const label = document.createElement('span');
    label.textContent = session.title;
    button.append(icon('chat'), label);
    button.addEventListener('click', () => {
      stopGeneration();
      currentId = session.id;
      $('message-input').value = '';
      resizeInput();
      closeMenu();
      renderMessages();
      renderHistory();
      updateSend();
      persist();
    });
    $('history').append(button);
  }
}

function thinkingTime(message) {
  const ms = message.trace ? elapsed(message.trace) : null;
  return Number.isFinite(ms) ? `已等待 ${(ms / 1000).toFixed(1)} 秒` : '';
}

function thinking(message) {
  const wrapper = document.createElement('div');
  wrapper.className = 'thinking';
  wrapper.setAttribute('role', 'status');
  wrapper.setAttribute('aria-label', '正在等待 Qwen 回复');
  for (let i = 0; i < 3; i++) wrapper.append(document.createElement('i'));
  const label = document.createElement('span');
  label.className = 'pending-label';
  label.textContent = '正在思考…';
  const timer = document.createElement('span');
  timer.className = 'thinking-time';
  timer.setAttribute('role', 'timer');
  timer.setAttribute('aria-live', 'off');
  timer.textContent = thinkingTime(message);
  wrapper.append(label, timer);
  return wrapper;
}

function renderMessageTiming(row, message) {
  if (message.state === 'streaming' || !message.trace) return;
  const measured = message.trace.summary?.duration_ms;
  const ms = Number.isFinite(measured) ? measured : elapsed(message.trace);
  if (!Number.isFinite(ms) || ms < 0) return;
  const timing = document.createElement('div');
  timing.className = 'message-state message-timing';
  timing.dataset.durationMs = String(ms);
  timing.dataset.source = Number.isFinite(measured) ? 'server' : 'browser';
  timing.textContent = Number.isFinite(measured)
    ? `本轮耗时 ${(ms / 1000).toFixed(2)} 秒`
    : `已等待 ${(ms / 1000).toFixed(2)} 秒（浏览器计时）`;
  timing.title = Number.isFinite(measured)
    ? '服务端实测，包含模型、工具调用及等待时间。'
    : '未收到服务端最终耗时，显示浏览器实际等待时间。';
  row.append(timing);
}

function renderAssistant(row, message, session) {
  row.replaceChildren();
  const heading = document.createElement('div');
  heading.className = 'assistant-heading';
  const name = document.createElement('span');
  name.textContent = '个人智能助理';
  heading.append(icon('mark'), name);
  const body = document.createElement('div');
  body.className = 'assistant-body';
  const plan = document.createElement('section');
  plan.className = 'task-plan';
  plan.setAttribute('aria-label', '任务计划');
  renderPlan(plan, message);
  if (!message.text && message.state === 'streaming') body.append(thinking(message));
  else renderMarkdown(body, message.text);
  row.append(heading, plan, body);
  if (message.toolProgress && message.state === 'streaming') {
    const progress = document.createElement('div');
    progress.className = 'message-state tool-progress';
    progress.setAttribute('role', 'status');
    progress.textContent = message.toolProgress;
    row.append(progress);
  }
  if (message.state === 'stopped') {
    const state = document.createElement('div');
    state.className = 'message-state';
    state.textContent = '已停止生成';
    row.append(state);
  }
  if (message.state === 'error') {
    const error = document.createElement('div');
    error.className = 'message-error';
    error.setAttribute('role', 'alert');
    const description = document.createElement('div');
    description.textContent = message.error || '回复未完成，请重试。';
    error.append(description);
    if (message.requestId) {
      const request = document.createElement('span');
      request.className = 'request-id';
      request.textContent = '请求编号：' + message.requestId;
      error.append(request);
    }
    if (session.messages.at(-1) === message) {
      const retry = document.createElement('button');
      retry.className = 'retry';
      retry.textContent = '重新生成';
      retry.disabled = !!job;
      retry.addEventListener('click', () => retryMessage(session, message));
      error.append(retry);
    }
    row.append(error);
  }
  renderMessageTiming(row, message);
  const hasTrace = (message.tracingEnabled ?? tracingEnabled) && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(message.requestId || '');
  if ((message.text || hasTrace || message.trace) && message.state !== 'streaming') {
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    if (message.text) {
      const copy = document.createElement('button');
      copy.className = 'copy';
      copy.setAttribute('aria-label', '复制这条回复');
      copy.append(icon('copy'), document.createTextNode('复制'));
      copy.addEventListener('click', () => copyText(body.innerText));
      actions.append(copy);
    }
    if (hasTrace) {
      const trace = document.createElement('a');
      trace.className = 'trace-link';
      trace.href = '/api/traces/' + encodeURIComponent(message.requestId);
      trace.target = '_blank';
      trace.rel = 'noopener noreferrer';
      trace.setAttribute('aria-label', '查看这条回复的 LangSmith 链路追踪（新标签页）');
      trace.title = '在 LangSmith 中查看本次请求的链路追踪';
      trace.append(icon('external'), document.createTextNode('LangSmith 追踪'));
      actions.append(trace);
    }
    if (message.trace) {
      const button = document.createElement('button');
      button.className = 'copy';
      button.textContent = '查看调用链路';
      button.addEventListener('click', () => { timeline.select(session, message); showTimeline(); });
      actions.append(button);
    }
    row.append(actions);
  }
}

function renderMessages() {
  const session = activeSession();
  timeline.render(session);
  const hasMessages = !!session?.messages.length;
  $('welcome').hidden = hasMessages;
  $('messages').hidden = !hasMessages;
  $('messages').replaceChildren();
  if (!hasMessages) return;
  for (const message of session.messages) {
    const row = document.createElement('article');
    row.className = 'message ' + message.role;
    row.dataset.messageId = message.id;
    if (message.role === 'user') {
      const bubble = document.createElement('div');
      bubble.className = 'user-bubble';
      bubble.textContent = message.text;
      row.append(bubble);
    } else renderAssistant(row, message, session);
    $('messages').append(row);
  }
  scrollBottom();
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('回复已复制');
  } catch { toast('暂时无法自动复制，请选择文字复制'); }
}

function resizeInput() {
  const input = $('message-input');
  input.style.height = 'auto';
  input.style.height = Math.min(140, Math.max(44, input.scrollHeight)) + 'px';
}

function updateSend() {
  const button = $('send');
  button.disabled = !job && !$('message-input').value.trim();
  button.classList.toggle('running', !!job);
  button.setAttribute('aria-label', job ? '停止生成' : '发送消息');
  button.title = job ? '停止生成' : '发送消息';
  button.replaceChildren();
  if (job) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 12 12');
    const rect = document.createElementNS(svg.namespaceURI, 'rect');
    rect.setAttribute('width', '12');
    rect.setAttribute('height', '12');
    rect.setAttribute('rx', '2');
    svg.append(rect);
    button.append(svg);
  } else button.append(icon('send'));
  $('input-note').textContent = job ? '正在生成回复 · 点击右侧可停止' : 'Enter 发送 · Shift + Enter 换行';
}

function finish(running) {
  if (job !== running) return;
  clearInterval(running.thinkingTimer);
  job = null;
  endTrace(running.message, running.message.state);
  persist();
  updateSend();
  if (currentId === running.session.id) {
    timeline.render(running.session);
    const row = document.querySelector(`[data-message-id="${running.message.id}"]`);
    if (row) renderAssistant(row, running.message, running.session);
  }
}

function stopGeneration() {
  if (!job) return;
  const running = job;
  running.cancelled = true;
  running.controller.abort();
  running.message.state = 'stopped';
  finish(running);
}

async function runConversation(session, message) {
  const todos = latestTodos(session.messages, message);
  message.todos = todos;
  const question = session.messages.slice(0, session.messages.indexOf(message)).findLast(m => m.role === 'user')?.text || '';
  beginTrace(message, question);
  timeline.select(session, message);
  const running = { session, message, controller: new AbortController(), cancelled: false };
  job = running;
  running.thinkingTimer = setInterval(() => {
    if (job !== running || currentId !== session.id) return;
    const timer = document.querySelector(`[data-message-id="${message.id}"] .thinking-time`);
    if (timer) timer.textContent = thinkingTime(message);
  }, 200);
  renderMessages();
  updateSend();
  persist();
  try {
    const messages = session.messages
      .filter(m => m !== message && m.state !== 'error' && m.text.trim())
      .map(m => ({ role: m.role, content: m.text }));
    if (messages.length > 60 || messages.reduce((n, m) => n + m.content.length, 0) > 80000) {
      throw new Error('这段对话较长，请开启新对话后继续。');
    }
    let done = false;
    for await (const event of streamChat({ conversation_id: session.id, messages, todos }, running.controller.signal)) {
      if (running.cancelled || job !== running) return;
      applyTraceEvent(message, event);
      if (event.type === 'meta') {
        message.requestId = event.request_id;
        message.tracingEnabled = !!event.tracing_enabled;
        timeline.render(session);
      } else if (event.type === 'trace') {
        timeline.render(session);
        persist();
      } else if (event.type === 'usage') {
        message.usage = event;
      } else if (event.type === 'plan') {
        const viewport = $('chat-scroll');
        const nearBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 100;
        message.todos = normalizeTodos(event.todos);
        const plan = document.querySelector(`[data-message-id="${message.id}"] .task-plan`);
        if (plan) renderPlan(plan, message);
        persist();
        if (nearBottom) scrollBottom();
      } else if (event.type === 'tool') {
        const states = { running: '正在调用', finished: '已返回', error: '调用失败' };
        message.toolProgress = `${states[event.status] || '调用工具'}：${event.name}`;
        const row = document.querySelector(`[data-message-id="${message.id}"]`);
        if (row) renderAssistant(row, message, session);
      } else if (event.type === 'delta') {
        const viewport = $('chat-scroll');
        const nearBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 100;
        message.text += event.text;
        const body = document.querySelector(`[data-message-id="${message.id}"] .assistant-body`);
        if (body) renderMarkdown(body, message.text);
        if (nearBottom) scrollBottom();
      } else if (event.type === 'error') {
        message.requestId = event.request_id;
        throw new Error(event.message);
      } else if (event.type === 'done') {
        done = true;
        message.state = 'complete';
        delete message.toolProgress;
      }
    }
    if (!done) throw new Error('回复连接中断，请重新生成。');
  } catch (error) {
    delete message.toolProgress;
    if (!running.cancelled) {
      message.state = 'error';
      message.error = error instanceof TypeError ? '网络连接中断，请检查服务后重试。' : error.message;
    }
  } finally { finish(running); }
}

function retryMessage(session, message) {
  if (job || session.messages.at(-1) !== message) return;
  message.text = '';
  message.error = '';
  message.state = 'streaming';
  message.requestId = null;
  delete message.todos;
  delete message.toolProgress;
  delete message.usage;
  runConversation(session, message);
}

function submitMessage() {
  if (job) { stopGeneration(); return; }
  const text = $('message-input').value.trim();
  if (!text) return;
  let session = activeSession();
  if (!session) {
    session = { id: crypto.randomUUID(), title: text.slice(0, 24), messages: [] };
    sessions.unshift(session);
    currentId = session.id;
  }
  session.messages.push({ id: crypto.randomUUID(), role: 'user', text, state: 'complete' });
  const answer = { id: crypto.randomUUID(), role: 'assistant', text: '', state: 'streaming' };
  session.messages.push(answer);
  $('message-input').value = '';
  resizeInput();
  renderHistory();
  runConversation(session, answer);
}

function newChat() {
  stopGeneration();
  currentId = null;
  $('message-input').value = '';
  resizeInput();
  closeMenu();
  renderMessages();
  renderHistory();
  updateSend();
  persist();
  $('message-input').focus();
}

async function refreshStatus() {
  try {
    const status = await getStatus();
    const tracingChanged = tracingEnabled !== !!status.tracing_enabled;
    tracingEnabled = !!status.tracing_enabled;
    if (tracingChanged) renderMessages();
    $('connection-note').hidden = status.model_configured;
    $('connection-note').textContent = 'Qwen 尚未就绪，请先完成服务端模型配置。';
  } catch {
    $('connection-note').hidden = false;
    $('connection-note').textContent = '暂时无法连接服务，请确认 Python 服务正在运行后刷新页面。';
  }
}

$('composer').addEventListener('submit', event => { event.preventDefault(); submitMessage(); });
$('message-input').addEventListener('input', () => { resizeInput(); updateSend(); });
$('message-input').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!job) submitMessage();
  }
});
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
  $('message-input').value = button.dataset.prompt;
  resizeInput();
  updateSend();
  $('message-input').focus();
}));
$('new-chat').addEventListener('click', newChat);
$('menu-toggle').addEventListener('click', () => {
  const open = $('app').classList.toggle('menu-open');
  $('menu-toggle').setAttribute('aria-expanded', String(open));
});
$('shade').addEventListener('click', closeMenu);
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });
$('trace-toggle').addEventListener('click', () => {
  const visible = window.matchMedia('(min-width: 1200px)').matches
    ? !$('app').classList.contains('trace-collapsed') : $('app').classList.contains('trace-open');
  if (visible) hideTimeline(); else showTimeline();
});
$('trace-close').addEventListener('click', hideTimeline);
$('trace-shade').addEventListener('click', hideTimeline);
document.addEventListener('keydown', event => { if (event.key === 'Escape') hideTimeline(); });
$('clear-history').addEventListener('click', () => {
  if (!sessions.length) { toast('还没有本机对话记录'); return; }
  if (!confirm('清空此浏览器内的对话记录？此操作无法撤销。')) return;
  stopGeneration();
  sessions = [];
  newChat();
  toast('本机对话记录已清空');
});
window.addEventListener('pagehide', () => { stopGeneration(); persist(); });
loadHistory();
renderHistory();
renderMessages();
updateSend();
refreshStatus();
