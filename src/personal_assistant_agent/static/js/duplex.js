import { beginTrace, endTrace } from './timeline.js?v=2';

/** Durable conversation projection. Each task and ordinary reply owns its own message. */
export const taskLabels = { QUEUED: '排队中', RUNNING: '执行中', PAUSING: '暂停中，等待当前工具结束',
  PAUSED: '已暂停', CANCELLING: '取消中，等待当前工具结算', CANCELLED: '已取消',
  SUCCEEDED: '已完成', FAILED: '执行失败', RECOVERING: '正在核对现场',
  RECOVERY_REQUIRED: '需要核对后恢复', WAITING_INPUT: '等待补充信息' };

export async function requestJSON(url, body) {
  const response = await fetch(url, { method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined, cache: 'no-store' });
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : '请求未被接收');
  return value;
}

export function describeTask(task) {
  const progress = task.progress || {};
  const todos = progress.todos || [];
  const tools = Object.values(progress.active_tools || {}).map(t => t.name);
  if (!tools.length && task.currentTool) tools.push(task.currentTool);
  const lines = [`${task.goal} · ${task.id.slice(0, 8)}：${taskLabels[task.status] || task.status}`];
  if (todos.length) {
    lines.push(`已完成 ${todos.filter(t => t.status === 'completed').length} 项`);
    const current = todos.filter(t => t.status === 'in_progress').map(t => t.content);
    if (current.length) lines.push('当前步骤：' + current.join('；'));
  }
  if (tools.length) lines.push('正在调用：' + tools.join('、'));
  if (task.reason || task.error?.message) lines.push(task.reason || task.error.message);
  if (task.revision > task.applied_revision && !['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(task.status)) lines.push('最新要求等待安全边界应用');
  else if (task.revision > 0) lines.push('最新要求已应用');
  return lines.join('\n');
}

function answerFor(session, key) {
  let answer = session.messages.find(m => m.replyTo === key);
  if (!answer) {
    answer = { id: `answer:${key}`, replyTo: key, role: 'assistant', text: '', state: 'streaming' };
    session.messages.push(answer);
  }
  return answer;
}

export function reduceEvent(session, event, trace = () => {}) {
  if (event.sequence <= (session.eventCursor || 0)) return false;
  session.eventCursor = event.sequence;
  const payload = event.payload;
  session.tasks ||= {};
  if (['task.created', 'task.state_changed', 'task.result'].includes(event.type)) {
    const previous = session.tasks[event.task_id];
    session.tasks[event.task_id] = payload;
    const statusAnswer = answerFor(session, `task:${event.task_id}`);
    statusAnswer.taskId = event.task_id;
    const existing = session.messages.find(m => m.replyTo === `task:${event.task_id}`);
    if (payload.status === 'RUNNING' && existing && previous?.status !== 'RUNNING') {
      existing.state = 'streaming'; existing.text = '';
      if (existing.trace) existing.trace.state = 'running';
    }
    if (['PAUSED', 'WAITING_INPUT', 'FAILED', 'CANCELLED', 'RECOVERY_REQUIRED'].includes(payload.status)) {
      const answer = session.messages.find(m => m.replyTo === `task:${event.task_id}`);
      if (answer) {
        Object.assign(answer, { state: 'complete', text: payload.reason || payload.error?.message || taskLabels[payload.status] });
        if (payload.trace_summary) trace(answer, { type: 'done', trace_summary: payload.trace_summary });
        endTrace(answer, 'complete');
      }
    }
    if (payload.result) {
      const answer = answerFor(session, `task:${event.task_id}`);
      if (!answer.taskResult) {
        session.messages.splice(session.messages.indexOf(answer), 1);
        session.messages.push(answer);
      }
      Object.assign(answer, { text: payload.result, state: 'complete', taskId: event.task_id, taskResult: true });
      if (payload.trace_summary) trace(answer, { type: 'done', trace_summary: payload.trace_summary });
      endTrace(answer, 'complete');
    }
  }
  if (event.type === 'trace' && event.task_id) {
    const answer = answerFor(session, `task:${event.task_id}`);
    answer.taskId = event.task_id;
    answer.requestId = session.tasks[event.task_id]?.attempt_id;
    answer.tracingEnabled = !!session.tasks[event.task_id]?.tracing_enabled;
    trace(answer, payload);
  }
  if (event.type === 'task.usage' && event.task_id) {
    const answer = answerFor(session, `task:${event.task_id}`);
    trace(answer, { type: 'done', trace_summary: payload });
  }
  if (event.type === 'plan.updated' && session.tasks[event.task_id]) {
    const task = session.tasks[event.task_id];
    task.progress = { ...task.progress, todos: payload.todos };
    answerFor(session, `task:${event.task_id}`).todos = payload.todos;
  }
  if (['tool.started', 'tool.finished'].includes(event.type) && session.tasks[event.task_id]) {
    const task = session.tasks[event.task_id];
    task.progress ||= {};
    const active = task.progress.active_tools ||= {};
    for (const [key, value] of Object.entries(active)) {
      if (value.call_id === payload.call_id && JSON.stringify(value.namespace) === JSON.stringify(payload.namespace)) delete active[key];
    }
    if (event.type === 'tool.started') active[JSON.stringify([payload.namespace, payload.call_id])] = payload;
    delete task.currentTool;
  }
  if (event.type === 'control.applied') {
    const task = session.tasks[event.task_id];
    if (task) {
      task.applied_revision = Math.max(task.applied_revision ?? -1, payload.directive_revision);
      task.controlNote = task.applied_revision < task.revision ? '新要求等待应用' : '新要求已应用';
    }
  }
  if (event.type === 'control.received') {
    const task = session.tasks[event.task_id];
    if (task) task.controlNote = '指令已收到，等待执行边界';
  }
  if (event.type.startsWith('chat.')) {
    const answer = answerFor(session, event.chat_message_id);
    trace(answer, payload);
    if (event.type === 'chat.meta') answer.tracingEnabled = !!payload.tracing_enabled;
    if (event.type === 'chat.delta') answer.text += payload.text;
    if (event.type === 'chat.meta') answer.requestId = payload.request_id;
    if (event.type === 'chat.plan') answer.todos = payload.todos;
    if (event.type === 'chat.tool') answer.toolProgress = `${payload.status}：${payload.name}`;
    if (event.type === 'chat.settled') {
      Object.assign(answer, { text: payload.text, state: payload.error ? 'error' : 'complete',
        error: payload.error });
      delete answer.toolProgress;
      delete answer.retryId;
      endTrace(answer, answer.state);
    }
    if (event.type === 'chat.retried') {
      Object.assign(answer, { text: '', state: 'streaming', acceptedIntent: 'CHAT' });
      delete answer.error; delete answer.trace;
    }
  }
  return true;
}

export class Duplex {
  constructor(change, trace) {
    this.change = change; this.source = null; this.epoch = 0;
    this.trace = (answer, event) => {
      if (!answer.trace && (event.type === 'meta' || event.type === 'trace' || event.trace_summary)) {
        beginTrace(answer, this.session?.messages.find(m => m.id === answer.replyTo)?.text
          || this.session?.tasks?.[answer.taskId]?.goal || '后台任务');
      }
      if (answer.trace && (event.type === 'trace' || event.trace_summary)) answer.trace.supported = true;
      trace(answer, event);
      if (answer.taskId && answer.trace) {
        // Callback sequences restart with each native run. Display one stable
        // task timeline across pause/resume, preserving the real start times.
        answer.trace.spans.sort((a, b) => Date.parse(a.started_at) - Date.parse(b.started_at)
          || a.sequence - b.sequence);
        answer.trace.spans.forEach((span, index) => { span.sequence = index + 1; });
      }
    };
  }

  async connect(session) {
    this.session = session;
    this.source?.close();
    this.source = null;
    const epoch = ++this.epoch;
    try {
      const snapshot = await requestJSON(`/api/conversations/${session.id}/tasks`);
      if (epoch !== this.epoch) return;
      session.tasks = Object.fromEntries(snapshot.tasks.map(t => [t.id, t]));
      // Replay persisted rows first so a fresh browser retains commit order.
      for (const row of snapshot.messages) {
        if (row.role === 'user' && !session.messages.some(m => m.id === row.id)) {
          session.messages.push({ id: row.id, role: 'user', text: row.content, state: 'complete' });
        }
        if (row.kind === 'RESULT') {
          const answer = answerFor(session, `task:${row.id.slice(7)}`);
          if (!answer.taskResult) {
            session.messages.splice(session.messages.indexOf(answer), 1);
            session.messages.push(answer);
          }
          Object.assign(answer, { text: row.content, state: 'complete', taskResult: true,
            taskId: row.id.slice(7) });
        }
        if (row.kind === 'CHAT' && row.role === 'assistant') Object.assign(
          answerFor(session, row.id.slice(5)), { text: row.content, state: 'complete' });
        if (row.kind === 'CHAT' && row.role === 'user') {
          const receipt = JSON.parse(row.response);
          if (receipt.pending) {
            const answer = answerFor(session, row.id); answer.state = 'streaming';
            if (answer.trace) answer.trace.state = 'running';
          }
          else if (receipt.error) Object.assign(answerFor(session, row.id), { state: 'error', error: receipt.error });
        }
      }
      for (const task of snapshot.tasks) {
        const taskAnswer = answerFor(session, `task:${task.id}`);
        taskAnswer.taskId = task.id;
        if (taskAnswer && ['RUNNING', 'QUEUED', 'PAUSING', 'CANCELLING'].includes(task.status)) {
          taskAnswer.state = 'streaming';
          if (taskAnswer.trace) taskAnswer.trace.state = 'running';
        }
        if (taskAnswer && !['RUNNING', 'QUEUED', 'PAUSING', 'CANCELLING'].includes(task.status)) {
          taskAnswer.state = 'complete'; taskAnswer.text ||= taskLabels[task.status] || task.status;
        }
        if (task.result) {
          const answer = answerFor(session, `task:${task.id}`);
          if (!answer.taskResult) {
            session.messages.splice(session.messages.indexOf(answer), 1);
            session.messages.push(answer);
          }
          Object.assign(answer, { text: task.result, state: 'complete', taskId: task.id, taskResult: true,
            requestId: task.attempt_id, tracingEnabled: task.tracing_enabled });
          if (task.trace_summary) this.trace(answer, { type: 'done', trace_summary: task.trace_summary });
          endTrace(answer, 'complete');
        }
      }
      session.eventCursor = snapshot.cursor;
      this.change();
      this.source = new EventSource(`/api/conversations/${session.id}/events?after=${snapshot.cursor}`);
      this.source.onmessage = input => {
        if (epoch !== this.epoch) return;
        if (reduceEvent(session, JSON.parse(input.data), this.trace)) this.change();
      };
    } catch (error) { if (epoch === this.epoch) this.change(error.message); }
  }

  disconnect() { this.epoch++; this.source?.close(); this.source = null; }

  async send(session, user, answer, mode = '') {
    answer.replyTo = user.id;
    const payload = { client_message_id: user.id, content: user.text };
    if (mode === 'CHAT' || mode === 'NEW_TASK') payload.intent = mode;
    else if (mode) payload.target_task_id = mode;
    try {
      const ack = await requestJSON(`/api/conversations/${session.id}/messages`, payload);
      answer.acceptedIntent = ack.intent;
      if (ack.intent === 'CHAT') { this.change(); return; }
      answer.state = 'complete';
      if (ack.intent === 'NEW_TASK') {
        answer.text = ack.revision_of ? '已创建关联原成果的修订任务。' : '任务已接收，后台执行期间可以继续聊天或补充要求。';
        answer.taskId = ack.task_id;
      } else if (ack.intent === 'STEER') {
        answer.text = `修正已收到，等待安全执行边界应用。\n\n当前接受的要求：${ack.effective_task_request}`;
      } else if (ack.intent === 'CLARIFY') {
        answer.text = ack.question + (ack.candidates?.length ? '\n\n' + ack.candidates.map((id, i) =>
          `${i + 1}. ${session.tasks?.[id]?.goal || '任务'} · ${id.slice(0, 8)}`).join('\n')
          + '\n\n请直接回复任务名称或编号。' : '');
      } else if (ack.intent === 'STATUS') {
        answer.text = ack.tasks.map(describeTask).join('\n\n') || '当前没有任务。';
      } else answer.text = '控制指令已收到，等待安全执行边界确认；任务状态会在对话中更新。';
      this.change();
    } catch (error) {
      Object.assign(answer, { state: 'error', error: error.message });
      this.change();
    }
  }
}
