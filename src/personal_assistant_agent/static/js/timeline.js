/** Measured invocation timeline. Missing usage and billing remain unknown. */
export function duration(ms) {
  if (!Number.isFinite(ms)) return '未知';
  if (ms < 1000) return `${Math.round(ms)} 毫秒`;
  return `${(ms / 1000).toFixed(2)} 秒`;
}

const money = value => value > 0 && value < 0.000001 ? '< ¥0.000001' : `¥${value.toFixed(6)}`;
const states = { running: '执行中', finished: '已完成', complete: '已完成', error: '失败', stopped: '已停止' };
const clocks = new WeakMap();

export function beginTrace(message, question) {
  message.trace = { question, startedAt: new Date().toISOString(), spans: [], state: 'running' };
  clocks.set(message.trace, performance.now());
}

export function applyTraceEvent(message, event) {
  const trace = message.trace;
  if (!trace) return;
  if (event.type === 'meta') {
    trace.startedAt = event.started_at || trace.startedAt;
    trace.supported = event.timeline_version === 1;
  } else if (event.type === 'trace') {
    const index = trace.spans.findIndex(span => span.id === event.span.id);
    if (index < 0) trace.spans.push(event.span);
    else trace.spans[index] = event.span;
    trace.spans.sort((a, b) => a.sequence - b.sequence);
  }
  if (event.trace_summary) trace.summary = event.trace_summary;
}

export function endTrace(message, state) {
  const trace = message.trace;
  if (!trace) return;
  trace.elapsed = trace.summary?.duration_ms ?? elapsed(trace);
  trace.state = state;
  for (const span of trace.spans) {
    if (span.status === 'running') {
      span.status = state === 'complete' ? 'stopped' : state;
      // No end callback was received: a browser stop time is not a measured span duration.
      span.duration_ms = null;
    }
  }
}

export function elapsed(trace) {
  if (trace.state !== 'running') return trace.summary?.duration_ms ?? trace.elapsed ?? null;
  const start = clocks.get(trace);
  return start === undefined ? null : performance.now() - start;
}

export function summarize(trace) {
  if (trace.summary) return trace.summary;
  const models = trace.spans.filter(s => s.kind === 'model');
  const tools = trace.spans.filter(s => s.kind === 'tool');
  return {
    total_tokens: models.reduce((sum, s) => sum + (s.usage?.total_tokens || 0), 0),
    usage_complete: models.every(s => s.usage != null),
    measured_model_calls: models.filter(s => s.usage != null).length,
    model_cost_estimate_cny: models.reduce((sum, s) => sum + (s.cost_estimate?.total_cost_cny || 0), 0),
    model_cost_complete: models.every(s => s.cost_estimate != null),
    priced_model_calls: models.filter(s => s.cost_estimate != null).length,
    cost_complete: models.every(s => s.cost_estimate != null) && tools.every(s => s.tool_cost?.status !== 'unknown'),
    unknown_tool_costs: tools.filter(s => s.tool_cost?.status === 'unknown').length,
    model_calls: models.length,
    tool_calls: tools.length,
  };
}

export function costLabel(span) {
  if (span.kind === 'model') return span.cost_estimate
    ? `${money(span.cost_estimate.total_cost_cny)}（估算）` : '未知';
  if (span.tool_cost?.status === 'no_external_fee') return '¥0（无外部 API 费用）';
  if (span.tool_cost?.status === 'included') return '内部调用单列，汇总不重复计费';
  return '未知（未返回调用账单）';
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function timeLabel(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.valueOf()) ? '' : date.toLocaleTimeString('zh-CN', { hour12: false });
}

function line(label, value) {
  const row = el('div', 'trace-field');
  row.append(el('span', '', label), el('span', '', value));
  return row;
}

function step(number, title, status, time) {
  const item = el('li', 'trace-step');
  item.dataset.status = status;
  const marker = el('span', 'trace-marker', String(number).padStart(2, '0'));
  const body = el('div', 'trace-step-body');
  const header = el('div', 'trace-step-heading');
  header.append(el('strong', '', title), el('span', 'trace-state', states[status] || status));
  body.append(header);
  if (time) body.append(el('div', 'trace-time', time));
  item.append(marker, body);
  return { item, body };
}

export class TimelinePanel {
  constructor() {
    this.root = document.getElementById('trace-content');
    this.selector = document.getElementById('trace-turn');
    this.selection = new Map();
    this.selector.addEventListener('change', () => {
      this.selection.set(this.session?.id, this.selector.value);
      this.render(this.session);
    });
    // Update just clocks: expanded details, selection and scroll position remain stable.
    this.timer = setInterval(() => this.tick(), 200);
  }

  select(session, message) {
    this.selection.set(session.id, message.id);
    this.render(session);
  }

  tick() {
    if (!this.trace || this.trace.state !== 'running') return;
    const total = elapsed(this.trace);
    const overall = this.root.querySelector('[data-total-time]');
    if (overall) overall.textContent = duration(total);
    for (const node of this.root.querySelectorAll('[data-span-clock]')) {
      const span = this.trace.spans.find(s => s.id === node.dataset.spanClock);
      if (span) node.textContent = `约 ${duration(Math.max(0, (total ?? 0) - span.offset_ms))} · 进行中`;
    }
  }

  render(session) {
    this.session = session;
    const messages = session?.messages.filter(m => m.role === 'assistant') || [];
    const selected = messages.find(m => m.id === this.selection.get(session?.id)) || messages.at(-1);
    this.selector.replaceChildren();
    for (const [index, message] of messages.entries()) {
      const option = el('option', '', `第 ${index + 1} 轮 · ${(message.trace?.question || '历史对话').slice(0, 32)}`);
      option.value = message.id;
      this.selector.append(option);
    }
    this.selector.hidden = !messages.length;
    if (selected) this.selector.value = selected.id;
    const trace = selected?.trace;
    this.trace = trace;
    const oldDetails = new Set([...this.root.querySelectorAll('details[open]')].map(d => d.dataset.spanId));
    const oldScroll = this.root.scrollTop;
    this.root.replaceChildren();
    if (!trace) {
      this.root.append(el('div', 'trace-empty', messages.length
        ? '这条历史回复没有保存调用链路。发送新问题后，将展示实际调用记录。'
        : '从一个问题开始。模型与工具的实际调用将在这里依次展开。'));
      document.getElementById('trace-status').textContent = '等待提问';
      return;
    }
    document.getElementById('trace-status').textContent = states[trace.state] || trace.state;
    if (trace.supported === false) {
      this.root.append(el('p', 'trace-unknown', '当前 Python 服务尚未加载调用链路接口。请重启服务后刷新页面；此轮耗时、用量和成本无法核验。'));
      return;
    }
    if (trace.supported === undefined && trace.state !== 'running') {
      this.root.append(el('p', 'trace-unknown', '尚未收到服务端调用记录，无法确认本轮用量、成本与耗时。'));
      return;
    }
    const summary = summarize(trace);
    const metrics = el('div', 'trace-metrics');
    const totalTime = el('strong', '', duration(elapsed(trace)));
    totalTime.dataset.totalTime = '';
    const timeMetric = el('div');
    const timeSource = trace.summary?.duration_ms != null ? '总耗时'
      : trace.state === 'running' ? '已等待 · 浏览器计时' : '等待时间 · 浏览器计时';
    timeMetric.append(el('span', '', timeSource), totalTime);
    const tokenMetric = el('div');
    const measuredCalls = summary.measured_model_calls ?? trace.spans.filter(s => s.kind === 'model' && s.usage != null).length;
    tokenMetric.append(el('span', '', summary.usage_complete ? 'Token' : '已知 Token · 不完整'),
      el('strong', '', summary.model_calls > 0 && measuredCalls === 0 ? '未知' : summary.total_tokens.toLocaleString('zh-CN')));
    const costMetric = el('div', 'trace-cost-total');
    const pricedCalls = summary.priced_model_calls ?? trace.spans.filter(s => s.kind === 'model' && s.cost_estimate != null).length;
    costMetric.append(el('span', '', summary.cost_complete ? '总成本 · 估算' : '已知成本 · 估算，不完整'),
      el('strong', '', summary.model_calls > 0 && pricedCalls === 0 ? '未知' : money(summary.model_cost_estimate_cny)));
    metrics.append(timeMetric, tokenMetric, costMetric);
    this.root.append(metrics);
    const count = el('p', 'trace-count', `${summary.model_calls} 次模型调用 · ${summary.tool_calls} 次工具调用`);
    this.root.append(count);
    if (summary.unknown_tool_costs) this.root.append(el('p', 'trace-unknown',
      `${summary.unknown_tool_costs} 次工具调用费用未知，未计入已知成本。`));
    if (trace.state === 'stopped' || trace.state === 'error') this.root.append(el('p', 'trace-unknown',
      '请求未完整结束，未收到的 Token 和费用无法确认。'));
    const list = el('ol', 'trace-timeline');
    const user = step(1, '用户提问', 'finished', timeLabel(trace.startedAt));
    user.body.append(el('p', 'trace-question', trace.question));
    list.append(user.item);
    for (const span of trace.spans) {
      const title = `${span.kind === 'model' ? '模型调用' : '工具调用'} · ${span.name}`;
      const { item, body } = step(span.sequence + 1, title, span.status, timeLabel(span.started_at));
      if (span.parent_tool_id) {
        const parent = trace.spans.find(s => s.id === span.parent_tool_id);
        body.append(el('div', 'trace-parent', `↳ ${parent?.name || '工具'} 内部调用`));
      }
      const timing = line('耗时', duration(span.duration_ms));
      if (span.status === 'running') timing.lastChild.dataset.spanClock = span.id;
      body.append(timing);
      if (span.kind === 'model') {
        body.append(line('输入 Token', span.usage?.input_tokens?.toLocaleString('zh-CN') ?? '未知'),
          line('输出 Token', span.usage?.output_tokens?.toLocaleString('zh-CN') ?? '未知'));
        const cached = span.usage?.input_token_details?.cache_read;
        if (cached !== undefined) body.append(line('缓存命中 Token（含于输入）', cached.toLocaleString('zh-CN')));
      }
      body.append(line('成本', costLabel(span)));
      const details = el('details', 'trace-details');
      details.dataset.spanId = span.id;
      details.open = oldDetails.has(span.id);
      details.append(el('summary', '', span.kind === 'model' ? '查看用量与计费来源' : '查看参数、结果与费用来源'));
      if (span.kind === 'model') {
        details.append(el('p', '', 'Token 来源：模型接口返回的 usage；未返回时显示未知。'));
        if (span.cost_estimate) {
          const estimate = span.cost_estimate;
          details.append(el('p', '', `输入 ${money(estimate.input_cost_cny)} + 输出 ${money(estimate.output_cost_cny)}。按每次输入长度选择阶梯，并按缓存命中计价。`));
          details.append(el('p', '', `单价（元 / 百万 Token）：输入 ${estimate.input_per_million_cny}、缓存命中 ${estimate.cache_read_per_million_cny}、输出 ${estimate.output_per_million_cny}。`));
          details.append(el('p', '', `单价核验日期：${estimate.pricing_verified_on}。估算不含免费额度、折扣及账单调整。`));
          const link = el('a', '', '阿里云官方模型定价 ↗');
          link.href = estimate.pricing_source === 'https://help.aliyun.com/zh/model-studio/qwen3-8-flash'
            ? estimate.pricing_source : 'https://help.aliyun.com/zh/model-studio/qwen-plus';
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          details.append(link);
        } else details.append(el('p', '', '缺少接口用量或匹配的模型单价，费用未知。'));
      } else {
        details.append(el('p', '', span.tool_cost?.source || '费用来源未知'),
          el('h4', '', '参数（脱敏预览）'),
          el('pre', '', JSON.stringify(span.arguments, null, 2)),
          el('h4', '', '返回结果（脱敏预览）'),
          el('pre', '', span.result === undefined ? '尚未收到返回结果' : JSON.stringify(span.result, null, 2)));
      }
      if (span.error) details.append(el('p', 'trace-unknown', `错误类型：${span.error}`));
      body.append(details);
      list.append(item);
    }
    if (trace.state !== 'running') {
      const end = step(trace.spans.length + 2,
        trace.state === 'complete' ? '回复完成' : trace.state === 'stopped' ? '已停止生成' : '请求失败',
        trace.state, '');
      list.append(end.item);
    }
    this.root.append(list);
    this.root.scrollTop = oldScroll;
    this.tick();
  }
}
