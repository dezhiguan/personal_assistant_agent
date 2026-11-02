# LangSmith 接入与验收

本项目直接通过 Python / Deep Agents 调用 Qwen。LangSmith 提供提示词版本管理、调用追踪、评测和部署入口，不依赖 Keel 运行时。

## 提示词托管

系统提示词源文件：`src/personal_assistant_agent/prompts/system.txt`。执行：

```bash
uv run python scripts/sync_prompt.py
```

脚本发布私有提示词、回读验证内容，把不可变提交哈希写入本机 `.env` 的 `LANGSMITH_PROMPT_REF`。服务启动时拉取该版本，只允许单条无变量 system 消息，不下载模型配置或密钥。更新版本后重启服务；回滚时把引用改回旧提交再重启。

当前发布版本：

```text
personal-assistant-system:5e18c4c5f0157b4e6acabe59475fc77350be93bf399483c8323a595b973c271e
```

[在 LangSmith 查看提示词](https://smith.langchain.com/prompts/personal-assistant-system/5e18c4c5?organizationId=a7cf4d73-f4f9-44f1-9a1f-a8ac88a4ccc0)。留空引用时使用打包的本地版本；配置托管版本后，拉取失败会阻止启动，避免悄悄使用另一版提示词。关闭 tracing 不等于关闭托管提示词读取。

## 数据集与自动评测

唯一数据源为 `evaluations/golden/cases.json`，共61个合成场景、12个维度，
同步到 [personal-assistant-golden-v2](https://smith.langchain.com/o/a7cf4d73-f4f9-44f1-9a1f-a8ac88a4ccc0/datasets/738db8d7-5ede-4e4b-8e77-8c7430ba0087)，每条重复两次。原10条
[历史数据集](https://smith.langchain.com/o/a7cf4d73-f4f9-44f1-9a1f-a8ac88a4ccc0/datasets/465f4c9b-37ab-4166-a4ed-a2db6bc39c9e)
及既有实验保留，源文件归档为 `evaluations/golden/history/assistant-v1.json`。
同步使用稳定用例ID，重复运行不重复插入；发现额外云端样例会失败并要求复核，不自动删除。
不会把真实用户聊天自动加入数据集。

```bash
make agent-check    # pytest、前端、Ruff，再运行真实模型实验
make golden         # 仅真实模型实验，122次场景检查
make golden-offline # 17条边界场景各运行两次，无网络
```

`scripts/run_golden.py` 与兼容入口 `scripts/evaluate_agent.py` 调用同一实现。
LangSmith 原生 `aevaluate` 负责实验、并发、重复运行和评分记录；官方 OpenEvals Correctness
负责开放题评分，Agent和裁判均使用 `qwen3.8-flash`，不需要 Qwen-Plus 权限。
项目仍保留工具授权、真实文件、待办、交付格式等业务断言，以及独立的真实 Graph 并发与进程恢复测试。
接入 LangSmith 不会自动生成这些业务正确性标准。

每次检查必须通过项目断言，需语义评分时分数至少0.8；全部122次通过才允许发布。
关键失败单独列出，执行或裁判错误、缺失反馈及不完整覆盖均失败。模型裁判可能误判，仍需人工复核。
历史基线原样保留，不用另一评分器的分数覆盖；在 LangSmith 内比较评分配置一致的实验。
实验记录包含模型、评分器版本和提示词哈希、用例哈希、Agent源码哈希、Git revision及未提交标记。
本地完整报告与合成工作区位于忽略的 `artifacts/golden/`，最新摘要为 `latest.json`。

只保留 `.github/workflows/agent-regression.yml`，免费检查后执行真实模型实验，
不再重复运行另一套10条用例流程。真实评测依赖 `golden-evaluation` 环境中的
`DASHSCOPE_API_KEY`、`LANGSMITH_API_KEY`；托管提示词可配置固定版本 `LANGSMITH_PROMPT_REF`。
缺少配置明确失败；报告在失败时仍上传。`--judge` 为兼容选项，正式实验始终启用质量评分。

2026-11-03 本地完整检查通过：449项Python、17项Node、Ruff，真实场景122/122，关键失败0。
[本轮实验](https://smith.langchain.com/o/a7cf4d73-f4f9-44f1-9a1f-a8ac88a4ccc0/datasets/738db8d7-5ede-4e4b-8e77-8c7430ba0087/compare?selectedSessions=173b3a07-4b76-4edf-9b73-613c344c5833) 为 `personal-assistant-golden-5cbfc3b8`；
源码指纹核验、完整日志与先前两次中止记录位于 `artifacts/ci/native-evaluation/`。
云托管与GitHub Actions结果应分别核验，不能由本地评估代替。

## Token 与费用

```bash
uv run python scripts/sync_pricing.py
# 启动聊天服务后：
uv run python scripts/verify_usage.py --planning
```

Qwen 传输开启 `stream_usage=True`，计数使用供应商响应的 usage，保留缓存命中数。NDJSON 在 `done` 前发送 `usage`。Planning 的多次模型调用分别计价后相加；不会把整轮 Token 当成一次长上下文，也不会对父子追踪重复计费。供应商未返回 usage 时不伪造零值。

LangSmith LLM span 写入 usage 和费用估算元数据；工作区价格表注册对应的 Token 单价、缓存单价和上下文分层，使原生 Input / Output / Total Cost 可以显示和汇总。价格表仅匹配 `qwen-plus` 与自定义追踪 provider `dashscope-beijing-nonthinking`，实际请求仍直接发往百炼，不改变模型身份或请求协议。

价格快照位于 `observability/qwen_pricing.json`（北京地域 qwen-plus 非思考模式，2026-10-28 核验）和 `observability/qwen_flash_pricing.json`（北京地域 qwen3.8-flash，2026-11-01 核验）。qwen-plus 按每次输入长度分为 ≤128000、≤256000、≤1000000 Token；qwen3.8-flash 输入/缓存命中/输出分别为 0.8/0.1/2.7 元每百万 Token，来源为[官方模型页](https://help.aliyun.com/zh/model-studio/qwen3-8-flash)。其他模型或地域不会套用这些估算。USD 使用文件中 ECB 2026-10-27 的汇率快照，页面以人民币展示。费用是公开标价估算，不包含免费额度、折扣、基础设施或实际账单调整。`CostTracer` 在每个模型叶节点写入对应估算；现有 `sync_pricing.py` 仅同步 qwen-plus 价格映射，LangSmith 不回溯重算旧追踪。

### 页面调用链路

右侧时间线通过请求范围的 LangChain 回调采集模型和工具开始/结束事件，含子代理内部调用。子调用归属优先使用回调父链；LangSmith 原生中间件未发送 LangChain 回调时，使用当前原生追踪的祖先 ID 补足归属，不按时间重叠猜测。`trace` 事件中的 `sequence` 按实际开始时间排序；同一 `id` 的结束事件更新原条目。耗时使用服务端单调时钟测量，包含传输、等待及同一次模型调用内的重试；属于调用层耗时，不是服务商 GPU 运算时间。总耗时为整轮请求的实际时间，不能将并行条目相加。模型原始提示词和推理内容不会发送到页面；工具仅展示受限的脱敏预览，文件/执行/子代理工具省略原始内容。

Token 来自接口 `usage_metadata`，缺失时显示未知。模型费用按各次调用自己的 Token、缓存用量和阶梯分别计算后汇总。工具账单尚未接入：已知本地工具仅注明“无外部 API 费用”（不包含服务器成本）；外部工具标记未知；task 内部调用分别计入汇总，父调用不重复收费。机票、酒店、汇率等业务数据不代表 API 调用费用。有缺项时汇总显示“已知成本 · 估算，不完整”。停止或连接中断后，未收到的 Token、费用与结束耗时不能补造。历史链路随对话保存在当前浏览器，旧记录不回填估算。

可使用 `uv run python scripts/verify_timeline.py --base-url http://127.0.0.1:8000` 运行真实付费验收，覆盖直接回答、多工具、子代理和缺配置工具失败。脚本核对原始事件与 LangSmith 同 ID 叶节点的 Token 和耗时，并复核费用公式；结果写入忽略的 `artifacts/timeline/live/`。该检查不验证账单实扣。网络不可用、服务未重启或追踪不完整都算验收失败。

`verify_usage.py` 实际运行一轮聊天，然后回读 LangSmith，检查供应商计数、各 LLM span 和根追踪汇总，核对 Input / Output / Total Cost。允许价格存储舍入误差（相对 1e-5、绝对 1e-9）；缺字段、少调用或金额不符均失败。报告保存在 `artifacts/usage/latest.json`，完整事件保存在同目录 `latest-events.json`，均被 Git / Docker 忽略。

## 部署入口

`langgraph.json` 注册 `assistant` graph，并挂载已有 HTML / 静态资源 / 聊天 API。部署工厂在线程中初始化模型和托管提示词；独立聊天接口继续显式传入历史和 todos。Agent Server 原生线程接口采用平台自己的持久化机制，和浏览器本地会话是两条独立入口。

```bash
uv sync --locked
uv run --env-file .env langgraph dev --no-browser --port 8123
```

本机打开 `http://127.0.0.1:8123/`。`/api/chat` 提供网页聊天；原生 `/runs/wait` 可使用 `assistant_id=assistant`。本地 dev 使用内存运行时与本地状态文件，不能当作正式云持久化服务。

生成部署镜像定义：

```bash
mkdir -p artifacts/deployment
uv run langgraph dockerfile artifacts/deployment/AgentServer.Dockerfile
```

已完成配置校验和本机 Agent Server 真实 Qwen 调用。未执行生产镜像构建或创建云端实例。`.env` 不写进配置或镜像；正式部署通过 LangSmith 的 Secrets / Environment Variables 注入 Qwen Key、LangSmith Key、提示词引用以及模型相关配置。追踪目标应采用所选部署环境返回的值，不要照搬 Development 的兼容名称。

部署的自定义路由显式开启平台认证，避免聊天入口绕过 Agent Server 的访问控制。本机 dev 默认无认证，只绑定回环地址；云上应通过认证客户端或已有受保护入口访问，不能把 LangSmith API Key 写进页面。当前未新增面向公众的登录系统。

**云托管尚未开通。** 账号部署页面显示需要升级 Plus（页面起价约 US$39/月），尚未购买订阅、创建付费实例或发布公网地址。审批开通后，可使用 LangSmith 部署页面选择仓库及 `langgraph.json`，或通过 CLI 从本地项目部署；随后还需验证云端构建、鉴权与健康检查。套餐及运行资源费用以当时页面为准。

## 本次验收记录（2026-10-29）

- Python 测试 43 项、浏览器流协议测试 5 项通过；Ruff 检查和 Python 包构建通过，wheel 包含页面、提示词和价格快照。
- [最新回归实验](https://smith.langchain.com/o/a7cf4d73-f4f9-44f1-9a1f-a8ac88a4ccc0/datasets/465f4c9b-37ab-4166-a4ed-a2db6bc39c9e/compare?selectedSessions=9ef3cb61-2817-4c00-a663-a7b9ed030a7d)：`personal-assistant-regression-93ef0ec1`，规则 10/10 通过，模型质量均分 1.00，每项 ≥ 0.8。10 条助理根追踪均能回读原生费用字段。
- 常用服务 `http://127.0.0.1:8000` 的 Planning 实测：2 次模型调用，输入 3855、输出 475、合计 4330 Token，其中缓存命中 1664。人民币标价估算 ¥0.00296904；LangSmith 汇总 US$0.000445297226。[查看真实追踪](https://smith.langchain.com/o/a7cf4d73-f4f9-44f1-9a1f-a8ac88a4ccc0/projects/p/01a181c8-5498-71af-8049-8f6c62952ea4/r/2d5f8c32-d6d9-4e41-8100-9c06af19cdf6?trace_id=2d5f8c32-d6d9-4e41-8100-9c06af19cdf6&start_time=2026-10-29T01:18:51.221515)。
- Agent Server `8123`：`/ok`、HTML、静态资源、健康检查正常；自定义聊天流与原生 `/runs/wait` 已分别收到真实 Qwen 回复。平台认证已配置，但本机 dev 的无认证模式不能作为云端鉴权验证。
- 云端实例尚未创建。GitHub Actions 随代码发布，真实评测依赖仓库 Secrets 配置。上述本机验证不能替代发布后的云端构建、网络、鉴权和持久化验收。

## 官方参考

- [Evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [Programmatic prompt management](https://docs.langchain.com/langsmith/manage-prompts-programmatically)
- [Token / cost tracking](https://docs.langchain.com/langsmith/cost-tracking)
- [Cloud deployment prerequisites](https://docs.langchain.com/langsmith/deploy-to-cloud)
- [Agent Server custom routes](https://docs.langchain.com/langsmith/custom-routes)
- [Qwen Plus 价格](https://help.aliyun.com/zh/model-studio/qwen-plus)
