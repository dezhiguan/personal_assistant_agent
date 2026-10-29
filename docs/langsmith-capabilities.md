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

数据集源文件：`evaluations/assistant-v1.json`，共 10 个合成用例，涵盖名称、历史上下文、请求隔离、计算、翻译、摘要、JSON、能力诚实性和 Planning。不会把真实用户聊天自动加入数据集。

[LangSmith 数据集](https://smith.langchain.com/o/a7cf4d73-f4f9-44f1-9a1f-a8ac88a4ccc0/datasets/465f4c9b-37ab-4166-a4ed-a2db6bc39c9e)。同步使用稳定用例 ID，更新已有样例，重复运行不重复插入，不删除平台中其他样例。额外样例导致数量与本地用例不符时，评测门槛会失败，需人工整理或使用新的数据集名称。

```bash
uv run python scripts/evaluate_agent.py --judge
```

命令同步数据集、调用真实 Qwen、创建 LangSmith experiment、写入规则评分与 Qwen 质量评分，保存 `artifacts/evaluations/latest.json`。每个用例的规则必须通过；使用 `--judge` 时，每项模型评分默认必须 ≥ 0.8，否则命令返回非零。阈值可用 `--min-quality` 调整；去掉 `--judge` 只运行确定性规则。实验记录包含提示词提交、用例文件 SHA256 和 Git revision。工作区未提交时，Git revision 只标识最近提交，不代表包含当前改动。

模型评分需要额外 Qwen 调用。同一系列 Qwen 充当裁判，结果是小规模回归信号，不能替代人工质量审核。

`.github/workflows/evaluation.yml` 已配置手动触发与 main 分支相关文件变更触发；`.github/workflows/ci.yml` 执行无付费调用的代码检查与测试。工作流随代码发布。真实评测需在仓库 Actions Secrets 配置 `DASHSCOPE_API_KEY`、`LANGSMITH_API_KEY`；不要把密钥填入工作流源码。缺少配置时，push 触发的真实评测会明确跳过并写入任务摘要；手动触发则报错，避免把未执行的评测当作通过。实际评测失败会保留报告并使任务失败。

## Token 与费用

```bash
uv run python scripts/sync_pricing.py
# 启动聊天服务后：
uv run python scripts/verify_usage.py --planning
```

Qwen 传输开启 `stream_usage=True`，计数使用供应商响应的 usage，保留缓存命中数。NDJSON 在 `done` 前发送 `usage`。Planning 的多次模型调用分别计价后相加；不会把整轮 Token 当成一次长上下文，也不会对父子追踪重复计费。供应商未返回 usage 时不伪造零值。

LangSmith LLM span 写入 usage 和费用估算元数据；工作区价格表注册对应的 Token 单价、缓存单价和上下文分层，使原生 Input / Output / Total Cost 可以显示和汇总。价格表仅匹配 `qwen-plus` 与自定义追踪 provider `dashscope-beijing-nonthinking`，实际请求仍直接发往百炼，不改变模型身份或请求协议。

价格快照位于 `observability/qwen_pricing.json`：北京地域 qwen-plus 非思考模式，核对日期 2026-10-28；按输入长度分为 ≤128000、≤256000、≤1000000 Token。其他模型或地域不会套用该估算。USD 使用文件中 ECB 2026-10-27 的汇率快照，人民币值一并保留。费用是公开标价估算，不包含免费额度、折扣或实际账单调整。更新价格或汇率快照后，重新执行 `sync_pricing.py`；LangSmith 不回溯重算旧追踪。

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
