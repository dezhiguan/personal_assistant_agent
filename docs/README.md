# 文档索引

| 文档 | 用途 |
| --- | --- |
| [tools.md](tools.md) | 工具配置、外部服务与预订确认机制 |
| [langsmith-capabilities.md](langsmith-capabilities.md) | LangSmith 能力说明 |
| [个人助理原型.html](个人助理原型.html) | 个人助理界面原型 |

## Design

| 文档 | 说明 | 状态 |
| --- | --- | --- |
| [design/full-duplex-agent-interaction.html](design/full-duplex-agent-interaction.html) | **技术设计**：全双工 Agent 交互，持续追加对话与同任务要求合并列为第一优先级；含“全球新闻改国内”及连续修正用例、10 个用户用例、架构、状态隔离、持久恢复与验收方案 | P1a 与 P1b 主链路已实现；本地完整门禁通过（2026-11-03，489项Python / 17项Node / 122次真实场景，关键失败0；CI为120/122、关键失败2次且LangSmith月度trace额度耗尽，整体未通过；历史失败与范围限制见实施记录） |

## Research

| 文档 | 说明 | 状态 |
| --- | --- | --- |
| [research/main-中国大陆天气预警与旅行预订API选型.md](research/main-中国大陆天气预警与旅行预订API选型.md) | **技术调研**：大陆天气预警、机票与酒店查询/创单及分销 API 的能力、准入和当前项目适配 | 已调研（2026-11-01，官方资料与源码核验；未实测供应商交易） |
