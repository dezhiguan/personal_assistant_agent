# 工具配置与可用性

清单共 32 个工具。保留原有 14 个，新增 18 个，没有重复注册。所有网络工具真实调用数据源；缺少配置、限流、空结果和错误都如实返回，不生成模拟业务结果。

| 工具 | 实现 / 条件 |
| --- | --- |
| baidu_search | 原有百度网页搜索，无密钥，可能遇到验证码 |
| get_current_time | 原有系统时钟，IANA 时区 |
| calculate | 原有安全 AST 数学计算 |
| validate_time_budget | 原有串行工期与预算核验 |
| write_todos | 原有 Deep Agents 计划中间件 |
| ls / read_file / write_file / edit_file | 原有受限工作区后端 |
| glob / grep / delete / execute / task | 原有工作区搜索、删除、受限命令及子代理 |
| get_city_weather | Open-Meteo 真实当前天气及 1–16 天预报，无密钥 |
| check_cn_holiday | chinese-calendar 1.11 起，当前覆盖 2004–2026 年，未覆盖年份报错 |
| convert_currency | Frankfurter v2 央行参考汇率，返回实际汇率日期，无密钥 |
| convert_units | Pint 本地单位换算，包含摄氏/华氏温度，限制单位语法和指数 |
| fetch_url | 静态 HTML/文本正文；不执行 JS；只允许公网 HTTP(S) |
| geocode | Open-Meteo / GeoNames 地名候选及 WGS84 经纬度，无密钥 |
| ip_locate | ipapi.co，公网 IP 粗定位；留空查询服务器出口位置 |
| get_weather_alert | QWeather 城市搜索和新版官方预警，需要 Host 和 API Key；预警链路不依赖 Open-Meteo |
| fetch_rss | RSS 2.0 / Atom，拒绝 XML 实体及内网地址 |
| translate | MyMemory，无密钥公共额度；最多 5000 UTF-8 字节，分段请求 |
| search_flights | 由 TRAVEL_PROVIDER 选择途牛国内航班或 Amadeus，需对应密钥 |
| search_hotels | 由 TRAVEL_PROVIDER 选择途牛或 Amadeus，需对应密钥；每次单间房 |
| book_flight | 供应商重新核价、本人确认后创单；需私有旅客档案和预订权限 |
| book_hotel | 供应商复核房型及条款、本人确认后创单；需私有住客档案和预订权限 |
| route_plan | OSRM / OpenStreetMap 驾车路线，无密钥，无实时交通；地名先 geocode |
| amap_maps | 高德地点/周边搜索、地理/逆地理编码、驾车/步行/骑行路线，需 Web服务类型 Key，坐标 GCJ-02 |
| create_calendar_event | 创建 ICS 日程及可选 VALARM，下载后导入日历才生效 |
| set_reminder | 创建带定时 VALARM 的 ICS 文件，下载后导入日历才生效 |

## 配置外部服务

在项目根目录 `.env` 填写，完整模板见 [../.env.example](../.env.example)。不要把密钥放进聊天、工具参数或工作区。

```dotenv
TOOL_TIMEOUT_SECONDS=15
AMAP_API_KEY=
QWEATHER_API_HOST=
QWEATHER_API_KEY=
TRAVEL_PROVIDER=tuniu
TUNIU_API_KEY=
TUNIU_MCP_BASE_URL=https://openapi.tuniu.cn/mcp
AMADEUS_CLIENT_ID=
AMADEUS_CLIENT_SECRET=
AMADEUS_ENVIRONMENT=test
TRAVEL_PRIVATE_DIR=.assistant-private
```

`QWEATHER_API_HOST` 为控制台分配的 `xxx.qweatherapi.com` 主机名，不含协议或路径。使用 `X-QW-Api-Key` 头认证及 `/weatheralert/v1/current/{latitude}/{longitude}` 新接口。旧 `/v7/warning/now` 已停止服务，不使用。

预警前先调用同一和风 Host 的 `/geo/v2/city/lookup` 解析地点，坐标按预警接口要求保留两位小数。此改动仅针对预警；普通天气、通用 `geocode` 仍使用 Open-Meteo。

## 大陆旅行接入：途牛

`.env.example` 推荐 `TRAVEL_PROVIDER=tuniu`；未设置该变量的旧部署仍使用 Amadeus，以保留兼容性。两家供应商不能混用密钥、报价或档案。途牛当前支持国内成人单程机票、CNY、1–9 位成人及每次一间酒店；往返分别搜索与确认。`origin/destination/city_code` 在途牛模式填写中文城市名，例如北京、上海；Amadeus 模式仍使用 IATA。

服务端通过 HTTP JSON-RPC 调用途牛 MCP，不需要给 Agent 开放通用 CLI。`tools/list` 核对当前账号的工具，兼容官方新旧酒店工具名；Key 只在 `apiKey` Header 中发送。默认地址采用官方客户端的 `/mcp`，如账户文档明确指定 `/hybrid/mcp`，再修改配置。实际权限及连通性必须通过真实只读检查，不能用 Key 存在代替。

```bash
# 本人本地交互终端输入一位成人的资料，输入不回显，不创建订单。
uv run python scripts/configure_traveler.py --name default
# 真实预警、航班和酒店只读验收；不会调用任何创单或付款接口。
uv run python scripts/verify_china_services.py --live
```

旅客档案在 `.assistant-private/profiles/default.json`，目录 `0700`、文件 `0600`。途牛数据放在顶层 `tuniu.flight` 与 `tuniu.hotel` 下，分别使用 `tourists/contactTourist` 和 `roomGuests/contactName/contactPhone`。酒店无需银行卡；真实个人信息不要放入聊天、工作区、命令行参数、仓库或调试日志。多人/护照档案须按[官方机票](https://open.tuniu.com/mcp/docs/apidoc/mcp/flightMCP.html)、[酒店契约](https://open.tuniu.com/mcp/docs/apidoc/mcp/hotelMCP.html)在私有文件中配置，人数须与报价一致。

首次预订只核价并返回 `requires_confirmation`。本人下一轮单独发送 `确认预订 <编号>` 后才创建真实订单；价格、条款或旅客改变必须重新确认。航班 `cabinPriceId`、酒店 `preBookParam` 只存服务端，报价15分钟过期。报价绑定会话和供应商凭据，切换账户后须重新搜索。

成功创单返回 **`pending_payment`**、订单号和 `payment_url`，本人到途牛核对最终价格并付款。该状态不代表付款、出票或酒店已确认。重复调用同一报价只返回已有记录；超时/取消/崩溃留下持久化 `unknown`，须到途牛核对，不能自动重新下单。无有效供应商链接时保留订单号，转途牛订单中心查询。当前未接入支付后状态同步、退改退款和自动取消，不承诺后台追踪。

独立报告保存在 `artifacts/china-services/latest.json`，区分 `missing_config`、`not_verified`、`failed`、`empty_results` 和 `verified`；预订始终列为单独未验收项。原免密钥验收脚本的固定清单不能用于判断途牛是否接通。

## Amadeus 兼容接入

`AMADEUS_ENVIRONMENT=test` 使用供应商测试库存和测试订单；切换 `production` 后需使用对应生产密钥。真实航班预订/出票另受 Amadeus 生产权限及出票合作资质限制。不能把测试响应称为真实可售库存、生产订单、付款或出票成功。

缺少上述配置时，6 个相关工具（高德地图、预警、航班/酒店查询及预订）会明确报告配置缺失。这表示集成尚未启用，不能以工具已经注册代替实际可用性验收。

## 高德地图

服务端 `.env` 配置 `AMAP_API_KEY`，须在高德开放平台申请 **Web服务类型** Key 并开通所用接口权限，不能使用 JS API 类型 Key。密钥不会进入模型工具参数或结果，也不会放入前端。未配置时工具明确返回“高德地图未启用”。

单个 `amap_maps` 工具通过 `action` 选择操作：

| action | 参数示例 | 接口 |
| --- | --- | --- |
| search | `query="故宫", city="北京市", max_results=5` | `/v5/place/text`，指定城市时严格限定区域 |
| nearby | `location="116.397128,39.916527", query="咖啡", radius=1000` | `/v5/place/around`，半径单位米 |
| geocode | `query="北京市东城区景山前街4号", city="北京市"` | `/v3/geocode/geo` |
| reverse_geocode | `location="116.397128,39.916527"` | `/v3/geocode/regeo` |
| route | `origin="116.397128,39.916527", destination="116.410000,39.920000", mode="walking"` | `/v5/direction/{driving,walking,bicycling}` |

坐标均为 **GCJ-02**，采用“经度,纬度”顺序，小数最多6位。不要直接传入现有 `geocode` / `route_plan` 的 WGS84 坐标；应先使用高德自身的搜索或地理编码核对地点候选。返回路线距离单位为米、耗时单位为秒；耗时未返回时不猜测。最多10个候选、每条路线100个路段，截断会标识。路线是查询时的估算，不是持续实时导航；当前不支持公交规划。

真实只读连通性检查（不需要模型调用）：

```bash
uv run python scripts/verify_amap.py --query 故宫 --city 北京市
```

报告保存到 `artifacts/amap/latest.json`。缺少 Key、权限不足、限流、空结果和网络失败均以非零退出码报告。接口契约测试使用合成数据，不代替配置真实 Key 后的联网验收。

官方文档：[地点搜索2.0](https://lbs.amap.com/api/webservice/guide/api/newpoisearch)、[地理/逆地理编码](https://lbs.amap.com/api/webservice/guide/api/geo)、[路径规划2.0](https://lbs.amap.com/api/webservice/guide/api/newroute)。

## 旅客档案与预订确认

使用浏览器聊天或 `/api/chat`，保持相同 `conversation_id`。原生图调用者需在受信任服务端提供 `configurable.travel_context`；仅构建图不会自行获得下单授权。

1. `search_flights` / `search_hotels` 返回当前会话绑定的 `quote_id`，报价 15 分钟有效。
2. 用户在服务端创建 `.assistant-private/profiles/<name>.json`。目录权限 `0700`，文件 `0600`。这一步不能由智能体文件工具执行，因为该目录在受限工作区之外。
3. 调用 `book_flight(quote_id, profile_name)` 或 `book_hotel(...)`，工具读取私有档案、重新验价，返回 `requires_confirmation`，展示旅客姓名、行程/酒店、价格、条款及确认文本。
4. 用户下一轮单独发送完整 `确认预订 <编号>`。服务端从本轮原始用户消息解析；模型不能通过设置工具参数伪造确认。沿用相同报价和档案再调用预订工具。
5. 旅客档案、价格或条款变化必须重新确认。成功返回供应商订单编号；航班 `reserved` 不等于已付款或出票。

档案顶层按预订类型组织。以下仅为结构说明，必须自行填入供应商要求的有效资料，不能用示例资料真实下单：

```json
{
  "flight": {
    "travelers": [],
    "contacts": []
  },
  "hotel": {
    "guests": [],
    "payments": [],
    "rooms": []
  }
}
```

`flight.travelers/contacts` 使用 Flight Create Orders 的原始字段，例如 `id/dateOfBirth/name/documents/contact` 等；`hotel.guests/payments/rooms` 使用 Hotel Bookings 的原始字段。联系人、证件、卡号不会作为工具参数或工具结果进入模型。工具仅返回旅客姓名和订单标识。付款数据的保存与处理需要符合实际供应商要求，生产部署优先使用供应商支持的托管付款方案。

报价和订单状态保存在 `.assistant-private/bookings.sqlite3`，单报价提交前以事务原子标记。提交后遇到超时、取消或进程崩溃，状态保持 `unknown`，服务重启也不会自动再次下单。必须人工核对供应商订单；这里没有自动取消或重新出票流程。该限制针对同一 `quote_id`，重新搜索产生的新报价仍需用户重新确认，不能用于绕过未知订单核对。

默认私有目录已排除在 Git 与 Docker 构建上下文之外。自定义 `TRAVEL_PRIVATE_DIR` 必须为项目内、工作区之外的独立相对目录，并自行加入忽略规则。Docker 默认预建私有目录并授权非 root 用户；持久化应挂载单独私有卷。当前应用仍是单用户应用，没有多租户登录权限体系。

## 日历与提醒

调用示例：

```text
create_calendar_event(title="项目评审", start="2030-01-02T10:00:00",
                      end="2030-01-02T11:00:00", timezone="Asia/Shanghai",
                      reminder_minutes=15)
set_reminder(title="提交报告", remind_at="2030-01-02T17:00:00+08:00")
```

文件真实保存在工作区 `/calendar/<event_id>.ics`，支持 UTF-8 折行、文本转义和 RFC 5545 VALARM。浏览器可通过返回的 `/api/calendar/<event_id>.ics` 下载并导入 Apple Calendar、Outlook 等兼容日历。返回状态为 `pending_import`，**导入之前没有激活任何通知**。当前没有 Google/微软账号写入、后台定时器、邮件或手机推送；不能声称已经同步账号或启用后台提醒。

## 网络与数据边界

- 所有新网络请求默认 15 秒，最大 60 秒；响应解压后最多 2 MiB；不继承系统代理。
- `fetch_url/fetch_rss` 校验每次重定向的公网地址，并连接已校验的 IP，保留 Host/TLS SNI，避免 DNS 重绑定；拒绝 URL 凭据及非 80/443 端口。
- 查询与正文属于外部不可信数据，不能作为工具执行或权限指令。
- 地名可能重名，核对返回省份和国家；气象预警查不到/接口失败不能等同于安全无预警。
- 汇率返回数据日期，周末/节假日可能沿用前一工作日；不代表银行成交价。
- IP 查询、天气地点、翻译文本和路线坐标会发给对应数据服务。公共服务存在额度、覆盖和可用性限制。

## 验证

```bash
make agent-check
uv run python scripts/verify_life_tools.py
```

前者包含单元测试、浏览器流协议检查、离线边界回归及真实模型黄金集。后者直接调用无需密钥的新数据工具及本地日历，不创建旅行订单。报告保存到 `artifacts/life-tools/latest.json`，外部服务失败会以非零退出码报告；另列需要密钥的未验收项。供应商接口测试使用合成夹具验证契约、确认、变价重新确认、私有档案权限和禁止重复订单，不等于真实供应商预订已经验收。

参考：[Open-Meteo](https://open-meteo.com/en/docs)、[Frankfurter](https://frankfurter.dev/)、[MyMemory](https://mymemory.translated.net/doc/spec.php)、[QWeather 预警](https://dev.qweather.com/en/docs/api/warning/weather-alert/)、[Amadeus 航班](https://github.com/amadeus4dev/developer-guides/blob/master/docs/resources/flights.md)、[Amadeus 酒店](https://github.com/amadeus4dev/developer-guides/blob/master/docs/resources/hotels.md)、[OSRM](https://project-osrm.org/docs/v5.5.1/api/)。
