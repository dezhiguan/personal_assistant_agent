"""Persist RFC 5545 calendar events with optional alarms as importable ICS files."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from personal_assistant_agent.agents.life_tools import guarded
from personal_assistant_agent.agents.tool_http import ToolError
from personal_assistant_agent.agents.workspace import WorkspaceBackend
from personal_assistant_agent.core.config import Settings


def timestamp(value: str, timezone: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    zone = ZoneInfo(timezone)
    if parsed.tzinfo is None:
        # Reject both gaps and ambiguous local times. Explicit UTC offsets let
        # users disambiguate DST transitions without silently moving an event.
        a, b = parsed.replace(tzinfo=zone, fold=0), parsed.replace(tzinfo=zone, fold=1)
        if a.utcoffset() != b.utcoffset():
            raise ToolError("本地时间处于夏令时切换，请提供带 UTC 偏移的 ISO 时间。")
        parsed = a
    return parsed.astimezone(UTC)


def escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
        .replace(";", "\\;")
        .replace(",", "\\,")
    )


def fold(line: str) -> str:
    result, current = [], ""
    for char in line:
        if len((current + char).encode("utf-8")) > 75:
            result.append(current)
            current = " "
        current += char
    return "\r\n".join([*result, current])


def build_calendar_tools(settings: Settings) -> list:
    def save(title, start, end, timezone, description, location, reminder_minutes):
        if not title.strip() or len(title) > 300 or len(description) > 5000 or len(location) > 500:
            raise ToolError("标题必须为 1–300 字符，描述最多 5000 字符，地点最多 500 字符。")
        if any(ord(c) < 32 and c not in "\r\n\t" for c in title + description + location):
            raise ToolError("日历文本含无效控制字符。")
        begin, finish = timestamp(start, timezone), timestamp(end, timezone)
        if finish <= begin:
            raise ToolError("结束时间必须晚于开始时间。")
        if reminder_minutes is not None and not 0 <= reminder_minutes <= 525600:
            raise ToolError("提醒提前分钟数须为 0–525600。")
        event_id = uuid4().hex

        def stamp(value):
            return value.strftime("%Y%m%dT%H%M%SZ")

        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Personal Assistant//Calendar//ZH",
            "CALSCALE:GREGORIAN",
            "BEGIN:VEVENT",
            f"UID:{event_id}@personal-assistant",
            f"DTSTAMP:{stamp(datetime.now(UTC))}",
            f"DTSTART:{stamp(begin)}",
            f"DTEND:{stamp(finish)}",
            f"SUMMARY:{escape(title)}",
            f"DESCRIPTION:{escape(description)}",
            f"LOCATION:{escape(location)}",
        ]
        if reminder_minutes is not None:
            lines += [
                "BEGIN:VALARM",
                "ACTION:DISPLAY",
                f"TRIGGER:-PT{reminder_minutes}M",
                f"DESCRIPTION:{escape(title)}",
                "END:VALARM",
            ]
        lines += ["END:VEVENT", "END:VCALENDAR"]
        content = "\r\n".join(fold(line) for line in lines) + "\r\n"
        backend = WorkspaceBackend(settings.workspace_dir)
        path = f"/calendar/{event_id}.ics"
        result = backend.write(path, content)
        if result.error:
            raise ToolError("日历文件未保存，请检查工作区权限。")
        return {
            "status": "pending_import",
            "event_id": event_id,
            "file_path": path,
            "download_url": f"/api/calendar/{event_id}.ics",
            "start_utc": begin.isoformat(),
            "end_utc": finish.isoformat(),
            "reminder_minutes": reminder_minutes,
            "message": "已保存 ICS 文件；须下载并导入日历后才会出现日程/触发提醒。"
            "本服务不提供后台通知。",
        }

    @tool
    @guarded
    async def create_calendar_event(
        title: str,
        start: str,
        end: str,
        timezone: str = "Asia/Shanghai",
        description: str = "",
        location: str = "",
        reminder_minutes: int | None = None,
    ) -> dict:
        """创建可下载的本地 ICS 日程；start/end 为 ISO 日期时间，支持 IANA 时区。
        可带 VALARM 提醒；需用户导入系统日历才生效，不会直接写入远端日历账号。
        返回 pending_import 时只能声称文件已保存，不得声称日程已同步或提醒已激活。
        """
        return save(title, start, end, timezone, description, location, reminder_minutes)

    @tool
    @guarded
    async def set_reminder(
        title: str, remind_at: str, timezone: str = "Asia/Shanghai", description: str = ""
    ) -> dict:
        """生成在指定时间弹出提醒的 ICS 日历文件，需用户下载并导入系统日历后生效。
        remind_at 为未来 ISO 时间，默认上海时区；本服务不提供后台推送或自动通知。
        """
        begin = timestamp(remind_at, timezone)
        if begin <= datetime.now(UTC):
            raise ToolError("提醒时间必须晚于当前时间。")
        return save(
            title,
            begin.isoformat(),
            (begin + timedelta(minutes=1)).isoformat(),
            timezone,
            description,
            "",
            0,
        )

    return [create_calendar_event, set_reminder]
