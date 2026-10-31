"""Collect one adult's Tuniu profile locally, without echoing personal data or creating orders."""

import argparse
import json
import os
import re
import sys
from datetime import date
from getpass import getpass
from pathlib import Path

from personal_assistant_agent.agents.travel_tools import TravelStore
from personal_assistant_agent.core.config import get_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="default", help="档案名，不含 .json")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.name):
        parser.error("档案名仅允许字母、数字、下划线和连字符")
    if not sys.stdin.isatty():
        parser.error("请在本人本地交互终端运行，不要通过命令行参数或日志传递个人资料")
    os.chdir(Path(__file__).resolve().parents[1])
    store = TravelStore(get_settings())
    store.secure_root()
    folder = store.root / "profiles"
    if folder.is_symlink():
        raise SystemExit("拒绝符号链接目录")
    folder.mkdir(mode=0o700, exist_ok=True)
    folder.chmod(0o700)
    path = folder / (args.name + ".json")
    if path.exists() or path.is_symlink():
        raise SystemExit("档案已存在，请使用其他 --name；不自动覆盖")
    print("仅支持一位成人、中国大陆居民身份证。输入不回显，保存后仍需逐单确认。")
    family = getpass("证件上的姓：").strip()
    given = getpass("证件上的名：").strip()
    ident = getpass("居民身份证号：").strip().upper()
    phone = getpass("本人及联系人手机号：").strip()
    if not family or not given or not re.fullmatch(r"\d{17}[\dX]", ident):
        raise SystemExit("姓名或身份证格式无效，未保存")
    if not re.fullmatch(r"1\d{10}", phone):
        raise SystemExit("手机号格式无效，未保存")
    try:
        birth = date(int(ident[6:10]), int(ident[10:12]), int(ident[12:14]))
    except ValueError:
        raise SystemExit("身份证出生日期无效，未保存") from None
    today = date.today()
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    if age < 18:
        raise SystemExit("当前录入脚本仅支持成人，未保存")
    data = {
        "tuniu": {
            "flight": {
                "tourists": [
                    {
                        "name": family + given,
                        "idType": "身份证",
                        "idNumber": ident,
                        "mobile": phone,
                        "type": "成人",
                    }
                ],
                "contactTourist": {"name": family + given, "mobile": phone},
            },
            "hotel": {
                "roomGuests": [{"guests": [{"firstName": given, "lastName": family}]}],
                "contactName": family + given,
                "contactPhone": phone,
            },
        }
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("私有档案已保存，权限 0600。聊天中使用档案名：" + args.name)


if __name__ == "__main__":
    main()
