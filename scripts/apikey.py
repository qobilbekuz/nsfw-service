#!/usr/bin/env python3
"""API kalitlarini boshqarish.

    .venv/bin/python scripts/apikey.py create --name "edugames" [--rpm 120]
    .venv/bin/python scripts/apikey.py list
    .venv/bin/python scripts/apikey.py revoke --id <key_id>

Kalitning o'zi Redis'da SAQLANMAYDI — faqat sha256 dayjesti. Ya'ni yaratilgan
paytda ko'rsatilgan kalitni yo'qotsangiz, uni tiklab bo'lmaydi: yangisini
yaratish kerak.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.cache import APIKEY_KEY, CacheClient  # noqa: E402

settings = get_settings()


def client() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)


def cmd_create(args: argparse.Namespace) -> int:
    api_key = "nsfw_" + secrets.token_urlsafe(32)
    key_id = "k_" + secrets.token_hex(6)
    record = {
        "id": key_id,
        "name": args.name,
        "rate_limit": args.rpm,
        "revoked": False,
        "created_at": int(time.time()),
    }
    rds = client()
    rds.set(APIKEY_KEY.format(CacheClient.hash_key(api_key)), json.dumps(record))
    rds.sadd("nsfw:keys", json.dumps({"id": key_id, "name": args.name}))

    print("Kalit yaratildi. Uni HOZIR saqlab qo'ying — qayta ko'rsatilmaydi:\n")
    print(f"  X-API-Key: {api_key}\n")
    print(f"  id        : {key_id}")
    print(f"  nomi      : {args.name}")
    print(f"  limit     : {args.rpm} so'rov/daqiqa")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rds = client()
    rows = []
    for raw_key in rds.scan_iter(match="nsfw:key:*", count=100):
        raw = rds.get(raw_key)
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rows.append(record)

    if not rows:
        print("Kalitlar yo'q.")
        return 0

    rows.sort(key=lambda r: r.get("created_at", 0))
    print(f"{'id':<16}{'nomi':<24}{'rpm':>6}  {'holat':<10}sana")
    for record in rows:
        created = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(record.get("created_at", 0))
        )
        state = "BEKOR" if record.get("revoked") else "faol"
        print(
            f"{record.get('id', '?'):<16}{record.get('name', '-'):<24}"
            f"{record.get('rate_limit', '-'):>6}  {state:<10}{created}"
        )
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    rds = client()
    for raw_key in rds.scan_iter(match="nsfw:key:*", count=100):
        raw = rds.get(raw_key)
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if record.get("id") == args.id:
            record["revoked"] = True
            rds.set(raw_key, json.dumps(record))
            print(f"Kalit bekor qilindi: {args.id} ({record.get('name')})")
            return 0
    print(f"Bunday id topilmadi: {args.id}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="NSFW API kalitlarini boshqarish")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="yangi kalit yaratish")
    create.add_argument("--name", required=True, help="kalit egasi/loyiha nomi")
    create.add_argument(
        "--rpm",
        type=int,
        default=settings.rate_limit_per_minute,
        help="daqiqasiga so'rovlar limiti (0 = cheksiz)",
    )
    create.set_defaults(func=cmd_create)

    listing = sub.add_parser("list", help="kalitlar ro'yxati")
    listing.set_defaults(func=cmd_list)

    revoke = sub.add_parser("revoke", help="kalitni bekor qilish")
    revoke.add_argument("--id", required=True, help="kalit id (k_...)")
    revoke.set_defaults(func=cmd_revoke)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
