#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("data/meta/code_name_map.json")


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "stocktradebyz/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read().decode(charset)
    return json.loads(data)


def fetch_all_codes(base_url: str) -> list[str]:
    payload = _fetch_json(f"{base_url.rstrip('/')}/api/stock-codes")
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    raw_list = data.get("list", []) if isinstance(data, dict) else []
    codes: list[str] = []
    for x in raw_list:
        s = str(x).strip()
        if len(s) >= 8 and s[:2] in ("sh", "sz", "bj"):
            tail = s[2:]
            if tail.isdigit() and len(tail) == 6:
                codes.append(tail)
    seen = set()
    out = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def search_name(base_url: str, code: str) -> dict[str, str] | None:
    qs = urllib.parse.urlencode({"keyword": code})
    payload = _fetch_json(f"{base_url.rstrip('/')}/api/search?{qs}")
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict):
            continue
        c = str(item.get("code", "")).strip()
        if c == code:
            name = str(item.get("name", "")).strip()
            exchange = str(item.get("exchange", "")).strip()
            if name:
                return {"name": name, "exchange": exchange}
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stock code -> name mapping cache from tdx-api.")
    parser.add_argument("--base-url", default=os.getenv("TDX_API_BASE_URL", "http://tdx-api:8080"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, str]] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")) or {}
        except Exception:
            existing = {}

    codes = fetch_all_codes(args.base_url)
    if not codes:
        print("No codes returned from /api/stock-codes", file=sys.stderr)
        return 2

    pending = [c for c in codes if not (args.skip_existing and c in existing and existing[c].get("name"))]
    total = len(pending)
    print(f"codes_total={len(codes)} pending={total} existing={len(existing)}")

    result = dict(existing)
    done = ok = failed = 0

    def _job(code: str):
        last_err: Exception | None = None
        for _ in range(args.retry + 1):
            try:
                item = search_name(args.base_url, code)
                if item:
                    return code, item, None
                return code, None, None
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        return code, None, last_err

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [ex.submit(_job, c) for c in pending]
        for fut in as_completed(futures):
            code, item, err = fut.result()
            done += 1
            if item:
                result[code] = item
                ok += 1
            else:
                failed += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if done % 200 == 0 or done == total:
                print(f"progress {done}/{total} ok={ok} failed={failed}")

    ordered = {k: result[k] for k in sorted(result.keys())}
    out_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} entries={len(ordered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
