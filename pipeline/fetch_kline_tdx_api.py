#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch kline data from tdx-api and write CSVs."
    )
    parser.add_argument(
        "--codes",
        required=False,
        default="",
        help="Comma/space separated stock codes, e.g. 000001,600000",
    )
    parser.add_argument(
        "--all-a",
        action="store_true",
        help="Fetch all A-share codes from tdx-api /api/stock-codes (sh/sz/bj) and download in batch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Parallelism when --all-a is set (default: 10)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip code if output CSV already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Number of records to fetch (default: 400)",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Output directory for CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override TDX_API_BASE_URL, e.g. http://tdx-api:8080",
    )
    parser.add_argument(
        "--price-scale",
        choices=["auto", "1", "0.001"],
        default="auto",
        help="Price scale: auto/1/0.001 (0.001 means divide by 1000)",
    )
    parser.add_argument(
        "--volume-mult",
        choices=["auto", "1", "100"],
        default="auto",
        help="Volume multiplier: auto/1/100 (100 means hand->shares)",
    )
    return parser.parse_args()


def _normalize_date(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        s = str(int(value))
    else:
        s = str(value).strip()
    if len(s) >= 10 and "-" in s:
        return s[:10]
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    if "T" in s:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    return s


def _split_codes(raw: str):
    parts = []
    for chunk in raw.replace(" ", ",").split(","):
        c = chunk.strip()
        if c:
            parts.append(c)
    return parts


def _guess_price_scale(prices):
    if not prices:
        return 1.0
    sorted_prices = sorted(prices)
    median = sorted_prices[len(sorted_prices) // 2]
    if median > 1000:
        return 0.001
    return 1.0


def _guess_volume_mult(vols):
    if not vols:
        return 1.0
    sorted_vols = sorted(vols)
    median = sorted_vols[len(sorted_vols) // 2]
    # Heuristic: if median is very small, treat as "hand" and convert to shares
    if median < 1000:
        return 100.0
    return 1.0


def _fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "stocktradebyz/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        data = resp.read().decode(charset)
    return json.loads(data)


def fetch_all_a_codes(base_url: str) -> list[str]:
    """Return 6-digit stock codes from /api/stock-codes list like sh600000."""
    url = f"{base_url.rstrip('/')}/api/stock-codes"
    payload = _fetch_json(url)
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    raw_list = data.get("list", []) if isinstance(data, dict) else []
    codes: list[str] = []
    for x in raw_list:
        s = str(x).strip()
        # Expect sh600000/sz000001/bj8xxxxx
        if len(s) >= 8 and s[:2] in ("sh", "sz", "bj"):
            tail = s[2:]
            if tail.isdigit() and len(tail) == 6:
                codes.append(tail)
    # de-dup, stable order
    seen = set()
    out = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _extract_list(payload):
    # Common shapes:
    # 1) { data: { list: [...] } }
    # 2) { data: [...] }
    # 3) { list: [...] }
    # 4) [...]
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            if "list" in data:
                return data.get("list") or []
            if "List" in data:
                return data.get("List") or []
        if isinstance(data, list):
            return data
        if "list" in payload:
            return payload.get("list") or []
        if "List" in payload:
            return payload.get("List") or []
    if isinstance(payload, list):
        return payload
    return []


def _coerce_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def fetch_one(code: str, base_url: str, limit: int):
    qs = urllib.parse.urlencode({"code": code, "type": "day", "limit": limit})
    url = f"{base_url.rstrip('/')}/api/kline-all/tdx?{qs}"
    payload = _fetch_json(url)
    rows = _extract_list(payload)
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "date": _normalize_date(r.get("time") or r.get("Time") or r.get("date") or r.get("Date")),
                "open": _coerce_float(r.get("open") or r.get("Open")),
                "high": _coerce_float(r.get("high") or r.get("High")),
                "low": _coerce_float(r.get("low") or r.get("Low")),
                "close": _coerce_float(r.get("close") or r.get("Close")),
                "volume": _coerce_float(r.get("volume") or r.get("Volume") or r.get("vol") or r.get("Vol")),
            }
        )
    return out


def main():
    args = _parse_args()
    base_url = args.base_url or os.getenv("TDX_API_BASE_URL")
    if not base_url:
        print("Missing TDX_API_BASE_URL (or use --base-url)", file=sys.stderr)
        sys.exit(2)

    if args.all_a:
        codes = fetch_all_a_codes(base_url)
        if not codes:
            print("No codes returned from /api/stock-codes", file=sys.stderr)
            sys.exit(2)
    else:
        codes = _split_codes(args.codes)

    if not codes:
        print("No valid codes provided (use --codes or --all-a)", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)

    # For --all-a, optionally run with simple thread pool
    if args.all_a and args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _job(code: str):
            out_path = os.path.join(args.out, f"{code}.csv")
            if args.skip_existing and os.path.exists(out_path):
                return code, "skipped", 0, None
            try:
                rows = fetch_one(code, base_url, args.limit)
                return code, "ok", len(rows), rows
            except Exception as e:
                return code, "err", 0, e

        total = len(codes)
        done = 0
        ok = 0
        skipped = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(_job, c) for c in codes]
            for fut in as_completed(futures):
                code, status, n, payload = fut.result()
                done += 1
                if status == "skipped":
                    skipped += 1
                    if done % 200 == 0:
                        print(f"Progress {done}/{total} (ok={ok}, skipped={skipped}, failed={failed})")
                    continue
                if status == "err":
                    failed += 1
                    if done % 50 == 0:
                        print(f"Progress {done}/{total} (ok={ok}, skipped={skipped}, failed={failed})")
                    continue

                rows = payload
                if not rows:
                    failed += 1
                    continue

                # scale + write
                prices = [r["close"] for r in rows if r["close"] > 0]
                vols = [r["volume"] for r in rows if r["volume"] >= 0]

                price_scale = _guess_price_scale(prices) if args.price_scale == "auto" else float(args.price_scale)
                volume_mult = _guess_volume_mult(vols) if args.volume_mult == "auto" else float(args.volume_mult)

                for r in rows:
                    r["open"] = round(r["open"] * price_scale, 6)
                    r["high"] = round(r["high"] * price_scale, 6)
                    r["low"] = round(r["low"] * price_scale, 6)
                    r["close"] = round(r["close"] * price_scale, 6)
                    r["volume"] = round(r["volume"] * volume_mult, 6)

                rows = sorted(rows, key=lambda x: x["date"])

                out_path = os.path.join(args.out, f"{code}.csv")
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["date", "open", "high", "low", "close", "volume"]
                    )
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(
                            {
                                "date": r["date"],
                                "open": r["open"],
                                "high": r["high"],
                                "low": r["low"],
                                "close": r["close"],
                                "volume": r["volume"],
                            }
                        )

                ok += 1
                if done % 200 == 0 or done == total:
                    print(f"Progress {done}/{total} (ok={ok}, skipped={skipped}, failed={failed})")

        print(f"Done all-a: total={total}, ok={ok}, skipped={skipped}, failed={failed}")
        return

    for code in codes:
        out_path = os.path.join(args.out, f"{code}.csv")
        if args.skip_existing and os.path.exists(out_path):
            print(f"Skipping {code} (exists)")
            continue
        print(f"Fetching {code} from {base_url} ...")
        rows = fetch_one(code, base_url, args.limit)
        if not rows:
            print(f"No data for {code}")
            continue

        prices = [r["close"] for r in rows if r["close"] > 0]
        vols = [r["volume"] for r in rows if r["volume"] >= 0]

        if args.price_scale == "auto":
            price_scale = _guess_price_scale(prices)
        else:
            price_scale = float(args.price_scale)

        if args.volume_mult == "auto":
            volume_mult = _guess_volume_mult(vols)
        else:
            volume_mult = float(args.volume_mult)

        for r in rows:
            r["open"] = round(r["open"] * price_scale, 6)
            r["high"] = round(r["high"] * price_scale, 6)
            r["low"] = round(r["low"] * price_scale, 6)
            r["close"] = round(r["close"] * price_scale, 6)
            r["volume"] = round(r["volume"] * volume_mult, 6)

        rows = sorted(rows, key=lambda x: x["date"])

        out_path = os.path.join(args.out, f"{code}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "open", "high", "low", "close", "volume"]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {
                        "date": r["date"],
                        "open": r["open"],
                        "high": r["high"],
                        "low": r["low"],
                        "close": r["close"],
                        "volume": r["volume"],
                    }
                )
        print(
            f"Wrote {len(rows)} rows to {out_path} "
            f"(price_scale={price_scale}, volume_mult={volume_mult})"
        )


if __name__ == "__main__":
    main()
