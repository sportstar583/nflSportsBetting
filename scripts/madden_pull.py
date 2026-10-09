"""Download Madden NFL player ratings from EA's ratings API (drop-api.ea.com) for every weekly update.

The API serves one edition (Madden 26 = the 2025 season as of 2026-10): base ratings, weeks 1-18
and the playoffs, 100 players per request. Writes data/raw/madden_ratings.parquet with one row per
player per update: EA id, name, birthdate, overall, the update's id/order, and the edition.

    python scripts/madden_pull.py
"""
import sys
import time
from pathlib import Path

import polars as pl
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "madden_ratings.parquet"
API = "https://drop-api.ea.com/rating/madden-nfl"
PAGE = 100
KEEP_STATS = ["speed", "acceleration", "awareness", "strength", "throwPower", "throwAccuracyShort", "throwAccuracyMid",
              "throwAccuracyDeep", "catching", "routeRunningShort", "routeRunningMedium", "routeRunningDeep",
              "runBlock", "passBlock", "blockShedding", "powerMoves", "finesseMoves", "manCoverage", "zoneCoverage",
              "tackle", "carrying", "breakTackle"]


def get(params: dict) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(API, params={"locale": "en", **params}, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"  {r.status_code} for {params}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  {e}", file=sys.stderr)
        time.sleep(2 * 2 ** attempt)
    raise RuntimeError(f"failed: {params}")


def main():
    filters = requests.get(f"{API}/filters", params={"locale": "en"}, timeout=60).json()
    iterations = [i["id"] for i in filters["iterations"]]
    edition = None
    rows = []
    for it in sorted(iterations, key=lambda s: int(s.split("-")[0])):
        offset, total = 0, None
        while total is None or offset < total:
            d = get({"limit": PAGE, "offset": offset, "iteration": it})
            total = d["totalItems"]
            for p in d["items"]:
                if edition is None and "madden-nfl-" in (p.get("avatarUrl") or ""):
                    edition = p["avatarUrl"].split("madden-nfl-")[1].split("/")[0]
                stats = p.get("stats") or {}
                rows.append({"ea_id": p["id"], "first_name": p["firstName"], "last_name": p["lastName"],
                             "birthdate": p["birthdate"], "overall": p["overallRating"], "age": p.get("age"),
                             "years_pro": p.get("yearsPro"), "jersey": p.get("jerseyNum"), "iteration": it,
                             "iter_order": int(it.split("-")[0]),
                             **{s: (stats.get(s) or {}).get("value") for s in KEEP_STATS}})
            offset += PAGE
            time.sleep(0.5)
        print(f"{it}: {total} players", flush=True)
    df = pl.DataFrame(rows).with_columns(pl.lit(edition).alias("edition"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    print(f"wrote {OUT.name}: {df.height} rows, edition {edition}")


if __name__ == "__main__":
    main()
