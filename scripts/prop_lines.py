"""Snapshot current NFL player prop lines from The Odds API (free tier) into props_log/.

Run it twice a week to build our own history: once early (near open) and once close to
kickoff (near close). Each run writes props_log/<UTC timestamp>.csv, one row per
book/player/market/side. The log is committed to the repo so it survives the container.

Cost: listing events is free; each event's props cost (markets x regions) credits.
The free tier has 500 credits a month, so the script checks the cost before spending.

    python scripts/prop_lines.py --dry-run          # show events and cost, spend nothing
    python scripts/prop_lines.py --days 7           # snapshot games in the next 7 days
"""
import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "props_log"
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"]
REGIONS = "us"
RESERVE = 50  # never spend the last credits of the month


def api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("No ODDS_API_KEY in the environment or .env")
    return key


def get(url: str, params: dict) -> requests.Response:
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        sys.exit(f"{r.status_code} from {url.split('?')[0]}: {r.text[:200]}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="only games starting within this many days")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    key = api_key()

    r = get(f"{BASE}/events", {"apiKey": key})  # free
    remaining = int(r.headers.get("x-requests-remaining", 0))
    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(days=args.days)
    events = [e for e in r.json()
              if now < dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= horizon]
    cost = len(events) * len(MARKETS) * len(REGIONS.split(","))
    print(f"{len(events)} games in the next {args.days} days; estimated cost {cost} credits; "
          f"{remaining} remaining this month")
    if args.dry_run:
        for e in events:
            print(f"  {e['commence_time']}  {e['away_team']} @ {e['home_team']}")
        return
    if cost > remaining - RESERVE:
        sys.exit(f"Not enough credits (need {cost}, keeping {RESERVE} in reserve). Use a smaller --days.")

    LOG.mkdir(exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    path = LOG / f"{stamp}.csv"
    n = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snapshot_utc", "event_id", "commence_time", "home_team", "away_team",
                    "book", "market", "player", "side", "point", "price", "book_last_update"])
        for e in events:
            r = get(f"{BASE}/events/{e['id']}/odds",
                    {"apiKey": key, "regions": REGIONS, "markets": ",".join(MARKETS), "oddsFormat": "american"})
            remaining = r.headers.get("x-requests-remaining")
            for b in r.json().get("bookmakers", []):
                for m in b.get("markets", []):
                    for o in m.get("outcomes", []):
                        w.writerow([stamp, e["id"], e["commence_time"], e["home_team"], e["away_team"],
                                    b["key"], m["key"], o.get("description"), o.get("name"),
                                    o.get("point"), o.get("price"), m.get("last_update")])
                        n += 1
    print(f"wrote {n} lines to {path.relative_to(ROOT)}; {remaining} credits left")


if __name__ == "__main__":
    main()
