"""Fetch past prop lines from prop-line.com (free tier) and grade them against actual stats.

The free tier redacts prices and results but shows every book's main line per game, back to
August, through /players/{name}/history. We take the actual stats from nflverse, so this gives a
graded history of lines without the paid tier. Budget: 1000 requests/day; one request per
player-market covers every week at once, and results are cached in props_log/propline/.

    python scripts/propline_history.py              # fetch (cached) + grade every completed week
    python scripts/propline_history.py --week 3     # report one week
    python scripts/propline_history.py --no-fetch   # grade from the cache only

Needs PROP_LINE_KEY in the environment (sent as X-API-Key).
"""
import argparse
import csv
import os
import sys
import time
import urllib.parse
from pathlib import Path

import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import props  # noqa: E402
from props_week import MARKET_KEYS, norm_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "props_log" / "propline"
BASE = "https://api.prop-line.com/v1/sports/americanfootball_nfl"
STAT = {"player_pass_yds": "passing_yards", "player_rush_yds": "rushing_yards",
        "player_reception_yds": "receiving_yards", "player_receptions": "receptions"}
# which markets to fetch per player, by his usage in the season (keeps the request count down)
ELIG = {"player_pass_yds": (["QB"], "attempts", 10), "player_rush_yds": (["RB", "QB"], "carries", 5),
        "player_reception_yds": (["WR", "TE", "RB"], "targets", 2), "player_receptions": (["WR", "TE", "RB"], "targets", 2)}
from props_week import TEAMS as TEAM_ABBR  # noqa: E402  prop-line full names -> nflverse codes


def players(season: int, week: int | None = None, markets: list[str] | None = None, scale: float = 1.0) -> pl.DataFrame:
    """Player-markets worth fetching: anyone who met the usage floor (x scale) in some game this
    season, or in `week` only. Used to stay inside the daily request budget."""
    ps = pl.read_parquet(DATA / "raw" / "player_stats.parquet").filter(pl.col("season") == season)
    if week:
        ps = ps.filter(pl.col("week") == week)
    rows = []
    for mkt, (positions, col, floor) in ELIG.items():
        if markets and mkt not in markets:
            continue
        d = (ps.filter(pl.col("position").is_in(positions) & (pl.col(col) >= floor * scale))
             .select("player_id", "player_display_name").unique())
        rows.append(d.with_columns(pl.lit(mkt).alias("market")))
    return pl.concat(rows)


def fetch(name: str, market: str, key: str) -> list[dict] | None:
    url = f"{BASE}/players/{urllib.parse.quote(name)}/history"
    r = requests.get(url, headers={"X-API-Key": key}, timeout=60,
                     params={"market": market, "main_line_only": "true", "limit": 100})
    if r.status_code == 429:
        sys.exit("prop-line daily request limit reached; rerun tomorrow (cache is kept)")
    if r.status_code != 200:
        return None
    remaining = r.headers.get("X-Daily-Remaining")
    return r.json().get("entries", []), remaining


def fetch_all(season: int, key: str, budget: int, **sel) -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = players(season, **sel)
    todo = todo.filter(~pl.struct("player_id", "market").map_elements(
        lambda r: (CACHE / f"{season}_{r['player_id']}_{r['market']}.csv").exists(), return_dtype=pl.Boolean))
    if todo.height > budget:
        print(f"{todo.height} player-markets to fetch but --budget is {budget}; fetching the first {budget}", file=sys.stderr)
        todo = todo.head(budget)
    out, n_req, remaining = [], 0, "?"
    for r in todo.iter_rows(named=True):
        path = CACHE / f"{season}_{r['player_id']}_{r['market']}.csv"
        if path.exists():
            continue
        res = fetch(r["player_display_name"], r["market"], key)
        n_req += 1
        rows = []
        if res is not None:
            entries, remaining = res
            for e in entries:
                if e.get("line") is None:
                    continue
                rows.append({"player_id": r["player_id"], "player": r["player_display_name"], "market": r["market"],
                             "event_id": e["event_id"], "commence_time": e["commence_time"],
                             "home_team": e["home_team"], "away_team": e["away_team"],
                             "book": e["bookmaker"], "line": e["line"], "is_main_line": e["is_main_line"]})
        with path.open("w", newline="") as f:  # an empty file marks "fetched, nothing there"
            w = csv.DictWriter(f, fieldnames=["player_id", "player", "market", "event_id", "commence_time",
                                              "home_team", "away_team", "book", "line", "is_main_line"])
            w.writeheader()
            w.writerows(rows)
        time.sleep(0.15)
        if n_req % 50 == 0:
            print(f"  {n_req} requests, {remaining} left today", file=sys.stderr)
    print(f"fetched {n_req} new player-markets; {remaining} requests left today", file=sys.stderr)
    return load_cache(season)


def load_cache(season: int) -> pl.DataFrame:
    files = [p for p in CACHE.glob(f"{season}_*.csv") if p.stat().st_size > 120]
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_csv(p, schema_overrides={"line": pl.Float64}) for p in files])


def grade(lines: pl.DataFrame, season: int) -> pl.DataFrame:
    """Consensus main line per player-market-game (median over books) joined to the actual stat."""
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet").filter(pl.col("season") == season)
             .with_columns(pl.col("home_team", "away_team").replace(props.RENAMES)))
    ps = (pl.read_parquet(DATA / "raw" / "player_stats.parquet").filter(pl.col("season") == season)
          .select("game_id", "player_id", "week", pl.col("team").replace(props.RENAMES),
                  *[pl.col(c).fill_null(0).cast(pl.Float64) for c in STAT.values()]))
    cons = (lines.filter(pl.col("is_main_line"))
            .with_columns(pl.col("home_team").replace_strict(TEAM_ABBR, default=None).alias("home"),
                          pl.col("away_team").replace_strict(TEAM_ABBR, default=None).alias("away"),
                          pl.col("commence_time").str.to_datetime(time_zone="UTC")
                          .dt.convert_time_zone("America/New_York").dt.date().cast(pl.String).alias("gameday"))
            .group_by("player_id", "player", "market", "home", "away", "gameday")
            .agg(pl.col("line").median().alias("line"), pl.len().alias("books")))
    g = (cons.join(sched.select("game_id", "week", pl.col("home_team").alias("home"), pl.col("away_team").alias("away"),
                                "gameday", "result"), on=["home", "away", "gameday"])
         .filter(pl.col("result").is_not_null())
         .join(ps.drop("week"), on=["game_id", "player_id"], how="left"))
    actual = pl.coalesce([pl.when(pl.col("market") == m).then(pl.col(c)) for m, c in STAT.items()])
    return (g.with_columns(actual.alias("actual"))
            .with_columns((pl.col("actual") - pl.col("line")).alias("diff"),
                          ((pl.col("actual") - pl.col("line")) / pl.col("line").clip(0.5)).alias("pct"),
                          pl.col("market").replace_strict(MARKET_KEYS).alias("mkt"),
                          pl.when(pl.col("actual").is_null()).then(pl.lit("void"))
                          .when(pl.col("actual") > pl.col("line")).then(pl.lit("over"))
                          .when(pl.col("actual") < pl.col("line")).then(pl.lit("under")).otherwise(pl.lit("push"))
                          .alias("result")))


def report(g: pl.DataFrame, week: int, top: int):
    w = g.filter((pl.col("week") == week) & (pl.col("result") != "void"))
    print(f"\n=== 2026 week {week}: {w.height} player-markets with a consensus main line "
          f"({g.filter((pl.col('week') == week) & (pl.col('result') == 'void')).height} void: player didn't play) ===")
    print("Over rate by market (pushes excluded):")
    for mkt in ["pass_yds", "rush_yds", "rec_yds", "receptions"]:
        d = w.filter((pl.col("mkt") == mkt) & (pl.col("result") != "push"))
        if d.height:
            print(f"  {mkt:<10} over {(d['result'] == 'over').mean():.1%} of {d.height}   "
                  f"mean actual - line {d['diff'].mean():+.1f}")
    with pl.Config(tbl_rows=top, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=1,
                   tbl_width_chars=160):
        cols = ["player", "team", "mkt", "line", "actual", "diff", "pct", "books"]
        yd = w.filter(pl.col("mkt") != "receptions")
        print(f"\nBiggest overs (yards above the line):")
        print(yd.sort("diff", descending=True).head(top).select(cols))
        print(f"\nBiggest unders (yards below the line):")
        print(yd.sort("diff").head(top).select(cols))
        rc = w.filter(pl.col("mkt") == "receptions")
        print(f"\nReceptions, most above / below the line:")
        print(pl.concat([rc.sort("diff", descending=True).head(top // 2), rc.sort("diff").head(top // 2)]).select(cols))
        print("\nBy team, share of its players' lines that went over (yardage + receptions):")
        bt = (w.filter(pl.col("result") != "push").group_by("team")
              .agg((pl.col("result") == "over").mean().alias("over_rate"), pl.len().alias("n")).sort("over_rate", descending=True))
        print(bt.head(top // 2).select("team", "over_rate", "n"), bt.tail(top // 2).select("team", "over_rate", "n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, help="report this week only (default: every completed week)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--budget", type=int, default=900, help="max requests this run (daily limit is 1000)")
    ap.add_argument("--only-week", type=int, help="fetch only players who played this week")
    ap.add_argument("--markets", help="comma-separated market keys to fetch")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the usage floors (fewer players)")
    args = ap.parse_args()
    if args.no_fetch:
        lines = load_cache(args.season)
    else:
        key = os.environ.get("PROP_LINE_KEY")
        if not key:
            sys.exit("PROP_LINE_KEY is not set")
        lines = fetch_all(args.season, key, args.budget, week=args.only_week,
                          markets=args.markets.split(",") if args.markets else None, scale=args.scale)
    if lines.is_empty():
        sys.exit("no lines in the cache")
    g = grade(lines, args.season)
    g.write_csv(CACHE / f"graded_{args.season}.csv")
    weeks = [args.week] if args.week else sorted(g.filter(pl.col("result") != "void")["week"].unique().to_list())
    for wk in weeks:
        report(g, wk, args.top)
    print(f"\nwrote {CACHE.relative_to(ROOT)}/graded_{args.season}.csv")


if __name__ == "__main__":
    main()
