"""Grade logged prop lines and the model's pre-game projections against actual results.

Two questions, answered separately:
1. Lines alone (every props_log/<stamp>.csv): how often did the over hit, and what would betting
   every over or every under at the listed price have returned? If unders win well above 50%,
   the books shade toward overs and the model's under lean may be real.
2. The model (props_log/projections/<stamp>.csv, written by props_week.py before kickoff): is
   p_model better calibrated than the books' no-vig probability (Brier score, lower is better),
   and what did its positive-EV picks (best book per player-market) return?

A player who didn't play is a void (books refund). "Played" means an offensive snap or a stat
line, so an active receiver with no catches grades as 0, not as a void. Run after nflverse has
the week's stats (`python scripts/pull_data.py` first, usually the morning after the games).

    python scripts/grade_props.py               # every snapshot whose games are final
    python scripts/grade_props.py 20261004      # only snapshots whose name starts with this
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

import props
from props_week import TEAMS, MARKET_KEYS, norm_name, payout

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG = ROOT / "props_log"
STAT = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards",
        "receptions": "receptions"}
EV_CUTS = [0.0, 0.05, 0.10, 0.20]


def participants() -> pl.DataFrame:
    """Everyone who played in each game: game_id, player_id, key, team, and the four stats (0 if none)."""
    ps = (pl.read_parquet(DATA / "raw" / "player_stats.parquet")
          .select("game_id", "player_id", "player_display_name", pl.col("team").replace(props.RENAMES),
                  *[pl.col(c).fill_null(0).cast(pl.Float64) for c in STAT.values()])
          .with_columns(pl.col("player_display_name").map_elements(norm_name, return_dtype=pl.String).alias("key"))
          .drop("player_display_name"))
    ids = pl.read_parquet(DATA / "raw" / "players.parquet").select("gsis_id", "pfr_id").drop_nulls().unique("pfr_id")
    snaps = (pl.read_parquet(DATA / "raw" / "snap_counts.parquet")
             .filter(pl.col("offense_snaps") > 0)
             .join(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
             .select("game_id", pl.col("gsis_id").alias("player_id"), pl.col("team").replace(props.RENAMES),
                     pl.col("player").map_elements(norm_name, return_dtype=pl.String).alias("key")))
    extra = (snaps.join(ps, on=["game_id", "key"], how="anti")
             .with_columns([pl.lit(0.0).alias(c) for c in STAT.values()]))
    return pl.concat([ps, extra.select(ps.columns)]).unique(["game_id", "key"], keep="first")


def actual_col() -> pl.Expr:
    return pl.coalesce([pl.when(pl.col("mkt") == m).then(pl.col(c)) for m, c in STAT.items()]).alias("actual")


def settle(df: pl.DataFrame) -> pl.DataFrame:
    """Add result (1 win, 0 push, -1 loss) and profit per unit for each side/price row."""
    diff = pl.col("actual") - pl.col("point")
    won = pl.when(pl.col("side") == "Over").then(diff > 0).otherwise(diff < 0)
    return df.with_columns(
        pl.when(diff == 0).then(0).when(won).then(1).otherwise(-1).alias("result")
    ).with_columns(
        pl.when(pl.col("result") == 1).then(pl.col("price").map_elements(payout, return_dtype=pl.Float64))
        .when(pl.col("result") == 0).then(0.0).otherwise(-1.0).alias("profit"))


def record(df: pl.DataFrame) -> str:
    w, l, p = (df["result"] == 1).sum(), (df["result"] == -1).sum(), (df["result"] == 0).sum()
    units = df["profit"].sum()
    pct = w / (w + l) if w + l else float("nan")
    return f"{w}-{l}-{p} ({pct:.1%}), {units:+.1f}u, ROI {units / max(len(df), 1):+.1%}"


def finished_games(sched: pl.DataFrame) -> pl.DataFrame:
    return (sched.filter(pl.col("result").is_not_null())
            .select("game_id", "season", "week", "gameday", pl.col("home_team").alias("home"),
                    pl.col("away_team").alias("away")))


def grade_lines(path: Path, games: pl.DataFrame, people: pl.DataFrame) -> pl.DataFrame | None:
    snap = (pl.read_csv(path)
            .with_columns(pl.col("home_team").replace_strict(TEAMS).alias("home"),
                          pl.col("away_team").replace_strict(TEAMS).alias("away"),
                          pl.col("player").map_elements(norm_name, return_dtype=pl.String).alias("key"),
                          pl.col("market").replace_strict(MARKET_KEYS).alias("mkt"),
                          pl.col("commence_time").str.to_datetime(time_zone="UTC")
                          .dt.convert_time_zone("America/New_York").dt.date().cast(pl.String).alias("gameday"))
            .join(games, on=["home", "away", "gameday"]))
    if snap.is_empty():
        return None
    # nflverse lags a day or so; grade only games whose stats are in
    have = people.select("game_id").unique()
    snap = snap.join(have, on="game_id")
    if snap.is_empty():
        return None
    g = snap.join(people.drop("team"), on=["game_id", "key"], how="left")
    void = g.filter(pl.col("receptions").is_null())
    g = settle(g.filter(pl.col("receptions").is_not_null()).with_columns(actual_col()))
    pm = g.select("game_id", "key", "mkt").unique().height
    print(f"\n=== {path.name}: {g['game_id'].n_unique()} games graded, {pm} player-markets, "
          f"{void.select('key', 'mkt').unique().height} void (didn't play or name not found) ===")
    print("Lines alone, every book line (over hit rate excludes pushes):")
    for mkt in sorted(g["mkt"].unique()):
        o = g.filter((pl.col("mkt") == mkt) & (pl.col("side") == "Over") & (pl.col("result") != 0))
        u = g.filter((pl.col("mkt") == mkt) & (pl.col("side") == "Under"))
        print(f"  {mkt:<10} over hit {(o['result'] == 1).mean():.1%} of {len(o)} | all overs: "
              f"{record(g.filter((pl.col('mkt') == mkt) & (pl.col('side') == 'Over')))} | all unders: {record(u)}")
    print(f"  {'total':<10} all overs: {record(g.filter(pl.col('side') == 'Over'))} | "
          f"all unders: {record(g.filter(pl.col('side') == 'Under'))}")
    return g


def grade_model(path: Path, people: pl.DataFrame) -> pl.DataFrame | None:
    proj = pl.read_csv(path)
    if "game_id" not in proj.columns:
        print(f"\n{path.name}: projections lack game_id/player_id (written by an older props_week.py); skipped")
        return None
    g = proj.join(people.select("game_id", "player_id", *STAT.values()), on=["game_id", "player_id"])
    if g.is_empty():
        return None
    g = settle(g.with_columns(actual_col()))
    print(f"\nModel, {path.name}: {g.select('game_id', 'player_id', 'mkt').unique().height} player-markets graded "
          f"({proj.select('player_id', 'mkt').unique().height - g.select('player_id', 'mkt').unique().height} void)")

    o = g.filter((pl.col("side") == "Over") & (pl.col("result") != 0) & pl.col("p_book_fair").is_not_null())
    y = (o["result"] == 1).cast(pl.Float64).to_numpy()
    bm, bb = np.mean((o["p_model"].to_numpy() - y) ** 2), np.mean((o["p_book_fair"].to_numpy() - y) ** 2)
    print(f"  calibration on {len(o)} book lines: Brier model {bm:.4f} vs book {bb:.4f} "
          f"({'model better' if bm < bb else 'book better'}); mean P(over) model {o['p_model'].mean():.3f}, "
          f"book {o['p_book_fair'].mean():.3f}, actual {y.mean():.3f}")

    best = (g.sort("ev", descending=True).group_by("game_id", "player_id", "mkt", maintain_order=True).first())
    for label, subset in (("no news flag", best.filter(~pl.col("news").fill_null(False))),
                          ("news-flagged", best.filter(pl.col("news").fill_null(False)))):
        print(f"  best bet per player-market, {label}:")
        for cut in EV_CUTS:
            b = subset.filter(pl.col("ev") > cut)
            if len(b):
                print(f"    EV > {cut:.2f}: {record(b)}   [overs {record(b.filter(pl.col('side') == 'Over'))}; "
                      f"unders {record(b.filter(pl.col('side') == 'Under'))}]")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="?", default="", help="only snapshots whose name starts with this")
    args = ap.parse_args()
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
             .with_columns(pl.col("home_team", "away_team").replace(props.RENAMES)))
    games, people = finished_games(sched), participants()
    out = LOG / "grades"
    out.mkdir(exist_ok=True)
    found = False
    for path in sorted(LOG.glob(f"{args.prefix}*.csv")):
        g = grade_lines(path, games, people)
        if g is None:
            continue
        found = True
        g.drop("key", "home", "away", "player_id").write_csv(out / f"{path.stem}_lines.csv")
        pj = LOG / "projections" / path.name
        if pj.exists():
            m = grade_model(pj, people)
            if m is not None:
                m.write_csv(out / f"{path.stem}_model.csv")
        else:
            print(f"\n(no saved projections for {path.name}, so the model isn't graded on it)")
    if not found:
        print("No snapshot has finished games with stats yet; pull fresh data and try again.")
    else:
        print(f"\nper-line results written to {out}/")


if __name__ == "__main__":
    main()
