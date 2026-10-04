"""Recency bias: does a bad week predict another bad week against the spread, or a bounce-back?

If bettors overreact to last week, a team that got blown out (vs. the spread) should be
undervalued this week and cover more than 50%. If bad form carries over, it should cover less.
Uses closing spreads, 2018-present, same-season consecutive games only (no week 1 carryover).
"""
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data"
RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
BUCKETS = [-100, -14, -7, -3, 3, 7, 14, 100]  # last week's ATS margin, points


def team_games() -> pl.DataFrame:
    """One row per team per game: ATS margin (points by which the team beat the spread)."""
    s = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
         .filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null())
         .with_columns(pl.col("home_team", "away_team").replace(RENAMES)))
    ats = pl.col("result") - pl.col("spread_line")  # home beat the spread by this much
    home = s.select("game_id", "season", "week", "gameday", pl.col("home_team").alias("team"),
                    pl.col("away_team").alias("opp"), ats.alias("ats"), pl.col("result").alias("margin"))
    away = s.select("game_id", "season", "week", "gameday", pl.col("away_team").alias("team"),
                    pl.col("home_team").alias("opp"), (-ats).alias("ats"), (-pl.col("result")).alias("margin"))
    return (pl.concat([home, away]).sort("team", "gameday")
            .with_columns(pl.col("ats", "margin").shift(1).over("team", "season").name.prefix("prev_"),
                          pl.col("ats").shift(2).over("team", "season").alias("prev2_ats")))


def summarize(df: pl.DataFrame, label: str) -> None:
    c = df["ats"].sign()
    w, l = int((c > 0).sum()), int((c < 0).sum())
    pct = w / max(w + l, 1)
    se = 0.5 / np.sqrt(max(w + l, 1))
    print(f"{label:<38} n={w + l:5d}  cover={pct:.3f} (+/-{1.96 * se:.3f})  "
          f"mean ATS={df['ats'].mean():+.2f}  units at -110={w * 100 / 110 - l:+.1f}")


def main():
    tg = team_games()
    d = tg.drop_nulls("prev_ats")
    print(f"team-games with a previous same-season game: {d.height}\n")

    r = np.corrcoef(d["prev_ats"], d["ats"])[0, 1]
    print(f"correlation of last week's ATS margin with this week's: {r:+.3f}\n")

    print("This week's cover rate by LAST WEEK's ATS margin")
    labels = [f"{lo:+d} to {hi:+d}" for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])]
    for (lo, hi), lab in zip(zip(BUCKETS[:-1], BUCKETS[1:]), labels):
        summarize(d.filter((pl.col("prev_ats") >= lo) & (pl.col("prev_ats") < hi)), f"  last week ATS {lab}")

    print("\nStraight-up blowouts last week")
    summarize(d.filter(pl.col("prev_margin") <= -17), "  lost by 17+ last week")
    summarize(d.filter(pl.col("prev_margin") >= 17), "  won by 17+ last week")

    print("\nTwo bad weeks in a row")
    two = d.drop_nulls("prev2_ats")
    summarize(two.filter((pl.col("prev_ats") < 0) & (pl.col("prev2_ats") < 0)), "  failed to cover both of last 2")
    summarize(two.filter((pl.col("prev_ats") > 0) & (pl.col("prev2_ats") > 0)), "  covered both of last 2")
    summarize(two.filter((pl.col("prev_ats") <= -7) & (pl.col("prev2_ats") <= -7)), "  missed by 7+ in both of last 2")

    # A bettable rule: back a team after it missed the spread by 14+, unless its opponent also did.
    print("\nRule: bet on a team that missed the spread by 14+ last week (opponent didn't)")
    opp = d.select("game_id", pl.col("team").alias("opp"), pl.col("prev_ats").alias("opp_prev_ats"))
    rule = (d.join(opp, on=["game_id", "opp"], how="left")
            .filter((pl.col("prev_ats") <= -14) & ~(pl.col("opp_prev_ats").fill_null(0) <= -14)))
    for season, g in rule.group_by("season", maintain_order=True):
        summarize(g, f"  {season[0]}")
    summarize(rule, "  all seasons")


if __name__ == "__main__":
    main()
