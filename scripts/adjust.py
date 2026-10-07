"""Opponent-adjusted EPA ratings and tempo, computed walk-forward (no look-ahead).

For each (season, week), fit on team-games played before that week:

    EPA/play of offense o vs defense d = mean + off[o] + def[d] + home_field

with a ridge regression weighted by plays and decayed by age. Ratings are what a team
would do against an average opponent at a neutral site; def is EPA allowed (lower is better).
Done separately for all plays, passes and runs.
"""
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

ALPHA = 300           # ridge penalty, in plays: pulls thinly-sampled teams toward average
HALF_LIFE_DAYS = 180  # last season's games count ~1/4 by the next opener
MAX_AGE_DAYS = 730
SPLITS = {"all": ("epa", "n"), "pass": ("pass_epa", "n_pass"), "run": ("run_epa", "n_run")}


# Garbage time: second-half plays with the offense's win probability under 10% or over 90%
# (21% of plays). Trailing teams pass 80% of the time against soft coverage, leaders run out the clock.
GARBAGE = (pl.col("qtr") >= 3) & ((pl.col("wp") < 0.10) | (pl.col("wp") > 0.90))


def team_games(pbp_path, sched: pl.DataFrame, drop_garbage: bool = False) -> pl.DataFrame:
    """One row per offense per game: EPA/play by split, play counts and seconds per play.
    drop_garbage leaves garbage-time plays out of the EPA and play counts."""
    keep = ~GARBAGE.fill_null(False) if drop_garbage else pl.lit(True)
    pbp = (
        pl.scan_parquet(pbp_path)
        .filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null() & keep)
        .select("game_id", "posteam", "defteam", "play_type", "epa", "drive",
                "drive_play_count", "drive_time_of_possession")
        .collect()
    )
    pt = pl.col("play_type")
    off = pbp.group_by("game_id", "posteam", "defteam").agg(
        pl.col("epa").mean().alias("epa"), pl.len().alias("n"),
        pl.col("epa").filter(pt == "pass").mean().alias("pass_epa"), (pt == "pass").sum().alias("n_pass"),
        pl.col("epa").filter(pt == "run").mean().alias("run_epa"), (pt == "run").sum().alias("n_run"),
    )
    # tempo: seconds per play from drive clock, one value per drive
    mmss = pl.col("drive_time_of_possession").str.split(":")
    drives = (
        pbp.drop_nulls(["drive", "drive_time_of_possession", "drive_play_count"])
        .unique(["game_id", "posteam", "drive"])
        .with_columns((mmss.list.get(0).cast(pl.Int32) * 60 + mmss.list.get(1).cast(pl.Int32)).alias("secs"))
        .group_by("game_id", "posteam")
        .agg((pl.col("secs").sum() / pl.col("drive_play_count").sum()).alias("sec_per_play"))
    )
    g = sched.select("game_id", "gameday", "season", "week", "home_team", "location")
    return (
        off.join(drives, on=["game_id", "posteam"], how="left")
        .join(g, on="game_id")
        .with_columns(
            pl.when(pl.col("location") == "Neutral").then(0)
            .when(pl.col("posteam") == pl.col("home_team")).then(1).otherwise(-1).alias("home"),
            pl.col("gameday").str.to_date().alias("date"),
        )
    )


def _fit(train: pl.DataFrame, teams: list[str], y_col: str, n_col: str, asof) -> dict:
    t = train.drop_nulls(y_col).filter(pl.col(n_col) > 0)
    idx = {tm: i for i, tm in enumerate(teams)}
    k = len(teams)
    X = np.zeros((t.height, 2 * k + 1))
    rows = np.arange(t.height)
    X[rows, [idx[o] for o in t["posteam"]]] = 1
    X[rows, [k + idx[d] for d in t["defteam"]]] = 1
    X[:, -1] = t["home"].to_numpy()
    age = (asof - t["date"]).dt.total_days().to_numpy()
    w = t[n_col].to_numpy() * 0.5 ** (age / HALF_LIFE_DAYS)
    m = Ridge(alpha=ALPHA).fit(X, t[y_col].to_numpy(), sample_weight=w)
    return {tm: (m.coef_[idx[tm]], m.coef_[k + idx[tm]]) for tm in teams}


def adjusted_ratings(tg: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Pre-game adjusted ratings and tempo for every team in every scheduled game."""
    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    games = sched.select("game_id", "season", "week", "home_team", "away_team",
                         pl.col("gameday").str.to_date().alias("date"))
    out = []
    for (season, week), grp in games.group_by("season", "week", maintain_order=True):
        asof = grp["date"].min()
        train = tg.filter((pl.col("date") < asof)
                          & ((asof - pl.col("date")).dt.total_days() <= MAX_AGE_DAYS))
        if train.height < 200:
            continue
        fits = {s: _fit(train, teams, y, n, asof) for s, (y, n) in SPLITS.items()}
        # tempo: decayed mean seconds/play for each offense
        tempo = (
            train.drop_nulls("sec_per_play")
            .with_columns((0.5 ** ((asof - pl.col("date")).dt.total_days() / HALF_LIFE_DAYS)).alias("w"))
            .group_by("posteam")
            .agg(((pl.col("sec_per_play") * pl.col("w")).sum() / pl.col("w").sum()).alias("sec_per_play"))
        )
        tempo = dict(zip(tempo["posteam"], tempo["sec_per_play"]))
        for gid, h, a in grp.select("game_id", "home_team", "away_team").iter_rows():
            for tm in (h, a):
                row = {"game_id": gid, "team": tm, "sec_per_play": tempo.get(tm)}
                for s, f in fits.items():
                    row[f"adj_off_{s}"], row[f"adj_def_{s}"] = f[tm]
                out.append(row)
    return pl.DataFrame(out)
