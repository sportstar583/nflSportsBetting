"""Build game-level features (no look-ahead) from data/raw/ -> data/features.parquet."""
from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data"
WINDOW = 8  # games, rolling across seasons


def team_game_epa() -> pl.DataFrame:
    """Per team-game offensive/defensive EPA/play and success rate."""
    pbp = (
        pl.scan_parquet(DATA / "raw" / "pbp.parquet")
        .filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null())
        .select("game_id", "posteam", "defteam", "epa", "success")
        .collect()
    )
    off = pbp.group_by("game_id", "posteam").agg(
        pl.col("epa").mean().alias("off_epa"), pl.col("success").mean().alias("off_sr")
    ).rename({"posteam": "team"})
    dfn = pbp.group_by("game_id", "defteam").agg(
        pl.col("epa").mean().alias("def_epa"), pl.col("success").mean().alias("def_sr")
    ).rename({"defteam": "team"})
    return off.join(dfn, on=["game_id", "team"])


def rolling_team_form(sched: pl.DataFrame, epa: pl.DataFrame) -> pl.DataFrame:
    """Rolling mean of prior WINDOW games per team (shifted so the current game is excluded)."""
    long = pl.concat([
        sched.select("game_id", "gameday", pl.col("home_team").alias("team"),
                     pl.col("home_score").alias("pf"), pl.col("away_score").alias("pa")),
        sched.select("game_id", "gameday", pl.col("away_team").alias("team"),
                     pl.col("away_score").alias("pf"), pl.col("home_score").alias("pa")),
    ]).join(epa, on=["game_id", "team"], how="left").sort("team", "gameday")
    stats = ["off_epa", "off_sr", "def_epa", "def_sr", "pf", "pa"]
    return long.with_columns(
        [pl.col(c).shift(1).rolling_mean(WINDOW, min_samples=3).over("team").alias(f"r_{c}")
         for c in stats]
    ).select("game_id", "team", *[f"r_{c}" for c in stats])


def build() -> pl.DataFrame:
    sched = (
        pl.read_parquet(DATA / "raw" / "schedules.parquet")
        .filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))
    )
    form = rolling_team_form(sched, team_game_epa())
    stats = ["off_epa", "off_sr", "def_epa", "def_sr", "pf", "pa"]
    h = form.rename({c: f"h_r_{c[2:]}" for c in form.columns if c.startswith("r_")})
    a = form.rename({c: f"a_r_{c[2:]}" for c in form.columns if c.startswith("r_")})
    df = (
        sched.join(h, left_on=["game_id", "home_team"], right_on=["game_id", "team"])
        .join(a, left_on=["game_id", "away_team"], right_on=["game_id", "team"])
    )
    diffs = [(pl.col(f"h_r_{c}") - pl.col(f"a_r_{c}")).alias(f"d_{c}") for c in stats]
    df = df.with_columns(
        *diffs,
        (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
        (pl.col("roof").is_in(["dome", "closed"])).cast(pl.Int8).alias("indoor"),
        pl.col("wind").fill_null(0).alias("wind"),
        pl.col("temp").fill_null(70).alias("temp"),
        (pl.col("location") == "Neutral").cast(pl.Int8).alias("neutral"),
        pl.col("div_game").cast(pl.Int8),
    )
    keep = ["game_id", "season", "week", "gameday", "home_team", "away_team", "result",
            "spread_line", "total_line", "home_moneyline", "away_moneyline",
            "rest_diff", "indoor", "wind", "temp", "neutral", "div_game",
            *[f"d_{c}" for c in stats]]
    return df.select(keep).sort("gameday")


if __name__ == "__main__":
    out = build()
    out.write_parquet(DATA / "features.parquet")
    print(f"features: {out.height} rows, {out.width} cols")
