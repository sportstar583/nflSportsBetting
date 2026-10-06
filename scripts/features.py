"""Build game-level features (no look-ahead) from data/raw/ -> data/features.parquet."""
from pathlib import Path

import polars as pl

import adjust

DATA = Path(__file__).resolve().parent.parent / "data"
WINDOW = 8  # games, rolling across seasons


# Missing outdoor weather (half of 2022, every future game) gets the outdoor median and a flag,
# instead of 0 mph / 70F, which made it look like a calm day.
OUTDOOR_MEDIAN_WIND, OUTDOOR_MEDIAN_TEMP = 8.0, 57.0
WIND_KNEE = 10  # mph; totals drop above it
WIND_CAP = 30   # a few readings (e.g. 44 mph, 2023 SF@PHI) look like data errors; cap their leverage

# schedules/injuries use the old abbreviation; pbp uses the current one throughout
RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

STATS = ["off_epa", "off_sr", "def_epa", "def_sr", "pf", "pa",
         "off_pass_epa", "off_run_epa", "def_pass_epa", "def_run_epa", "off_plays", "off_pass_rate"]


def team_game_epa() -> pl.DataFrame:
    """Per team-game EPA/play (overall, pass, run), success rate and pace (plays, pass rate)."""
    pbp = (
        pl.scan_parquet(DATA / "raw" / "pbp" / "*.parquet")
        .filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null())
        .select("game_id", "posteam", "defteam", "play_type", "epa", "success")
        .collect()
    )

    def split(side: str, prefix: str) -> pl.DataFrame:
        pt = pl.col("play_type")
        aggs = [
            pl.col("epa").mean().alias(f"{prefix}_epa"),
            pl.col("success").mean().alias(f"{prefix}_sr"),
            pl.col("epa").filter(pt == "pass").mean().alias(f"{prefix}_pass_epa"),
            pl.col("epa").filter(pt == "run").mean().alias(f"{prefix}_run_epa"),
        ]
        if prefix == "off":
            aggs += [pl.len().alias("off_plays"), (pt == "pass").mean().alias("off_pass_rate")]
        return pbp.group_by("game_id", side).agg(aggs).rename({side: "team"})

    return split("posteam", "off").join(split("defteam", "def"), on=["game_id", "team"])


def rolling_team_form(sched: pl.DataFrame, epa: pl.DataFrame) -> pl.DataFrame:
    """Rolling mean of prior WINDOW games per team (shifted so the current game is excluded)."""
    long = pl.concat([
        sched.select("game_id", "gameday", pl.col("home_team").alias("team"),
                     pl.col("home_score").alias("pf"), pl.col("away_score").alias("pa")),
        sched.select("game_id", "gameday", pl.col("away_team").alias("team"),
                     pl.col("away_score").alias("pf"), pl.col("home_score").alias("pa")),
    ]).join(epa, on=["game_id", "team"], how="left").sort("team", "gameday")
    stats = STATS
    return long.with_columns(
        [pl.col(c).shift(1).rolling_mean(WINDOW, min_samples=3).over("team").alias(f"r_{c}")
         for c in stats]
    ).select("game_id", "team", *[f"r_{c}" for c in stats])


QB_PRIOR_DB = 200  # dropbacks of shrinkage toward the league-average QB


def qb_form(sched: pl.DataFrame) -> pl.DataFrame:
    """Shrunk career EPA/dropback for each QB entering each game (prior games only)."""
    db = (
        pl.scan_parquet(DATA / "raw" / "pbp" / "*.parquet")
        .filter((pl.col("qb_dropback") == 1) & pl.col("qb_epa").is_not_null()
                & pl.col("passer_player_id").is_not_null())
        .group_by("game_id", "passer_player_id")
        .agg(pl.col("qb_epa").sum().alias("epa_sum"), pl.len().alias("n"))
        .collect()
        .join(sched.select("game_id", "gameday"), on="game_id")
        .sort("passer_player_id", "gameday")
    )
    league = db["epa_sum"].sum() / db["n"].sum()
    db = db.with_columns(
        pl.col("epa_sum").cum_sum().shift(1).over("passer_player_id").fill_null(0).alias("p_epa"),
        pl.col("n").cum_sum().shift(1).over("passer_player_id").fill_null(0).alias("p_n"),
    ).with_columns(
        ((pl.col("p_epa") + QB_PRIOR_DB * league) / (pl.col("p_n") + QB_PRIOR_DB)).alias("qb_epa"),
    )
    return db.select("game_id", pl.col("passer_player_id").alias("qb_id"), "qb_epa",
                     pl.col("p_n").alias("qb_n")), league


INJ_GROUPS = {  # QB is left out: the starting-QB rating already covers it
    "ol": (["T", "G", "C", "OL", "OT", "OG"], "off_share"),
    "skill": (["WR", "TE", "RB", "FB"], "off_share"),
    "def": (["DE", "DT", "NT", "DL", "LB", "OLB", "ILB", "MLB", "CB", "S", "FS", "SS", "DB"], "def_share"),
    # finer groups for player props: who gets the vacated targets/carries, and which unit is depleted
    "wr": (["WR"], "off_share"), "rb": (["RB", "FB"], "off_share"), "te": (["TE"], "off_share"),
    "db": (["CB", "S", "FS", "SS", "DB"], "def_share"),
    "front": (["DE", "DT", "NT", "DL", "LB", "OLB", "ILB", "MLB"], "def_share"),
}
SNAP_GAMES = 4  # a player's role = mean snap share over his last 4 games played


def injured_players() -> pl.DataFrame:
    """Out/Doubtful players per team-week, each with his snap share (off_share, def_share) over
    his last SNAP_GAMES games before that week: a starter is ~1.0, a backup ~0.1. Players with no
    snaps in the past year (practice squad, long-term absences already in team form) have null.
    """
    key = (pl.col("season").cast(pl.Int32) * 100 + pl.col("week").cast(pl.Int32)) * 10
    snaps = (
        pl.read_parquet(DATA / "raw" / "snap_counts.parquet")
        .with_columns(key.alias("k"))
        .sort("pfr_player_id", "k")
        .with_columns(
            pl.col("offense_pct").rolling_mean(SNAP_GAMES, min_samples=1).over("pfr_player_id").alias("off_share"),
            pl.col("defense_pct").rolling_mean(SNAP_GAMES, min_samples=1).over("pfr_player_id").alias("def_share"),
        )
        .select("pfr_player_id", "k", "off_share", "def_share")
    )
    ids = (pl.read_parquet(DATA / "raw" / "players.parquet")
           .select("gsis_id", "pfr_id").drop_nulls().unique("gsis_id"))
    inj = (
        pl.read_parquet(DATA / "raw" / "injuries.parquet")
        .filter(pl.col("report_status").is_in(["Out", "Doubtful"]) & pl.col("game_type").is_not_null())
        .unique(["season", "week", "team", "gsis_id"])
        .with_columns(pl.col("team").replace(RENAMES), (key - 1).alias("k"),
                      pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
        .join(ids, on="gsis_id", how="left")
        .sort("k")
        .join_asof(snaps.sort("k"), on="k", by_left="pfr_id", by_right="pfr_player_id",
                   strategy="backward", tolerance=1000)  # last game strictly before, within ~a year
    )
    return inj


def injury_snaps() -> pl.DataFrame:
    """Snap share lost to Out/Doubtful players per team-week, by group (see injured_players)."""
    inj = injured_players()
    lost = []
    for g, (positions, share) in INJ_GROUPS.items():
        lost.append(pl.col(share).filter(pl.col("position").is_in(positions)).fill_null(0).sum().alias(f"inj_{g}"))
    return inj.group_by("season", "week", "team").agg(lost)


def build() -> pl.DataFrame:
    sched = (
        pl.read_parquet(DATA / "raw" / "schedules.parquet")
        .filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))
        .with_columns(pl.col("home_team", "away_team").replace(RENAMES))
    )
    form = rolling_team_form(sched, team_game_epa())
    stats = STATS
    h = form.rename({c: f"h_r_{c[2:]}" for c in form.columns if c.startswith("r_")})
    a = form.rename({c: f"a_r_{c[2:]}" for c in form.columns if c.startswith("r_")})
    df = (
        sched.join(h, left_on=["game_id", "home_team"], right_on=["game_id", "team"])
        .join(a, left_on=["game_id", "away_team"], right_on=["game_id", "team"])
    )
    qb, league = qb_form(sched)
    for side in ("home", "away"):
        q = qb.rename({"qb_epa": f"{side}_qb_epa", "qb_n": f"{side}_qb_n"})
        df = df.join(q, left_on=["game_id", f"{side}_qb_id"], right_on=["game_id", "qb_id"], how="left")
    df = df.with_columns(  # QB with no prior dropbacks in the data -> league-average, n=0
        pl.col("home_qb_epa").fill_null(league), pl.col("away_qb_epa").fill_null(league),
        pl.col("home_qb_n").fill_null(0), pl.col("away_qb_n").fill_null(0),
    ).with_columns(
        (pl.col("home_qb_epa") - pl.col("away_qb_epa")).alias("d_qb_epa"),
        (pl.col("home_qb_n").clip(0, 1500) - pl.col("away_qb_n").clip(0, 1500)).alias("d_qb_exp"),
    )
    inj = injury_snaps()
    inj_cols = [c for c in inj.columns if c.startswith("inj_")]
    for side in ("home", "away"):
        i = inj.rename({"team": f"{side}_team", **{c: f"{side}_{c}" for c in inj_cols}})
        df = df.join(i, on=["season", "week", f"{side}_team"], how="left")
    df = df.with_columns(
        [pl.col(f"{s}_{c}").fill_null(0) for s in ("home", "away") for c in inj_cols]
    ).with_columns(
        [(pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"d_{c}") for c in inj_cols]
        + [(pl.col(f"home_{c}") + pl.col(f"away_{c}")).alias(f"s_{c}") for c in inj_cols]
    )
    tg = adjust.team_games(DATA / "raw" / "pbp" / "*.parquet", sched)
    adj = adjust.adjusted_ratings(tg, sched)
    adj_cols = [c for c in adj.columns if c not in ("game_id", "team")]
    for side, pre in (("home", "h"), ("away", "a")):
        df = df.join(adj.rename({c: f"{pre}_{c}" for c in adj_cols}),
                     left_on=["game_id", f"{side}_team"], right_on=["game_id", "team"], how="left")
    adj_feats = []
    for sp in adjust.SPLITS:
        o, d = f"adj_off_{sp}", f"adj_def_{sp}"
        adj_feats += [  # net rating gap (spreads) and combined expected EPA (totals)
            ((pl.col(f"h_{o}") - pl.col(f"h_{d}")) - (pl.col(f"a_{o}") - pl.col(f"a_{d}"))).alias(f"d_adj_{sp}"),
            (pl.col(f"h_{o}") + pl.col(f"a_{d}") + pl.col(f"a_{o}") + pl.col(f"h_{d}")).alias(f"s_adj_{sp}"),
        ]
    adj_feats += [(pl.col("h_sec_per_play") + pl.col("a_sec_per_play")).alias("s_sec_per_play"),
                  (pl.col("h_sec_per_play") - pl.col("a_sec_per_play")).alias("d_sec_per_play")]
    df = df.with_columns(adj_feats)
    # weather: future games have no roof listed, so take the stadium's last known roof
    df = df.sort("gameday").with_columns(pl.col("roof").fill_null(strategy="forward").over("stadium_id"))
    indoor = pl.col("roof").is_in(["dome", "closed"]).fill_null(False).cast(pl.Int8)
    missing = ((indoor == 0) & pl.col("wind").is_null()).cast(pl.Int8)
    wind = (pl.when(indoor == 1).then(pl.lit(0.0))
            .otherwise(pl.col("wind").fill_null(OUTDOOR_MEDIAN_WIND).clip(upper_bound=WIND_CAP)))
    diffs = [(pl.col(f"h_r_{c}") - pl.col(f"a_r_{c}")).alias(f"d_{c}") for c in stats]
    sums = [(pl.col(f"h_r_{c}") + pl.col(f"a_r_{c}")).alias(f"s_{c}") for c in stats]
    df = df.with_columns(
        *diffs, *sums,
        (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
        indoor.alias("indoor"),
        missing.alias("weather_missing"),
        wind.alias("wind"),
        ((wind - WIND_KNEE).clip(lower_bound=0)).alias("wind_over_10"),
        pl.when(indoor == 1).then(pl.lit(70.0)).otherwise(pl.col("temp").fill_null(OUTDOOR_MEDIAN_TEMP)).alias("temp"),
        (pl.col("location") == "Neutral").cast(pl.Int8).alias("neutral"),
        pl.col("div_game").cast(pl.Int8),
    )
    keep = ["game_id", "season", "week", "gameday", "home_team", "away_team", "result", "total",
            "spread_line", "total_line", "home_moneyline", "away_moneyline",
            "rest_diff", "indoor", "wind", "wind_over_10", "weather_missing", "temp", "neutral", "div_game",
            *[f"d_{c}" for c in stats], *[f"s_{c}" for c in stats],
            "d_qb_epa", "d_qb_exp", *[f"d_{c}" for c in inj_cols], *[f"s_{c}" for c in inj_cols],
            *[f"{p}_adj_{sp}" for sp in adjust.SPLITS for p in ("d", "s")],
            "s_sec_per_play", "d_sec_per_play"]
    return df.select(keep).sort("gameday")


if __name__ == "__main__":
    out = build()
    out.write_parquet(DATA / "features.parquet")
    print(f"features: {out.height} rows, {out.width} cols")
