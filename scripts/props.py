"""Player prop projections: passing yards, rushing yards, receiving yards, receptions.

Projection = opportunity (snap share, target/carry share) x efficiency (yards per target/carry)
x game environment (implied team total, spread, pace, weather) x opponent (adjusted defense,
yards allowed to the position) x teammate injuries. Every input is known before kickoff
(rolling stats are shifted so the current game is excluded).

Models predict the MEDIAN (absolute-error loss), since an over/under is won at the median.
Walk-forward: each test season is predicted from a model trained on earlier seasons.

No historical prop lines yet, so this measures accuracy against naive baselines only. Beating
a recent average is necessary but nowhere near sufficient to beat a sportsbook's line.
"""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

import adjust

DATA = Path(__file__).resolve().parent.parent / "data"
RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
HALF_LIFE = 4  # games; recent games count more, older games fade
TEST_SEASONS = [2022, 2023, 2024, 2025]
SKILL = ["QB", "RB", "WR", "TE"]


def ewm(col: str, by: str) -> pl.Expr:
    """Exponentially weighted mean of prior games only (shifted), per `by`."""
    return pl.col(col).shift(1).ewm_mean(half_life=HALF_LIFE, ignore_nulls=True).over(by)


def load(upcoming: pl.DataFrame | None = None, wind: dict | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`upcoming`: stat-less player rows for unplayed games to project (see props_week.py);
    `wind`: game_id -> forecast mph for those games, since nflverse only fills wind after kickoff."""
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
             .with_columns(pl.col("home_team", "away_team").replace(RENAMES)))
    if wind:
        sched = sched.with_columns(pl.col("game_id").replace_strict(wind, default=None, return_dtype=pl.Float64)
                                   .fill_null(pl.col("wind").cast(pl.Float64)).alias("wind"))
    ps = pl.read_parquet(DATA / "raw" / "player_stats.parquet")
    if upcoming is not None:
        ps = pl.concat([ps, upcoming], how="diagonal_relaxed")
    ps = (ps.filter(pl.col("position").is_in(SKILL))
          .with_columns(pl.col("team", "opponent_team").replace(RENAMES))
          .join(sched.select("game_id", "gameday"), on="game_id"))
    return sched, ps


def team_context(sched: pl.DataFrame, ps: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: pre-game implied total, spread, weather, pace and pass rate."""
    side = []
    for home in (True, False):
        t, o = ("home_team", "away_team") if home else ("away_team", "home_team")
        sgn = 1 if home else -1
        side.append(sched.select(
            "game_id", "gameday", "week", pl.col(t).alias("team"), pl.col(o).alias("opp"),
            # schedule situation: days of rest (short week), coming off a bye
            pl.col("home_rest" if home else "away_rest").cast(pl.Float64).alias("rest"),
            (pl.col("total_line") / 2 + sgn * pl.col("spread_line") / 2).alias("implied_total"),
            (sgn * pl.col("spread_line")).alias("team_spread"),  # + means team favored
            pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8).alias("indoor"),
            pl.when(pl.col("roof").is_in(["dome", "closed"])).then(0.0)
            .otherwise(pl.col("wind").cast(pl.Float64)).alias("wind"),
        ))
    ctx = (pl.concat(side).sort("team", "gameday")
           .with_columns((pl.col("rest") <= 5).cast(pl.Int8).alias("short_rest"),
                         ((pl.col("week") - pl.col("week").shift(1).over("team") >= 2) & (pl.col("week") > 1))
                         .fill_null(False).cast(pl.Int8).alias("off_bye"))
           .drop("week"))
    vol = ps.group_by("game_id", "team").agg(
        pl.col("attempts").sum().alias("team_att"), pl.col("carries").sum().alias("team_car"),
        pl.col("targets").sum().alias("team_tgt"))
    ctx = (ctx.join(vol, on=["game_id", "team"], how="left").sort("team", "gameday")
           .with_columns((pl.col("team_att") + pl.col("team_car")).alias("team_plays"),
                         (pl.col("team_att") / (pl.col("team_att") + pl.col("team_car"))).alias("team_pass_rate"))
           .with_columns(ewm("team_plays", "team").alias("e_team_plays"),
                         ewm("team_pass_rate", "team").alias("e_team_pass_rate")))
    return ctx


def defense_allowed(ps: pl.DataFrame, ctx: pl.DataFrame) -> pl.DataFrame:
    """Per defense-game: rolling yards allowed to each position group, by game order."""
    allowed = (ps.group_by("game_id", "opponent_team").agg(
        pl.col("passing_yards").sum().alias("pass_yds_alw"),
        pl.col("rushing_yards").filter(pl.col("position") == "RB").sum().alias("rb_rush_alw"),
        pl.col("receiving_yards").filter(pl.col("position") == "WR").sum().alias("wr_rec_alw"),
        pl.col("receiving_yards").filter(pl.col("position") == "TE").sum().alias("te_rec_alw"),
        pl.col("receiving_yards").filter(pl.col("position") == "RB").sum().alias("rb_rec_alw"),
    ).rename({"opponent_team": "def_team"}))
    order = ctx.select("game_id", pl.col("team").alias("def_team"), "gameday")
    cols = ["pass_yds_alw", "rb_rush_alw", "wr_rec_alw", "te_rec_alw", "rb_rec_alw"]
    return (order.join(allowed, on=["game_id", "def_team"], how="left").sort("def_team", "gameday")
            .with_columns([ewm(c, "def_team").alias(f"opp_{c}") for c in cols])
            .select("game_id", "def_team", *[f"opp_{c}" for c in cols]))


def snap_share() -> pl.DataFrame:
    """Offensive snap share per player-game, keyed by gsis player_id."""
    ids = pl.read_parquet(DATA / "raw" / "players.parquet").select("gsis_id", "pfr_id").drop_nulls().unique("pfr_id")
    sc = pl.read_parquet(DATA / "raw" / "snap_counts.parquet").select("game_id", "pfr_player_id", "offense_pct")
    return (sc.join(ids, left_on="pfr_player_id", right_on="pfr_id")
            .select("game_id", pl.col("gsis_id").alias("player_id"), "offense_pct"))


def build(upcoming: pl.DataFrame | None = None, wind: dict | None = None) -> pl.DataFrame:
    sched, ps = load(upcoming, wind)
    ctx = team_context(sched, ps)
    dfn = defense_allowed(ps, ctx)

    # opponent-adjusted defense ratings (pre-game), from adjust.py
    s2 = sched.filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))
    tg = adjust.team_games(DATA / "raw" / "pbp" / "*.parquet", s2)
    adj = adjust.adjusted_ratings(tg, s2).select(
        "game_id", pl.col("team").alias("def_team"),
        pl.col("adj_def_pass").alias("opp_adj_def_pass"), pl.col("adj_def_run").alias("opp_adj_def_run"))

    # teammates' skill snap share lost to injury (more volume for those who play)
    import features
    cov_def, cov_rec = coverage_feats()
    inj_all = features.injury_snaps()
    inj = inj_all.select("season", "week", "team", pl.col("inj_skill").alias("team_inj_skill"),
                         pl.col("inj_wr").alias("team_inj_wr"), pl.col("inj_rb").alias("team_inj_rb"),
                         pl.col("inj_te").alias("team_inj_te"))
    opp_inj = inj_all.select("season", "week", pl.col("team").alias("opponent_team"),
                             pl.col("inj_db").alias("opp_inj_db"), pl.col("inj_front").alias("opp_inj_front"))

    df = (ps.join(snap_share(), on=["game_id", "player_id"], how="left")
          .join(extra_stats(), on=["game_id", "player_id"], how="left")
          .join(ctx.drop("gameday"), on=["game_id", "team"], how="left")
          .with_columns((pl.col("carries") / pl.col("team_car")).alias("carry_share"),
                        # null (not 0/0 = NaN, which would poison the rolling mean) when there was no volume
                        pl.when(pl.col("targets") > 0).then(pl.col("receiving_yards") / pl.col("targets")).alias("yds_per_tgt"),
                        pl.when(pl.col("carries") > 0).then(pl.col("rushing_yards") / pl.col("carries")).alias("yds_per_car"),
                        pl.when(pl.col("attempts") > 0).then(pl.col("passing_yards") / pl.col("attempts")).alias("yds_per_att"))
          .sort("player_id", "gameday"))
    rolling = ["passing_yards", "attempts", "yds_per_att", "rushing_yards", "carries", "carry_share",
               "yds_per_car", "receiving_yards", "receptions", "targets", "target_share",
               "air_yards_share", "yds_per_tgt", "offense_pct", "passing_epa", "fantasy_points_ppr"] + EXTRA
    df = df.with_columns([ewm(c, "player_id").alias(f"e_{c}") for c in rolling]
                         + [pl.col("game_id").cum_count().over("player_id").alias("n_prior")])
    df = (df.join(own_injury(), on=["season", "week", "team", "player_id"], how="left")
          .with_columns(pl.col("own_q", "own_dnp", "own_out").fill_null(0)))
    df = role_defense(df)
    df = qb_usage(df)
    df = (df.join(dfn, left_on=["game_id", "opponent_team"], right_on=["game_id", "def_team"], how="left")
          .join(defense_yoe(ctx), left_on=["game_id", "opponent_team"], right_on=["game_id", "def_team"], how="left")
          .join(adj, left_on=["game_id", "opponent_team"], right_on=["game_id", "def_team"], how="left")
          .join(inj, on=["season", "week", "team"], how="left")
          .join(opp_inj, on=["season", "week", "opponent_team"], how="left")
          .join(cov_def, on=["season", "opponent_team"], how="left")
          .join(cov_rec, on=["season", "player_id"], how="left")
          .with_columns((pl.col("opp_man_rate_prev") * pl.col("plr_ypt_man_prev")
                         + (1 - pl.col("opp_man_rate_prev")) * pl.col("plr_ypt_zone_prev")).alias("plr_cov_matchup_prev"),
                        (pl.col("plr_ypt_man_prev") - pl.col("plr_ypt_zone_prev")).alias("plr_man_edge_prev"))
          .with_columns(pl.col("team_inj_skill", "team_inj_wr", "team_inj_rb", "team_inj_te",
                               "opp_inj_db", "opp_inj_front").fill_null(0)))
    return vacated_volume(df, sched)


ABSORB_CAR, ABSORB_TGT = 0.36, 0.17  # refit below after any change to players_out / VACATE_DAYS
VACATE_DAYS = 16  # the absent player must have played within this many days (his last 2 games)


def vacated_volume(df: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Redistribute injured teammates' usage to the players who are active.

    For each team-game, the carry share (RBs) and target share (WR/TE/RB) of players listed
    Out/Doubtful, taken from their rolling usage entering that week, is split among the active
    players at the position in proportion to their own usage. e_carries_adj / e_targets_adj and
    the matching yardage are the model's volume inputs with the injury already applied, so a
    backup whose starter is out is projected on the starter's volume, not on his own history.
    """
    # each player's usage *including* his latest game, to carry into a week he misses
    usage = (df.sort("player_id", "gameday")
             .with_columns(pl.col("carry_share").ewm_mean(half_life=HALF_LIFE, ignore_nulls=True).over("player_id").alias("cs"),
                           pl.col("target_share").ewm_mean(half_life=HALF_LIFE, ignore_nulls=True).over("player_id").alias("ts"))
             .select("player_id", "position", pl.col("gameday").str.to_date().alias("asof"), "cs", "ts")
             .drop_nulls("asof").sort("asof"))
    games = pl.concat([sched.select("game_id", "season", "week", pl.col(t).alias("team"), "gameday")
                       for t in ("home_team", "away_team")]).with_columns(pl.col("season", "week").cast(pl.Int32))
    # only players who played recently vacate anything: a long absence is already reflected in
    # the teammates' rolling usage, so counting it again would double the adjustment
    out = (players_out()
           .join(games, on=["season", "week", "team"])
           .with_columns((pl.col("gameday").str.to_date() - pl.duration(days=1)).alias("asof"))
           .sort("asof")
           .join_asof(usage, on="asof", by="player_id", strategy="backward", tolerance=f"{VACATE_DAYS}d"))
    vac = out.group_by("game_id", "team").agg(
        pl.col("cs").filter(pl.col("position") == "RB").sum().alias("vac_car"),
        pl.col("ts").filter(pl.col("position").is_in(["WR", "TE", "RB"])).sum().alias("vac_tgt"))
    active = pl.col("own_out") == 0
    healthy = df.group_by("game_id", "team").agg(
        pl.col("e_carry_share").filter(active & (pl.col("position") == "RB")).sum().alias("hc"),
        pl.col("e_target_share").filter(active & pl.col("position").is_in(["WR", "TE", "RB"])).sum().alias("ht"))
    df = (df.join(vac, on=["game_id", "team"], how="left").join(healthy, on=["game_id", "team"], how="left")
          .with_columns(pl.col("vac_car", "vac_tgt").fill_null(0)))
    # only part of the vacated work reaches the known backups (the rest goes to call-ups, or the
    # team runs/throws less at the spot); fitted on 2018-2021: carries 0.35, targets 0.19
    car_mult = 1 + ABSORB_CAR * pl.col("vac_car") / pl.col("hc").clip(0.3)
    tgt_mult = 1 + ABSORB_TGT * pl.col("vac_tgt") / pl.col("ht").clip(0.3)
    return df.with_columns(
        (pl.col("e_carries") * car_mult).alias("e_carries_adj"), (pl.col("e_rushing_yards") * car_mult).alias("e_rushing_yards_adj"),
        (pl.col("e_targets") * tgt_mult).alias("e_targets_adj"), (pl.col("e_receiving_yards") * tgt_mult).alias("e_receiving_yards_adj"),
        (pl.col("e_receptions") * tgt_mult).alias("e_receptions_adj"),
        (pl.col("e_carry_share") * car_mult).alias("e_carry_share_adj"), (pl.col("e_target_share") * tgt_mult).alias("e_target_share_adj"),
        car_mult.alias("car_mult"), tgt_mult.alias("tgt_mult"))


def qb_usage(df: pl.DataFrame) -> pl.DataFrame:
    """A receiver's rolling usage with the quarterback expected to start this game.

    The expected starter is the team's primary passer (most attempts) in its previous game.
    For each receiver, target share / targets / receiving yards are rolled over his prior games
    with that same passer only (e_*_qb), plus how many such games (n_qb). Receivers' roles change
    with the passer (Drake London: 39% target share with Penix, 23% with Cousins), and the plain
    rolling average blurs them together."""
    qb = (df.filter(pl.col("position") == "QB").sort("attempts", descending=True)
          .group_by("game_id", "team").first().select("game_id", "team", pl.col("player_id").alias("game_qb")))
    tg = (df.select("game_id", "team", "gameday").unique().join(qb, on=["game_id", "team"], how="left")
          .sort("team", "gameday")
          .with_columns(pl.col("game_qb").shift(1).over("team").alias("expected_qb")))
    df = df.join(tg.select("game_id", "team", "game_qb", "expected_qb"), on=["game_id", "team"], how="left")
    hist = (df.filter(pl.col("game_qb").is_not_null() & pl.col("position").is_in(["WR", "TE", "RB"]))
            .sort("player_id", "game_qb", "gameday")
            .with_columns(pl.col("target_share").ewm_mean(half_life=HALF_LIFE, ignore_nulls=True).over("player_id", "game_qb").alias("e_target_share_qb"),
                          pl.col("targets").ewm_mean(half_life=HALF_LIFE, ignore_nulls=True).over("player_id", "game_qb").alias("e_targets_qb"),
                          pl.col("receiving_yards").ewm_mean(half_life=HALF_LIFE, ignore_nulls=True).over("player_id", "game_qb").alias("e_receiving_yards_qb"),
                          pl.col("game_id").cum_count().over("player_id", "game_qb").alias("n_qb"))
            .select("player_id", pl.col("game_qb").alias("qb"), pl.col("gameday").str.to_date().alias("asof"),
                    "e_target_share_qb", "e_targets_qb", "e_receiving_yards_qb", "n_qb")
            .sort("asof"))
    keyed = (df.with_columns((pl.col("gameday").str.to_date() - pl.duration(days=1)).alias("asof"))
             .sort("asof")
             .join_asof(hist, on="asof", by_left=["player_id", "expected_qb"], by_right=["player_id", "qb"],
                        strategy="backward", tolerance="400d"))
    return keyed.drop("asof").with_columns(pl.col("n_qb").fill_null(0))


ROLES = {"WR1": ("WR", 1), "WR2": ("WR", 2), "WR3": ("WR", 3), "RB1": ("RB", 1), "TE1": ("TE", 1)}


def role_defense(df: pl.DataFrame) -> pl.DataFrame:
    """Pre-game role (WR1/WR2/WR3/RB1/TE1 by rolling usage) and the opponent's record against it.

    For each defense-game, the yards gained by the opposing WR1 (etc.) and that player's yards
    relative to his own rolling average entering the game. Both are then rolled per defense
    (prior games only) and joined to each player as opp_role_alw / opp_role_ratio for his role.
    """
    # rank among players who are active: a listed-Out starter must not hold the RB1/WR1 slot
    # (in completed games Out players have no row, so this only matters for upcoming games)
    usage = (pl.when(pl.col("own_out") == 1).then(None)
             .when(pl.col("position") == "RB").then(pl.col("e_carry_share"))
             .otherwise(pl.col("e_target_share")))
    df = df.with_columns(
        usage.rank(method="ordinal", descending=True).over("game_id", "team", "position").alias("pos_rank"))
    df = df.with_columns(
        pl.when(usage.is_null()).then(None)
        .when((pl.col("position") == "WR") & (pl.col("pos_rank") <= 3)).then(pl.lit("WR") + pl.col("pos_rank").cast(pl.String))
        .when((pl.col("position") == "RB") & (pl.col("pos_rank") == 1)).then(pl.lit("RB1"))
        .when((pl.col("position") == "TE") & (pl.col("pos_rank") == 1)).then(pl.lit("TE1"))
        .alias("role"))
    yds = pl.when(pl.col("position") == "RB").then(pl.col("rushing_yards")).otherwise(pl.col("receiving_yards"))
    exp = pl.when(pl.col("position") == "RB").then(pl.col("e_rushing_yards")).otherwise(pl.col("e_receiving_yards"))
    per_game = (df.filter(pl.col("role").is_not_null() & (pl.col("n_prior") >= 3))
                .select("game_id", "gameday", pl.col("opponent_team").alias("def_team"), "role",
                        yds.alias("yds"), (yds / exp.clip(5.0)).alias("ratio"))
                .group_by("game_id", "gameday", "def_team", "role").agg(pl.col("yds", "ratio").mean()))
    wide = per_game.pivot(on="role", index=["game_id", "gameday", "def_team"], values=["yds", "ratio"])
    # every defense-game, even those with no qualifying opponent, keeps the rolling window honest
    order = df.select("game_id", "gameday", pl.col("opponent_team").alias("def_team")).unique()
    wide = (order.join(wide, on=["game_id", "gameday", "def_team"], how="left").sort("def_team", "gameday")
            .with_columns([ewm(f"{k}_{r}", "def_team").alias(f"opp_{k}_{r}") for r in ROLES for k in ("yds", "ratio")])
            .select("game_id", "def_team", *[f"opp_{k}_{r}" for r in ROLES for k in ("yds", "ratio")]))
    df = df.join(wide, left_on=["game_id", "opponent_team"], right_on=["game_id", "def_team"], how="left")
    pick = lambda k: pl.coalesce([pl.when(pl.col("role") == r).then(pl.col(f"opp_{k}_{r}")) for r in ROLES])
    return df.with_columns(pick("yds").alias("opp_role_alw"), pick("ratio").alias("opp_role_ratio"),
                           pl.col("pos_rank").clip(1, 4).alias("role_rank"))


NGS_REC = {"avg_separation": "ngs_sep", "avg_cushion": "ngs_cushion", "avg_intended_air_yards": "ngs_iay",
           "avg_yac_above_expectation": "ngs_yac_oe", "catch_percentage": "ngs_catch_pct"}
NGS_RUSH = {"efficiency": "ngs_eff", "percent_attempts_gte_eight_defenders": "ngs_8box",
            "rush_yards_over_expected_per_att": "ngs_ryoe_att", "expected_rush_yards": "ngs_exp_rush"}
NGS_PASS = {"avg_time_to_throw": "ngs_ttt", "aggressiveness": "ngs_agg", "avg_intended_air_yards": "ngs_iay_pass",
            "completion_percentage_above_expectation": "ngs_cpoe", "avg_air_yards_to_sticks": "ngs_ays"}
PFR = {"rec": {"receiving_drop_pct": "pfr_drop_pct", "receiving_broken_tackles": "pfr_rec_bt"},
       "rush": {"rushing_yards_before_contact_avg": "pfr_ybc", "rushing_yards_after_contact_avg": "pfr_yac_c",
                "rushing_broken_tackles": "pfr_rush_bt"},
       "pass": {"times_pressured_pct": "pfr_pressure_pct", "passing_bad_throw_pct": "pfr_bad_throw_pct",
                "times_blitzed": "pfr_blitzed"}}
FF = {"rec_yards_gained_exp": "ff_rec_yds_exp", "receptions_exp": "ff_rec_exp", "rush_yards_gained_exp": "ff_rush_yds_exp",
      "pass_yards_gained_exp": "ff_pass_yds_exp"}
EXTRA = (list(NGS_REC.values()) + list(NGS_RUSH.values()) + list(NGS_PASS.values())
         + [v for d in PFR.values() for v in d.values()] + list(FF.values())
         + ["ff_rec_yoe", "ff_rush_yoe", "ff_pass_yoe"])


def extra_stats() -> pl.DataFrame:
    """Per player-game: Next Gen Stats, PFR advanced stats and expected yards from opportunity.

    NGS lists only players over its weekly minimums (nulls otherwise). Keyed (game_id, player_id);
    NGS has no game_id so it's mapped through (season, week, team)."""
    raw = DATA / "raw"
    sched = (pl.read_parquet(raw / "schedules.parquet")
             .with_columns(pl.col("home_team", "away_team").replace(RENAMES)))
    team_game = pl.concat([sched.select("game_id", "season", "week", pl.col(t).alias("team"))
                           for t in ("home_team", "away_team")])
    out = []
    for name, cols in (("receiving", NGS_REC), ("rushing", NGS_RUSH), ("passing", NGS_PASS)):
        ngs = (pl.read_parquet(raw / f"ngs_{name}.parquet").filter(pl.col("week") >= 1)
               .with_columns(pl.col("team_abbr").replace(RENAMES).alias("team"),
                             pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
               .join(team_game, on=["season", "week", "team"])
               .select("game_id", pl.col("player_gsis_id").alias("player_id"),
                       *[pl.col(c).cast(pl.Float64).alias(v) for c, v in cols.items()]))
        out.append(ngs)
    ids = pl.read_parquet(raw / "players.parquet").select("gsis_id", "pfr_id").drop_nulls().unique("pfr_id")
    for name, cols in PFR.items():
        pfr = (pl.read_parquet(raw / f"pfr_{name}.parquet").join(ids, left_on="pfr_player_id", right_on="pfr_id")
               .select("game_id", pl.col("gsis_id").alias("player_id"),
                       *[pl.col(c).cast(pl.Float64).alias(v) for c, v in cols.items()]))
        out.append(pfr)
    ff = (pl.read_parquet(raw / "ff_opportunity.parquet").filter(pl.col("player_id").is_not_null())
          .select("game_id", "player_id", *[pl.col(c).cast(pl.Float64).alias(v) for c, v in FF.items()],
                  (pl.col("rec_yards_gained") - pl.col("rec_yards_gained_exp")).alias("ff_rec_yoe"),
                  (pl.col("rush_yards_gained") - pl.col("rush_yards_gained_exp")).alias("ff_rush_yoe"),
                  (pl.col("pass_yards_gained") - pl.col("pass_yards_gained_exp")).alias("ff_pass_yoe")))
    out.append(ff)
    df = out[0].unique(["game_id", "player_id"])
    for o in out[1:]:
        df = df.join(o.unique(["game_id", "player_id"]), on=["game_id", "player_id"], how="full", coalesce=True)
    return df


def defense_yoe(ctx: pl.DataFrame) -> pl.DataFrame:
    """Per defense-game: yards over expectation allowed (receiving, rushing, passing), rolled."""
    ff = (pl.read_parquet(DATA / "raw" / "ff_opportunity.parquet")
          .with_columns(pl.col("posteam").replace(RENAMES).alias("team"))
          .group_by("game_id", "team").agg(
              (pl.col("rec_yards_gained") - pl.col("rec_yards_gained_exp")).sum().alias("rec_yoe"),
              (pl.col("rush_yards_gained") - pl.col("rush_yards_gained_exp")).sum().alias("rush_yoe"),
              (pl.col("pass_yards_gained") - pl.col("pass_yards_gained_exp")).sum().alias("pass_yoe")))
    order = ctx.select("game_id", "gameday", pl.col("opp").alias("def_team"), "team")
    cols = ["rec_yoe", "rush_yoe", "pass_yoe"]
    return (order.join(ff, on=["game_id", "team"], how="left").sort("def_team", "gameday")
            .with_columns([ewm(c, "def_team").alias(f"opp_{c}_alw") for c in cols])
            .select("game_id", "def_team", *[f"opp_{c}_alw" for c in cols]))


COV_PRIOR = 25  # targets of shrinkage toward the league average for a receiver's man/zone splits


def coverage_feats() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prior-season coverage tendencies (the per-play coverage feed ends after 2025, so only
    last season's numbers are known before a current-season game).

    Returns (defense, receiver) frames keyed on the season they apply to (= data season + 1):
    defense: man-coverage rate; receiver: yards per target vs man and vs zone, shrunk."""
    part = (pl.read_parquet(DATA / "raw" / "participation.parquet")
            .filter(pl.col("defense_man_zone_type").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"]))
            .select(pl.col("nflverse_game_id").alias("game_id"), "play_id",
                    (pl.col("defense_man_zone_type") == "MAN_COVERAGE").alias("man")))
    pbp = (pl.scan_parquet(DATA / "raw" / "pbp" / "*.parquet")
           .filter(pl.col("pass") == 1)
           .select("game_id", "play_id", "season", "defteam", "receiver_player_id", "yards_gained",
                   "complete_pass", "sack")
           .collect())
    plays = (part.join(pbp, on=["game_id", "play_id"])
             .with_columns(pl.col("defteam").replace(RENAMES)))
    defense = (plays.group_by("season", "defteam").agg(pl.col("man").mean().alias("opp_man_rate_prev"))
               .with_columns((pl.col("season") + 1).alias("season")).rename({"defteam": "opponent_team"}))
    tg = plays.filter(pl.col("receiver_player_id").is_not_null() & (pl.col("sack") == 0))
    lg = tg.group_by("man").agg(pl.col("yards_gained").mean().alias("lg")).sort("man")
    lg_zone, lg_man = lg["lg"].to_list()
    rec = (tg.group_by("season", "receiver_player_id", "man")
           .agg(pl.len().alias("n"), pl.col("yards_gained").sum().alias("yds"))
           .with_columns(pl.when(pl.col("man")).then(pl.lit(lg_man)).otherwise(pl.lit(lg_zone)).alias("lg"))
           .with_columns(((pl.col("yds") + COV_PRIOR * pl.col("lg")) / (pl.col("n") + COV_PRIOR)).alias("ypt"))
           .pivot(on="man", index=["season", "receiver_player_id"], values=["ypt", "n"])
           .rename({"ypt_true": "plr_ypt_man_prev", "ypt_false": "plr_ypt_zone_prev",
                    "n_true": "plr_tgt_man_prev", "n_false": "plr_tgt_zone_prev", "receiver_player_id": "player_id"})
           .with_columns((pl.col("season") + 1).alias("season"),
                         pl.col("plr_ypt_man_prev").fill_null(lg_man), pl.col("plr_ypt_zone_prev").fill_null(lg_zone),
                         pl.col("plr_tgt_man_prev", "plr_tgt_zone_prev").fill_null(0)))
    return defense, rec


NOT_PLAYING = ["RES", "PUP", "NFI", "SUS", "INA", "RSN", "EXE"]  # weekly roster statuses that mean no game


def players_out() -> pl.DataFrame:
    """Every (season, week, team, player_id) not playing: Out/Doubtful on the final injury report,
    or a roster status like reserve/IR, PUP or suspended. Players on IR are not on the weekly
    injury report at all, so the roster file is the only place they show up."""
    inj = (pl.read_parquet(DATA / "raw" / "injuries.parquet")
           .filter(pl.col("report_status").is_in(["Out", "Doubtful"]) & pl.col("gsis_id").is_not_null())
           .select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), pl.col("team").replace(RENAMES),
                   pl.col("gsis_id").alias("player_id")))
    ros = (pl.read_parquet(DATA / "raw" / "rosters_weekly.parquet")
           .filter(pl.col("status").is_in(NOT_PLAYING) & pl.col("gsis_id").is_not_null())
           .select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), pl.col("team").replace(RENAMES),
                   pl.col("gsis_id").alias("player_id")))
    return pl.concat([inj, ros]).unique()


def own_injury() -> pl.DataFrame:
    """The player's own status: Questionable, missed/limited practice, not playing (report or roster)."""
    inj = (pl.read_parquet(DATA / "raw" / "injuries.parquet")
           .filter(pl.col("game_type").is_not_null() & pl.col("gsis_id").is_not_null())
           .with_columns(pl.col("team").replace(RENAMES), pl.col("season").cast(pl.Int32),
                         pl.col("week").cast(pl.Int32), pl.col("gsis_id").alias("player_id")))
    rep = inj.group_by("season", "week", "team", "player_id").agg(
        (pl.col("report_status") == "Questionable").any().cast(pl.Int8).alias("own_q"),
        pl.col("practice_status").str.contains("Did Not|Limited").any().cast(pl.Int8).alias("own_dnp"))
    out = players_out().with_columns(pl.lit(1, dtype=pl.Int8).alias("own_out"))
    return (rep.join(out, on=["season", "week", "team", "player_id"], how="full", coalesce=True)
            .with_columns(pl.col("own_q", "own_dnp", "own_out").fill_null(0)))


COMMON = ["implied_total", "team_spread", "indoor", "wind", "e_team_plays", "e_team_pass_rate",
          "e_offense_pct", "team_inj_skill", "team_inj_wr", "team_inj_rb", "team_inj_te",
          "opp_inj_db", "opp_inj_front", "own_q", "own_dnp", "n_prior", "rest", "short_rest", "off_bye"]
# Volume features are the injury-ADJUSTED ones (vacated_volume): a backup whose starter is out is
# shown to the model as a lead back. Same overall MAE; clearly better when a starter is out.
# Next Gen / PFR / expected-yards features were chosen by ablation (walk-forward MAE); the rest of
# EXTRA is computed but unused: it overfit (rushing MAE rose from 25.31 to 25.49 with all of them).
MARKETS = {
    # name: (target, positions, eligibility filter on pre-game usage, features)
    "pass_yds": ("passing_yards", ["QB"], pl.col("e_attempts") >= 20,
                 ["e_passing_yards", "e_attempts", "e_yds_per_att", "e_passing_epa",
                  "opp_pass_yds_alw", "opp_adj_def_pass",
                  "e_ngs_ttt", "e_ngs_agg", "e_ngs_iay_pass", "e_ngs_cpoe", "e_ngs_ays"]),
    "rush_yds": ("rushing_yards", ["RB"], pl.col("e_carries") >= 6,
                 ["e_rushing_yards_adj", "e_carries_adj", "e_carry_share_adj", "e_yds_per_car",
                  "opp_rb_rush_alw", "opp_adj_def_run", "role_rank", "opp_role_alw", "opp_role_ratio",
                  "e_pfr_rush_bt", "e_ff_rush_yoe"]),
    "rec_yds": ("receiving_yards", ["WR", "TE", "RB"], pl.col("e_targets") >= 3,
                ["e_receiving_yards_adj", "e_receptions_adj", "e_targets_adj", "e_target_share_adj", "e_air_yards_share",
                 "e_yds_per_tgt", "opp_wr_rec_alw", "opp_te_rec_alw", "opp_rb_rec_alw", "opp_adj_def_pass",
                 "is_wr", "is_te", "role_rank", "opp_role_alw", "opp_role_ratio",
                 "e_ngs_iay", "e_pfr_drop_pct", "e_target_share_qb", "e_targets_qb", "n_qb"]),
    "receptions": ("receptions", ["WR", "TE", "RB"], pl.col("e_targets") >= 3,
                   ["e_receptions_adj", "e_targets_adj", "e_target_share_adj", "e_air_yards_share",
                    "opp_wr_rec_alw", "opp_te_rec_alw", "opp_rb_rec_alw", "opp_adj_def_pass",
                    "is_wr", "is_te", "role_rank", "opp_role_alw", "opp_role_ratio",
                    "e_ngs_iay", "e_pfr_drop_pct", "e_target_share_qb", "e_targets_qb", "e_receiving_yards_qb", "n_qb"]),
}


def add_flags(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns((pl.col("position") == "WR").cast(pl.Int8).alias("is_wr"),
                           (pl.col("position") == "TE").cast(pl.Int8).alias("is_te"))


def eligible(df: pl.DataFrame, name: str) -> pl.DataFrame:
    _, positions, elig, _ = MARKETS[name]
    return df.filter(pl.col("position").is_in(positions) & (pl.col("n_prior") >= 3)
                     & elig.fill_null(False) & pl.col("implied_total").is_not_null())


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=50, random_state=0)


def evaluate(df: pl.DataFrame) -> pl.DataFrame:
    df = add_flags(df)
    out = []
    for name, (target, positions, elig, feats) in MARKETS.items():
        m = eligible(df, name)
        cols = feats + COMMON
        print(f"\n== {name} ({target}, {'/'.join(positions)}) ==")
        preds = []
        for s in TEST_SEASONS:
            tr, te = m.filter(pl.col("season") < s), m.filter(pl.col("season") == s)
            model = new_model()
            model.fit(tr.select(cols).to_numpy(), tr[target].to_numpy())
            preds.append(te.with_columns(pl.Series("pred", model.predict(te.select(cols).to_numpy()))))
        te = pl.concat(preds)
        y = te[target].to_numpy()
        base_ewm = te[f"e_{target}"].to_numpy()
        res = {"market": name, "n": len(y),
               "mae_model": np.abs(te["pred"].to_numpy() - y).mean(),
               "mae_recent_avg": np.abs(base_ewm - y).mean()}
        # a fair over/under check without lines: how often does the model's side of the
        # recent average win? (the average stands in for a naive line)
        side = np.sign(te["pred"].to_numpy() - base_ewm)
        hit = np.sign(y - base_ewm) * side
        res["beats_recent_avg_pct"] = (hit > 0).sum() / max((hit != 0).sum(), 1)
        print(f"n={res['n']}  MAE model={res['mae_model']:.2f}  recent avg={res['mae_recent_avg']:.2f}  "
              f"model side vs recent-avg 'line': {res['beats_recent_avg_pct']:.3f}")
        out.append(res)
        te.select("season", "week", "player_display_name", "team", "opponent_team", target, "pred",
                  f"e_{target}").write_parquet(DATA / f"props_{name}_preds.parquet")
    return pl.DataFrame(out)


if __name__ == "__main__":
    df = build()
    print(f"player-games: {df.height}")
    print(evaluate(df))
