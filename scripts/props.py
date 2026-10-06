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
# Weight of a past game in a WR/TE/RB's rolling averages when he was not 100% (on the injury
# report as Questionable/Doubtful without full practice, or first game back from Out) or caught
# passes from a different QB than today's starter. 1.0 = count normally, 0 = drop it.
# Same-rows backtest 2022-2025: rec yds MAE 22.93 -> 22.84, rush yds 25.59 -> 25.45, receptions
# flat. Weights below 0.5, or adding "left early" low-snap games, made it worse; so did
# applying the health weight to QBs (pass yds 64.2 -> 64.5), so QBs are exempt.
HURT_WEIGHT = 0.5
OTHER_QB_WEIGHT = 0.5
TEST_SEASONS = [2022, 2023, 2024, 2025]
SKILL = ["QB", "RB", "WR", "TE"]


def ewm(col: str, by: str) -> pl.Expr:
    """Exponentially weighted mean of prior games only (shifted), per `by`."""
    return pl.col(col).shift(1).ewm_mean(half_life=HALF_LIFE, ignore_nulls=True).over(by)


def wewm(col: str, by: str, w: str = "w") -> pl.Expr:
    """Like ewm(), but each prior game also counts by its weight `w` (0-1).

    Ratio of two decayed sums, so a down-weighted game still ages the games before it.
    """
    x = pl.col(col).cast(pl.Float64)
    wt = pl.when(x.is_null() | x.is_nan()).then(0.0).otherwise(pl.col(w))
    num = (x.fill_nan(None).fill_null(0) * wt).shift(1).ewm_mean(half_life=HALF_LIFE, adjust=True).over(by)
    den = wt.shift(1).ewm_mean(half_life=HALF_LIFE, adjust=True).over(by)
    return pl.when(den > 0).then(num / den)


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
             .with_columns(pl.col("home_team", "away_team").replace(RENAMES)))
    ps = (pl.read_parquet(DATA / "raw" / "player_stats.parquet")
          .filter(pl.col("position").is_in(SKILL))
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
            "game_id", "gameday", pl.col(t).alias("team"), pl.col(o).alias("opp"),
            (pl.col("total_line") / 2 + sgn * pl.col("spread_line") / 2).alias("implied_total"),
            (sgn * pl.col("spread_line")).alias("team_spread"),  # + means team favored
            pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8).alias("indoor"),
            pl.when(pl.col("roof").is_in(["dome", "closed"])).then(0.0)
            .otherwise(pl.col("wind").cast(pl.Float64)).alias("wind"),
        ))
    ctx = pl.concat(side)
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


def health_flags() -> pl.DataFrame:
    """Per player-week: played not at 100% per the final injury report.

    `hurt_report`: Questionable/Doubtful without full practice that week.
    `back_from_out`: listed Out the previous week (first game back).
    """
    inj = (pl.read_parquet(DATA / "raw" / "injuries.parquet")
           .filter(pl.col("game_type").is_not_null())
           .select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
                   pl.col("gsis_id").alias("player_id"), "report_status", "practice_status")
           .unique(["season", "week", "player_id"]))
    hurt = (inj.filter(pl.col("report_status").is_in(["Questionable", "Doubtful"])
                       & ~pl.col("practice_status").fill_null("").str.starts_with("Full"))
            .select("season", "week", "player_id", pl.lit(True).alias("hurt_report")))
    back = (inj.filter(pl.col("report_status") == "Out")
            .select("season", (pl.col("week") + 1).alias("week"), "player_id", pl.lit(True).alias("back_from_out")))
    return hurt.join(back, on=["season", "week", "player_id"], how="full", coalesce=True)


def starting_qb(ps: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: the QB with the most pass attempts."""
    return (ps.filter(pl.col("position") == "QB").group_by("game_id", "team")
            .agg(pl.col("player_id").sort_by("attempts", descending=True).first().alias("qb_id")))


def build(hurt_weight: float = HURT_WEIGHT, other_qb_weight: float = OTHER_QB_WEIGHT) -> pl.DataFrame:
    sched, ps = load()
    ctx = team_context(sched, ps)
    dfn = defense_allowed(ps, ctx)

    # opponent-adjusted defense ratings (pre-game), from adjust.py
    s2 = sched.filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))
    tg = adjust.team_games(DATA / "raw" / "pbp.parquet", s2)
    adj = adjust.adjusted_ratings(tg, s2).select(
        "game_id", pl.col("team").alias("def_team"),
        pl.col("adj_def_pass").alias("opp_adj_def_pass"), pl.col("adj_def_run").alias("opp_adj_def_run"))

    # teammates' skill snap share lost to injury (more volume for those who play)
    import features
    inj = features.injury_snaps().select("season", "week", "team", pl.col("inj_skill").alias("team_inj_skill"))

    df = (ps.join(snap_share(), on=["game_id", "player_id"], how="left")
          .join(health_flags(), on=["season", "week", "player_id"], how="left")
          .join(starting_qb(ps), on=["game_id", "team"], how="left")
          .join(ctx.drop("gameday"), on=["game_id", "team"], how="left")
          .with_columns((pl.col("carries") / pl.col("team_car")).alias("carry_share"),
                        (pl.col("receiving_yards") / pl.col("targets")).alias("yds_per_tgt"),
                        (pl.col("rushing_yards") / pl.col("carries")).alias("yds_per_car"),
                        (pl.col("passing_yards") / pl.col("attempts")).alias("yds_per_att"))
          .sort("player_id", "gameday"))
    # Down-weight games a skill player was not 100% for (on the report as hurt, or first game
    # back). Also games with a different QB than today's starter (today's starter counts as
    # known before kickoff).
    hurt = (pl.col("hurt_report").fill_null(False) | pl.col("back_from_out").fill_null(False)) & (pl.col("position") != "QB")
    df = df.with_columns(pl.when(hurt).then(hurt_weight).otherwise(1.0).alias("w_health"))
    rolling = ["passing_yards", "attempts", "yds_per_att", "rushing_yards", "carries", "carry_share",
               "yds_per_car", "receiving_yards", "receptions", "targets", "target_share",
               "air_yards_share", "yds_per_tgt", "offense_pct", "passing_epa", "fantasy_points_ppr"]
    # Only discount the hurt games when the player is healthy today (not on this week's report as
    # hurt or just back): a still-hurt player's hurt games are the best guide to how he plays now.
    healthy_now = ~(pl.col("hurt_report").fill_null(False) | pl.col("back_from_out").fill_null(False))
    df = df.with_columns(pl.col("game_id").cum_count().over("player_id").alias("n_prior"),
                         healthy_now.alias("healthy_now"))
    if hurt_weight != 1.0:
        df = df.with_columns(pl.lit(1.0).alias("w_one"))
        df = df.with_columns([wewm(c, "player_id", "w_one").alias(f"raw_{c}") for c in rolling]).drop("w_one")
    if other_qb_weight == 1.0:
        df = df.with_columns([wewm(c, "player_id", "w_health").alias(f"e_{c}") for c in rolling])
    else:
        # each game needs its own weights (relative to that game's QB), so compute the
        # weighted average per distinct (player, today's QB) and keep rows that match
        df = df.with_columns([pl.lit(None, pl.Float64).alias(f"e_{c}") for c in rolling])
        parts = []
        for (pos,), g in df.partition_by("position", as_dict=True).items():
            if pos == "QB":
                parts.append(g.with_columns([wewm(c, "player_id", "w_health").alias(f"e_{c}") for c in rolling]))
                continue
            pairs = g.select("player_id", pl.col("qb_id").alias("cur_qb")).unique()
            x = (g.join(pairs, on="player_id")
                 .with_columns((pl.col("w_health") * pl.when(pl.col("qb_id") == pl.col("cur_qb")).then(1.0)
                                .otherwise(other_qb_weight)).alias("w_qb"))
                 .sort("player_id", "cur_qb", "gameday")
                 .with_columns([wewm(c, ["player_id", "cur_qb"], "w_qb").alias(f"e_{c}") for c in rolling])
                 .filter(pl.col("qb_id").eq_missing(pl.col("cur_qb"))).drop("cur_qb", "w_qb"))
            parts.append(x)
        df = pl.concat(parts, how="vertical_relaxed").sort("player_id", "gameday")
    if hurt_weight != 1.0:
        df = df.with_columns([pl.when(pl.col("healthy_now")).then(pl.col(f"e_{c}")).otherwise(pl.col(f"raw_{c}"))
                              .alias(f"e_{c}") for c in rolling]).drop([f"raw_{c}" for c in rolling])
    df = (df.join(dfn, left_on=["game_id", "opponent_team"], right_on=["game_id", "def_team"], how="left")
          .join(adj, left_on=["game_id", "opponent_team"], right_on=["game_id", "def_team"], how="left")
          .join(inj, on=["season", "week", "team"], how="left")
          .with_columns(pl.col("team_inj_skill").fill_null(0)))
    return df


COMMON = ["implied_total", "team_spread", "indoor", "wind", "e_team_plays", "e_team_pass_rate",
          "e_offense_pct", "team_inj_skill", "n_prior"]
MARKETS = {
    # name: (target, positions, eligibility filter on pre-game usage, features)
    "pass_yds": ("passing_yards", ["QB"], pl.col("e_attempts") >= 20,
                 ["e_passing_yards", "e_attempts", "e_yds_per_att", "e_passing_epa",
                  "opp_pass_yds_alw", "opp_adj_def_pass"]),
    "rush_yds": ("rushing_yards", ["RB"], pl.col("e_carries") >= 6,
                 ["e_rushing_yards", "e_carries", "e_carry_share", "e_yds_per_car",
                  "opp_rb_rush_alw", "opp_adj_def_run"]),
    "rec_yds": ("receiving_yards", ["WR", "TE", "RB"], pl.col("e_targets") >= 3,
                ["e_receiving_yards", "e_receptions", "e_targets", "e_target_share", "e_air_yards_share",
                 "e_yds_per_tgt", "opp_wr_rec_alw", "opp_te_rec_alw", "opp_rb_rec_alw", "opp_adj_def_pass",
                 "is_wr", "is_te"]),
    "receptions": ("receptions", ["WR", "TE", "RB"], pl.col("e_targets") >= 3,
                   ["e_receptions", "e_targets", "e_target_share", "e_air_yards_share",
                    "opp_wr_rec_alw", "opp_te_rec_alw", "opp_rb_rec_alw", "opp_adj_def_pass",
                    "is_wr", "is_te"]),
}


def evaluate(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns((pl.col("position") == "WR").cast(pl.Int8).alias("is_wr"),
                         (pl.col("position") == "TE").cast(pl.Int8).alias("is_te"))
    out = []
    for name, (target, positions, elig, feats) in MARKETS.items():
        m = df.filter(pl.col("position").is_in(positions) & (pl.col("n_prior") >= 3)
                      & elig.fill_null(False) & pl.col("implied_total").is_not_null())
        cols = feats + COMMON
        print(f"\n== {name} ({target}, {'/'.join(positions)}) ==")
        preds = []
        for s in TEST_SEASONS:
            tr, te = m.filter(pl.col("season") < s), m.filter(pl.col("season") == s)
            model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                                  max_leaf_nodes=15, min_samples_leaf=50, random_state=0)
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
    import sys
    args = [float(a) for a in sys.argv[1:]]
    df = build(*args)
    print(f"player-games: {df.height}")
    print(evaluate(df))
