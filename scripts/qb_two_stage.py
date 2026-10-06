"""Does splitting QB passing yards into attempts x yards per attempt beat the one-step model?

Walk-forward over props.TEST_SEASONS (train on earlier seasons, predict the next), QBs eligible
for the pass_yds market. Variants, all scored by MAE on passing yards:
  A  current one-step pass_yds model (props.MARKETS features)
  C  one-step model + the new volume features (defense attempts / YPA allowed, opponent pace)
  B  two-stage: attempts model x yards-per-attempt model (YPA fit weighted by attempts)
  D  average of A and B
Also scores the 2026 games against the graded prop-line passing lines (model trained < 2026):
MAE and how often each model's side of the line won.

New features (pre-game, exponentially weighted like the rest):
  opp_att_alw    pass attempts the defense allows per game
  opp_ypa_alw    passing yards per attempt the defense allows
  opp_plays_alw  offensive plays the defense faces per game
  opp_off_plays  the opponent's own offensive plays per game (pace of the other side)
  opp_off_pass_rate  the opponent's own pass rate

    python scripts/qb_two_stage.py      # uses data/props_features.parquet (from props.py)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import props  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ATT_FEATS = ["e_attempts", "e_team_plays", "e_team_pass_rate", "e_proe_neutral", "implied_total", "team_spread",
             "opp_att_alw", "opp_plays_alw", "opp_off_plays", "opp_off_pass_rate", "e_giveaways", "indoor", "wind",
             "rest", "short_rest", "off_bye", "qb_qtile", "age"]
YPA_FEATS = ["e_yds_per_att", "e_passing_epa", "e_ngs_cpoe", "e_ngs_agg", "e_ngs_iay_pass", "e_ngs_ays", "e_ngs_ttt",
             "opp_ypa_alw", "opp_adj_def_pass", "opp_pass_yds_alw", "qb_qtile", "def_x_qb", "implied_total",
             "team_spread", "indoor", "wind", "age", "jump_pass"]
NEW = ["opp_att_alw", "opp_ypa_alw", "opp_plays_alw", "opp_off_plays", "opp_off_pass_rate"]


def volume_feats() -> pl.DataFrame:
    """Per (game_id, team): what the opponent's defense allows in volume, and the opponent offense's pace."""
    sched = pl.read_parquet(DATA / "raw" / "schedules.parquet").with_columns(
        pl.col("home_team", "away_team").replace(props.RENAMES))
    ps = (pl.read_parquet(DATA / "raw" / "player_stats.parquet")
          .with_columns(pl.col("team", "opponent_team").replace(props.RENAMES)))
    games = pl.concat([sched.select("game_id", "gameday", pl.col("home_team").alias("team"), pl.col("away_team").alias("opp")),
                       sched.select("game_id", "gameday", pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"))])
    off = ps.group_by("game_id", "team").agg(pl.col("attempts").fill_null(0).sum().alias("att"),
                                             pl.col("passing_yards").fill_null(0).sum().alias("pyds"),
                                             pl.col("carries").fill_null(0).sum().alias("car"))
    g = (games.join(off, on=["game_id", "team"], how="left")
         .with_columns((pl.col("att") + pl.col("car")).alias("plays"),
                       (pl.col("att") / (pl.col("att") + pl.col("car"))).alias("pass_rate"),
                       pl.when(pl.col("att") > 0).then(pl.col("pyds") / pl.col("att")).alias("ypa")))
    # offense side, by the team: its own pace and pass rate
    o = (g.sort("team", "gameday")
         .with_columns(props.ewm("plays", "team").alias("off_plays"), props.ewm("pass_rate", "team").alias("off_pass_rate"))
         .select("game_id", pl.col("team").alias("opp"), pl.col("off_plays").alias("opp_off_plays"),
                 pl.col("off_pass_rate").alias("opp_off_pass_rate")))
    # defense side, by the defending team (= opp): what offenses did against it
    d = (g.sort("opp", "gameday")
         .with_columns(props.ewm("att", "opp").alias("opp_att_alw"), props.ewm("ypa", "opp").alias("opp_ypa_alw"),
                       props.ewm("plays", "opp").alias("opp_plays_alw"))
         .select("game_id", "team", "opp", "opp_att_alw", "opp_ypa_alw", "opp_plays_alw"))
    return d.join(o, on=["game_id", "opp"], how="left").drop("opp")


def fit_predict(tr, te, feats, target, weight=None):
    m = props.new_model()
    m.fit(tr.select(feats).to_numpy(), tr[target].to_numpy(),
          sample_weight=None if weight is None else tr[weight].to_numpy())
    return m.predict(te.select(feats).to_numpy())


def predict_all(tr: pl.DataFrame, te: pl.DataFrame) -> pl.DataFrame:
    base = props.MARKETS["pass_yds"][3] + props.COMMON
    tr_ypa = tr.filter(pl.col("attempts") > 0)
    a = fit_predict(tr, te, base, "passing_yards")
    c = fit_predict(tr, te, base + NEW, "passing_yards")
    att = fit_predict(tr, te, ATT_FEATS, "attempts")
    ypa = fit_predict(tr_ypa, te, YPA_FEATS, "yds_per_att", weight="attempts")
    b = att * ypa
    return te.with_columns(pl.Series("A", a), pl.Series("C", c), pl.Series("B", b), pl.Series("D", (a + b) / 2),
                           pl.Series("att_pred", att), pl.Series("ypa_pred", ypa))


def main():
    df = props.add_flags(pl.read_parquet(DATA / "props_features.parquet"))
    df = df.join(volume_feats(), on=["game_id", "team"], how="left")
    m = props.eligible(df, "pass_yds").filter(pl.col("passing_yards").is_not_null())
    m = m.with_columns(pl.when(pl.col("attempts") > 0).then(pl.col("passing_yards") / pl.col("attempts"))
                       .alias("yds_per_att"))

    preds = [predict_all(m.filter(pl.col("season") < s), m.filter(pl.col("season") == s)) for s in props.TEST_SEASONS]
    te = pl.concat(preds)
    y = te["passing_yards"].to_numpy()
    print(f"walk-forward {props.TEST_SEASONS}, {te.height} QB games — MAE on passing yards:")
    for k, lab in [("A", "current one-step"), ("C", "one-step + volume features"), ("B", "attempts x YPA"),
                   ("D", "average of A and B")]:
        print(f"  {k}  {lab:<28} {np.abs(te[k].to_numpy() - y).mean():6.2f}")
    print(f"  attempts model MAE {np.abs(te['att_pred'] - te['attempts']).mean():.2f} "
          f"(recent average {np.abs(te['e_attempts'] - te['attempts']).mean():.2f})")
    yp = te.filter(pl.col("attempts") >= 10)
    print(f"  YPA model MAE {np.abs(yp['ypa_pred'] - yp['yds_per_att']).mean():.3f} "
          f"(recent average {np.abs(yp['e_yds_per_att'] - yp['yds_per_att']).mean():.3f}), games with 10+ attempts")
    print("  by season:", {s: {k: round(float(np.abs(x[k] - x['passing_yards']).mean()), 2) for k in "ACBD"}
                           for (s,), x in sorted(te.group_by("season"))})

    # 2026 against the real lines
    cur = predict_all(m.filter(pl.col("season") < 2026), m.filter(pl.col("season") == 2026))
    lines = (pl.read_csv(ROOT / "props_log" / "propline" / "graded_2026.csv")
             .filter((pl.col("mkt") == "pass_yds") & ~pl.col("result").is_in(["void"]))
             .select("player_id", pl.col("week").cast(pl.Int32), "line"))
    j = cur.with_columns(pl.col("week").cast(pl.Int32)).join(lines, on=["player_id", "week"])
    y, ln = j["passing_yards"].to_numpy(), j["line"].to_numpy()
    print(f"\n2026 weeks 1-4 vs prop-line passing lines, n={j.height}:  line MAE {np.abs(ln - y).mean():.2f}")
    for k in "ACBD":
        p = j[k].to_numpy()
        side, res = np.sign(p - ln), np.sign(y - ln)
        dec = (side * res != 0)
        big = np.abs(p - ln) >= 15
        print(f"  {k}  MAE {np.abs(p - y).mean():6.2f}  side wins {(side * res > 0).sum()}/{dec.sum()} "
              f"= {(side * res > 0).sum() / max(dec.sum(), 1):.1%}  | 15+ yd disagreements "
              f"{(side * res > 0)[big].sum()}/{dec[big].sum()}  | took under {(side < 0).mean():.0%}")
    j.select("season", "week", "player_display_name", "team", "opponent_team", "line", "passing_yards", "attempts",
             "att_pred", "ypa_pred", "A", "B", "C").write_csv(ROOT / "props_log" / "qb_two_stage_2026.csv")
    print("wrote props_log/qb_two_stage_2026.csv")


if __name__ == "__main__":
    main()
