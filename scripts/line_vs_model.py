"""How much is the model worth next to the market line?  Re-estimate the blend weights.

Joins every graded line we hold (prop-line history in props_log/propline/graded_<season>.csv and
our own snapshots graded in props_log/grades/) to out-of-sample model projections for the same
player-games (model trained on earlier seasons only), then reports per market:
- MAE of the line, the model, the recent average, and the best blend line + k*(model - line);
- whether the model's disagreement with the line predicts the result (slope, side hit rate);
- whether beating the line last week predicts this week (streaks).
The k values go into props_week.BLEND.

    python scripts/line_vs_model.py            # also writes data/props_<season>_oos.parquet
    python scripts/line_vs_model.py --no-refit  # reuse the saved out-of-sample projections
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import props  # noqa: E402
from props_week import norm_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def oos_projections(season: int) -> pl.DataFrame:
    """Model trained on seasons before `season`, predicting every completed game of `season`."""
    df = props.add_flags(props.build())
    out = []
    for name, (target, positions, elig, feats) in props.MARKETS.items():
        m = props.eligible(df, name)
        tr = m.filter(pl.col("season") < season)
        te = m.filter((pl.col("season") == season) & pl.col(target).is_not_null())
        mdl = props.BlendModel(name).fit(tr)
        out.append(te.select("season", "week", "game_id", "player_id", "player_display_name", "team",
                             pl.col(target).alias("actual"), pl.col(f"e_{target}").alias("recent_avg"))
                   .with_columns(pl.lit(name).alias("mkt"), pl.Series("pred", mdl.predict(te))))
    oos = pl.concat(out)
    oos.write_parquet(DATA / f"props_{season}_oos.parquet")
    return oos


def graded_lines(season: int) -> pl.DataFrame:
    """Consensus line + actual per (player, week, market) from prop-line history and our snapshots."""
    parts = []
    f = ROOT / "props_log" / "propline" / f"graded_{season}.csv"
    if f.exists():
        parts.append(pl.read_csv(f).filter(pl.col("result") != "void")
                     .select("player_id", pl.col("week").cast(pl.Int64), "mkt", "line", "actual", pl.col("player")))
    sched = pl.read_parquet(DATA / "raw" / "schedules.parquet").filter(pl.col("season") == season).select("game_id", "week")
    for g in (ROOT / "props_log" / "grades").glob("*_lines.csv"):
        d = pl.read_csv(g).filter(pl.col("side") == "Over").join(sched, on="game_id")
        parts.append(d.group_by("player", "mkt", "book", "week").agg(pl.col("point").median().alias("pt"), pl.col("actual").first())
                     .group_by("player", "mkt", "week").agg(pl.col("pt").median().alias("line"), pl.col("actual").first())
                     .with_columns(pl.lit(None, dtype=pl.String).alias("player_id"), pl.col("week").cast(pl.Int64)))
    return pl.concat([p.select("player_id", "player", "week", "mkt", "line", "actual") for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--no-refit", action="store_true")
    args = ap.parse_args()
    oos = (pl.read_parquet(DATA / f"props_{args.season}_oos.parquet") if args.no_refit else oos_projections(args.season))
    oos = oos.with_columns(pl.col("week").cast(pl.Int64),
                           pl.col("player_display_name").map_elements(norm_name, return_dtype=pl.String).alias("key"))
    lines = graded_lines(args.season).with_columns(pl.col("player").map_elements(norm_name, return_dtype=pl.String).alias("key"))
    by_id = lines.filter(pl.col("player_id").is_not_null()).join(oos.select("player_id", "week", "mkt", "pred", "recent_avg"),
                                                                 on=["player_id", "week", "mkt"])
    by_name = lines.filter(pl.col("player_id").is_null()).join(oos.select("key", "week", "mkt", "pred", "recent_avg"),
                                                               on=["key", "week", "mkt"])
    cols = ["key", "week", "mkt", "line", "actual", "pred", "recent_avg"]
    d = pl.concat([by_id.select(cols), by_name.select(cols)]).unique(["key", "week", "mkt"])
    print(f"{d.height} player-markets with a line, a model projection and a result "
          f"({d.group_by('mkt').len().sort('mkt').to_dicts()})")

    print("\nMAE vs actual (lower is better) and the best blend line + k*(model - line):")
    ks = {}
    for mkt in props.MARKETS:
        x = d.filter(pl.col("mkt") == mkt)
        if x.height < 20:
            continue
        line, pred, act, avg = (x[c].to_numpy() for c in ("line", "pred", "actual", "recent_avg"))
        err, k = min((np.abs(line + k * (pred - line) - act).mean(), k) for k in np.arange(0, 1.01, 0.1))
        ks[mkt] = k
        print(f"  {mkt:<10} n={x.height:4d}  line {np.abs(line - act).mean():6.2f}  model {np.abs(pred - act).mean():6.2f}  "
              f"recent-avg {np.abs(avg - act).mean():6.2f}  | best k={k:.1f}: {err:6.2f}")

    print("\nDoes the model's disagreement with the line predict the result?")
    for mkt in props.MARKETS:
        x = d.filter(pl.col("mkt") == mkt)
        if x.height < 20:
            continue
        u = (x["pred"] - x["line"]).to_numpy()
        v = (x["actual"] - x["line"]).to_numpy()
        side = np.sign(u)
        dec = np.sign(v) * side != 0
        hit = (np.sign(v) * side > 0).sum()
        big = np.abs(u) > np.median(np.abs(u))
        hitb, nb = (np.sign(v[big]) * side[big] > 0).sum(), (np.sign(v[big]) * side[big] != 0).sum()
        print(f"  {mkt:<10} slope {np.sum(u * v) / np.sum(u * u):+.2f}  corr {np.corrcoef(u, v)[0, 1]:+.2f}  "
              f"model side wins {hit}/{dec.sum()} = {hit / max(dec.sum(), 1):.1%}  "
              f"| larger half of disagreements {hitb}/{nb} = {hitb / max(nb, 1):.1%}")
    u = (d["pred"] - d["line"]).to_numpy()
    v = (d["actual"] - d["line"]).to_numpy()
    side = np.sign(u)
    print(f"  ALL        model side wins {(np.sign(v) * side > 0).sum()}/{(np.sign(v) * side != 0).sum()} = "
          f"{(np.sign(v) * side > 0).sum() / max((np.sign(v) * side != 0).sum(), 1):.1%} (breakeven at -110 is 52.4%); "
          f"model took the under {(side < 0).mean():.0%} of the time")

    s = (lines.with_columns((pl.col("actual") > pl.col("line")).alias("over")).sort("key", "mkt", "week")
         .with_columns(pl.col("over").shift(1).over("key", "mkt").alias("prev"),
                       (pl.col("week") - pl.col("week").shift(1).over("key", "mkt")).alias("gap"))
         .filter(pl.col("gap") == 1))
    print(f"\nStreaks: after an over last week, over rate {s.filter(pl.col('prev'))['over'].mean():.1%} "
          f"(n={s.filter(pl.col('prev')).height}); after an under, {s.filter(~pl.col('prev'))['over'].mean():.1%} "
          f"(n={s.filter(~pl.col('prev')).height})")
    print("\nblend weights to put in props_week.BLEND:", {k: round(v, 1) for k, v in ks.items()})


if __name__ == "__main__":
    main()
