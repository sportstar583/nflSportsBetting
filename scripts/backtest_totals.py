"""Walk-forward backtest on game totals (over/under) vs. the closing total.

Same protocol as backtest.py: train on prior seasons, bet when |model - line| >= EDGE,
-110 pricing. Features are team-pair sums (pace/scoring environment) plus weather.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"
TEST_SEASONS = [2022, 2023, 2024, 2025]
EDGES = [0, 1, 2, 3]
BASE = ["total_line", "indoor", "wind", "wind_over_10", "weather_missing", "temp", "neutral", "div_game"]
INJ = ["s_inj_ol", "s_inj_skill", "s_inj_def"]  # snap share lost, both teams
STATS = ["s_off_epa", "s_off_sr", "s_def_epa", "s_def_sr", "s_pf", "s_pa"]
SPLITS = ["s_off_pass_epa", "s_off_run_epa", "s_def_pass_epa", "s_def_run_epa",
          "s_off_plays", "s_off_pass_rate"]
ADJ = ["s_adj_all", "s_adj_pass", "s_adj_run", "s_sec_per_play"]
FEATURE_SETS = {"line+env": BASE, "line+env+stats": BASE + STATS,
                "line+env+splits+pace": BASE + SPLITS,
                "line+env+adj+tempo": BASE + ADJ,
                "line+env+adj+tempo+inj": BASE + ADJ + INJ}
# residual target: predict (total - total_line), so the line is an offset, not a feature to fit
MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=50)),
    "gbm": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                             learning_rate=0.03, subsample=0.8, random_state=0),
}


def ou(pred, line, total, edge):
    gap = pred - line
    mask = (np.abs(gap) >= edge) & (gap != 0)
    over = gap > 0
    cover = np.sign(total - line)  # +1 over, -1 under, 0 push
    win = np.where(over, cover == 1, cover == -1) & mask
    loss = np.where(over, cover == -1, cover == 1) & mask
    n = int(win.sum() + loss.sum())
    return n, int(win.sum()), win.sum() * (100 / 110) - loss.sum()


CARD_N = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # picks each way per week


def card(te: pl.DataFrame, pred: np.ndarray, n: int = CARD_N) -> pl.DataFrame:
    """Each week: the n biggest over edges and n biggest under edges (as in the college repo's card)."""
    d = te.select("season", "week", "total", "total_line").with_columns(edge=pl.Series(pred) - pl.col("total_line"))
    overs = d.filter(pl.col("edge") > 0).sort("edge", descending=True).group_by("season", "week").head(n)
    unders = d.filter(pl.col("edge") < 0).sort("edge").group_by("season", "week").head(n)
    picks = pl.concat([overs, unders])
    side = pl.when(pl.col("edge") > 0).then(1).otherwise(-1)
    res = side * (pl.col("total") - pl.col("total_line")).sign()
    return picks.with_columns(res.alias("res")).group_by("season").agg(
        (pl.col("res") == 1).sum().alias("w"), (pl.col("res") == -1).sum().alias("l")).sort("season")


def _rule_line(label: str, d: pl.DataFrame, wind_col: str, knee: float = 10) -> None:
    r = (d.filter(pl.col(wind_col) >= knee)
         .with_columns((pl.col("total") - pl.col("total_line")).sign().alias("c"))
         .group_by("season").agg((pl.col("c") == -1).sum().alias("w"), (pl.col("c") == 1).sum().alias("l"))
         .sort("season"))
    rows = "  ".join(f"{s}: {a}-{b}" for s, a, b in r.iter_rows())
    W, L = r["w"].sum(), r["l"].sum()
    print(f"{label}: {rows}  total {W}-{L} ({W / max(W + L, 1):.3f}), units={W * 100 / 110 - L:+.1f}")


def wind_rule(df: pl.DataFrame) -> None:
    """Fixed rule, no model: bet the under in outdoor games with wind >= 10 mph.

    nflverse wind is measured at kickoff (hindsight); the forecast version uses only what
    was knowable before the game (scripts/weather_forecast.py).
    """
    out = df.filter((pl.col("indoor") == 0) & (pl.col("weather_missing") == 0))
    print()
    _rule_line("wind rule, observed wind>=10", out, "wind")
    fc_path = DATA / "raw" / "forecast_weather.parquet"
    if fc_path.exists():
        fc = df.filter(pl.col("indoor") == 0).join(pl.read_parquet(fc_path), on="game_id").drop_nulls("fc_wind")
        _rule_line("wind rule, FORECAST wind>=10", fc, "fc_wind")
        both = fc.filter(pl.col("weather_missing") == 0)
        _rule_line("  observed>=10, same games", both, "wind")
        _rule_line("  forecast>=10, same games", both, "fc_wind")


def main():
    df = (pl.read_parquet(DATA / "features.parquet")
          .filter(pl.col("total").is_not_null() & pl.col("total_line").is_not_null())
          .drop_nulls(BASE + STATS + SPLITS + ADJ + INJ))
    wind_rule(df)
    for (name, make), (fs_name, feats) in itertools.product(MODELS.items(), FEATURE_SETS.items()):
        print(f"\n== {name} / {fs_name} ==")
        preds, rows = [], []
        for s in TEST_SEASONS:
            tr, te = df.filter(pl.col("season") < s), df.filter(pl.col("season") == s)
            m = make().fit(tr.select(feats).to_numpy(),
                           (tr["total"] - tr["total_line"]).to_numpy())
            preds.append(te["total_line"].to_numpy() + m.predict(te.select(feats).to_numpy()))
            rows.append(te)
        te = pl.concat(rows)
        p, ln, y = np.concatenate(preds), te["total_line"].to_numpy(), te["total"].to_numpy()
        print(f"games={len(y)}  MAE model={np.abs(p - y).mean():.3f}  market={np.abs(ln - y).mean():.3f}")
        for e in EDGES:
            n, w, u = ou(p, ln, y, e)
            print(f"edge>={e}: bets={n:4d} win%={w / max(n, 1):.3f} units={u:+.1f} roi={u / max(n, 1):+.3f}")
        c = card(te, p)
        by_season = "  ".join(f"{s}: {w}-{l}" for s, w, l in c.iter_rows())
        w, l = c["w"].sum(), c["l"].sum()
        print(f"card top {CARD_N} each way: {by_season}  total {w}-{l} ({w / (w + l):.3f}), "
              f"units={w * 100 / 110 - l:+.1f}")


if __name__ == "__main__":
    main()
