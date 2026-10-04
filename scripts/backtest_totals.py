"""Walk-forward backtest on game totals (over/under) vs. the closing total.

Same protocol as backtest.py: train on prior seasons, bet when |model - line| >= EDGE,
-110 pricing. Features are team-pair sums (pace/scoring environment) plus weather.
"""
import itertools
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
BASE = ["total_line", "indoor", "wind", "temp", "neutral", "div_game"]
STATS = ["s_off_epa", "s_off_sr", "s_def_epa", "s_def_sr", "s_pf", "s_pa"]
FEATURE_SETS = {"line+env": BASE, "line+env+stats": BASE + STATS}
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


def main():
    df = (pl.read_parquet(DATA / "features.parquet")
          .filter(pl.col("total").is_not_null() & pl.col("total_line").is_not_null())
          .drop_nulls(BASE + STATS))
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


if __name__ == "__main__":
    main()
