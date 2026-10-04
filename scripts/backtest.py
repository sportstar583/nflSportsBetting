"""Walk-forward backtest: predict home margin, compare to the closing spread.

For each test season, train on all prior seasons, predict result, and bet
against the closing line when |model - market| >= EDGE points.
Standard -110 pricing (win +0.909u, lose -1u, push 0).
"""
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
FEATS = ["spread_line", "rest_diff", "indoor", "wind", "temp", "neutral", "div_game",
         "d_off_epa", "d_off_sr", "d_def_epa", "d_def_sr", "d_pf", "d_pa"]

MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=50)),
    "gbm": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                             learning_rate=0.03, subsample=0.8, random_state=0),
}


def ats(pred, spread, result, edge):
    """Units won betting model vs. market. Home covers if result > spread_line."""
    gap = pred - spread
    mask = np.abs(gap) >= edge
    if edge == 0:
        mask &= gap != 0
    bet_home = gap > 0
    cover = np.sign(result - spread)  # +1 home covers, -1 away covers, 0 push
    win = np.where(bet_home, cover == 1, cover == -1) & mask
    loss = np.where(bet_home, cover == -1, cover == 1) & mask
    n = int(win.sum() + loss.sum())
    units = win.sum() * (100 / 110) - loss.sum()
    return n, int(win.sum()), units


def main():
    df = (pl.read_parquet(DATA / "features.parquet")
          .filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null())
          .drop_nulls(FEATS))
    for name, make in MODELS.items():
        print(f"\n== {name} ==")
        preds, rows = [], []
        for s in TEST_SEASONS:
            tr, te = df.filter(pl.col("season") < s), df.filter(pl.col("season") == s)
            m = make().fit(tr.select(FEATS).to_numpy(), tr["result"].to_numpy())
            preds.append(m.predict(te.select(FEATS).to_numpy()))
            rows.append(te)
        te = pl.concat(rows)
        p, sp, y = np.concatenate(preds), te["spread_line"].to_numpy(), te["result"].to_numpy()
        print(f"games={len(y)}  MAE model={np.abs(p - y).mean():.3f}  market={np.abs(sp - y).mean():.3f}")
        for e in EDGES:
            n, w, u = ats(p, sp, y, e)
            print(f"edge>={e}: bets={n:4d} win%={w / max(n, 1):.3f} units={u:+.1f} roi={u / max(n, 1):+.3f}")


if __name__ == "__main__":
    main()
