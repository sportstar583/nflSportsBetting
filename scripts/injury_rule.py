"""Skill-player injury rule: back the side that lost less WR/RB/TE playing time to injury.

Each Out/Doubtful skill player counts by his snap share over his last 4 games (features.py),
so 1.0 is about one full-time starter. Bet against the spread on the less-injured team when
the gap is at least THRESHOLD. Closing spreads at -110, 2018-present.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data"
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5


def main():
    d = (pl.read_parquet(DATA / "features.parquet")
         .filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null()
                 & (pl.col("d_inj_skill").abs() >= THRESHOLD)))
    bet_home = np.where(d["d_inj_skill"].to_numpy() < 0, 1, -1)  # d = home minus away lost
    c = np.sign((d["result"] - d["spread_line"]).to_numpy()) * bet_home
    d = d.with_columns(pl.Series("c", c))
    print(f"Skill-injury rule, gap >= {THRESHOLD} starters' worth of snaps")
    by = d.group_by("season").agg((pl.col("c") > 0).sum().alias("w"), (pl.col("c") < 0).sum().alias("l")).sort("season")
    for s, w, l in by.iter_rows():
        print(f"  {s}: {w}-{l} ({w / max(w + l, 1):.3f})")
    w, l = int((c > 0).sum()), int((c < 0).sum())
    se = 0.5 / np.sqrt(w + l)
    print(f"  all: {w}-{l} ({w / (w + l):.3f} +/- {1.96 * se:.3f}), units at -110={w * 100 / 110 - l:+.1f}, "
          f"breakeven 0.524")
    wk = d.filter(pl.col("week") < 18)
    cw = wk["c"].to_numpy()
    print(f"  excluding week 18+ (starters rested): {(cw > 0).sum()}-{(cw < 0).sum()}")


if __name__ == "__main__":
    main()
