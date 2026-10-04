# nflSportsBetting

Data pipeline for an NFL betting model, sourced from [nflverse](https://github.com/nflverse) via `nflreadpy`.

```
pip install -r requirements.txt
python scripts/pull_data.py   # writes parquet files to data/raw/ (gitignored, ~110MB)
```

Datasets (seasons 2018–2026): schedules (with spread/total/moneyline), play-by-play, team stats, player stats, injuries, rosters.

## Features and baseline backtest
```
python scripts/features.py   # -> data/features.parquet (rolling 8-game EPA/success/points diffs, rest, weather; no look-ahead)
python scripts/backtest.py   # walk-forward 2022-2025, ridge + GBM vs. closing spread
```
Features also include shrunk career QB EPA/dropback for each starter (`d_qb_epa`, `d_qb_exp`) and counts of Out/Doubtful players by position group (`d_inj_*`). `backtest.py` compares feature sets `base`, `base+qb`, `base+qb+inj`.

Result: the models do not beat the closing line (MAE ~9.57 vs 9.54 for the market), and ATS results at -110 are within noise (break-even is 52.4%). Adding QB and injury features did not help (MAE 9.59 / 9.64 vs 9.57 base): the closing spread already prices in the starter and injuries, so the model is feeding on information the market has absorbed. Treat this as a starting point.

## Totals (over/under) backtest
```
python scripts/backtest_totals.py   # predicts total - total_line, walk-forward 2022-2025
```
Also no edge: MAE 10.25-10.28 vs 10.19 for the closing total, and every edge cutoff with a meaningful sample loses money at -110.

## Run/pass splits and pace
Features now include rolling pass/run EPA (offense and defense), plays per game and pass rate, all computed from play-by-play, so no pace API or key is needed. Spread and totals backtests were re-run with them (`base+splits`, `all`, `line+env+splits+pace`): still no edge (MAE 9.61-9.73 vs 9.54 for spreads, 10.27-10.31 vs 10.19 for totals).
