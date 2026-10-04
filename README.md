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
