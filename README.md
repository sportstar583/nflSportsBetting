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

## Opponent-adjusted EPA and tempo (ported from the college repo)
`scripts/adjust.py` fits, walk-forward for every week, a play-weighted ridge regression
(EPA/play = mean + offense + opponent defense + home field) over the prior two years with a
180-day half-life, for all plays, passes and runs; plus decayed seconds per play from drive clock.
Features: `d_adj_*` (net rating gap, spreads), `s_adj_*` (combined expected EPA, totals),
`d_/s_sec_per_play`. Backtest sets `adj`, `adj+qb+inj` (spreads) and `line+env+adj+tempo` (totals).

The adjusted ratings are better ratings (correlation with the closing spread 0.86 vs 0.78 for
raw rolling EPA) but give no betting edge: spread MAE 9.65-9.79 vs 9.54 market, and they
lost at every edge cutoff; totals MAE 10.25-10.29 vs 10.19. Better ratings mostly agree
more with the market.

Also fixed: schedules and injuries use `OAK` for 2018-19 while pbp uses `LV`, which dropped all
32 of those Raiders games from training. Team codes are now normalized.

## Weekly card (top 3 overs + top 3 unders), totals
Same strategy as the college repo's card, printed by `backtest_totals.py` for each model/feature set.
2022-2025: 51.0%-53.0% across the 8 variants (best: GBM line+env, 236-209, +5.5u), no variant
above breakeven in every season. No edge; note an NFL week has ~16 games, so 6 picks is over a
third of the slate, far less selective than the college card.

`python scripts/backtest_totals.py N` sets picks per side (default 3). Top 2 each way: 50.5%-54.9%
(best GBM line+env 168-138, but 50.6% in 2024). Top 1 each way: 49.4%-56.5% on ~160 bets each
(ridge adj+tempo 91-72 is the only variant above breakeven every season). With 24 card variants
tried, the best results are what chance alone would produce; none is significant.

## Recency bias
`python scripts/recency.py`: does a bad week against the spread predict the next one (or a
bounce-back)? 2018-present, closing spreads, same-season consecutive games.
No: last week's ATS margin has ~zero correlation with this week's (-0.017), every bucket of
last week's result covers 49-52%, teams after two straight non-covers cover 50.2%, and backing
teams that missed by 14+ went 51.3% (-9u at -110). The closing line already adjusts for last week.

## Skill-player injury rule
`python scripts/injury_rule.py [gap]`: back the team that lost less WR/RB/TE snap share to
Out/Doubtful players (default gap 0.5 starters). 2018-present: 428-372 (53.5%, +17u at -110),
about 2 standard errors above a coin flip but only 0.6 above breakeven, and seasons are
inconsistent. A weak lean to track, not an edge.

## Pace and totals
Faster-paced matchups do score more, but the closing total already prices it: pace correlates
+0.24 with the total line and about 0 with (total - line); over rates by pace quintile are 46-51%.

## Player props (projections; no prop lines yet)
`python scripts/props.py`: walk-forward 2022-2025 projections of the median for QB passing
yards, RB rushing yards, and WR/TE/RB receiving yards and receptions, from opportunity (snap share,
target/carry share), efficiency, game environment (implied team total and spread from the closing
lines, pace, wind), opponent (adjusted defense, yards allowed to the position) and teammate injuries.
Per-player predictions are written to `data/props_<market>_preds.parquet`.

| Market | Player-games | MAE model | MAE recent avg | Model side vs skew-corrected naive line |
| --- | --- | --- | --- | --- |
| Passing yards | 2,212 | 64.1 | 66.2 | 59.7% |
| Rushing yards | 3,360 | 25.6 | 26.4 | 55.4% |
| Receiving yards | 9,863 | 22.9 | 23.9 | 55.2% |
| Receptions | 9,863 | 1.67 | 1.72 | 54.1% |

Yardage is right-skewed, so results land under a player's recent average 52-60% of the time; a
median model "beats" a raw average just by leaning under. The last column removes that by shifting
the naive line by the typical gap (fit on earlier seasons). The model clearly adds information over
recent averages, but sportsbook lines already use game script and matchups, so this says nothing
yet about beating real prop lines. That needs historical prop lines.

## Prop line log (free, The Odds API)
`python scripts/prop_lines.py --dry-run` shows upcoming games and the credit cost;
`python scripts/prop_lines.py` snapshots pass/rush/receiving yards and receptions props for games
in the next 7 days into `props_log/<timestamp>.csv` (committed, so the history survives).
Needs `ODDS_API_KEY` in the environment or a gitignored `.env`. Free tier: 500 credits/month;
each game costs 4 credits (4 markets x 1 region). Run near open and near kickoff each week.
