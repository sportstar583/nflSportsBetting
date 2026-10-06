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

## Wind rule on forecast wind (what a bettor knew)
`python scripts/weather_forecast.py` pulls Open-Meteo's archived short-range forecast for every
outdoor game (1,597 games, 2018-2026); `backtest_totals.py` then tests the under rule on it.
Forecast and measured wind correlate 0.74; the forecast reads ~1 mph lower on average.

| Under when... | Record | Win % | Units (-110) | Seasons > breakeven |
| --- | --- | --- | --- | --- |
| Measured wind >= 10 (hindsight) | 276-193 | 58.8% | +57.9 | 7/8 full |
| Forecast >= 10 mph | 186-143 | 56.5% | +26.1 | 5/8 full |
| Forecast >= 8.1 mph (same share of games as measured >= 10) | 320-230 | 58.2% | +60.9 | 6/8 full |
| Forecast >= 12 mph | 100-81 | 55.2% | +9.9 | 5/8 full |

Part of the hindsight result was surprise wind (measured >= 10 but forecast calm: 63.3%), which
nobody could bet. But games that were forecast windy still went under 56-58% at every threshold,
against the closing total. Losing seasons: 2020 and 2024. This is the strongest result in the repo,
but it is a well-known angle, three thresholds were looked at, and the forecast used is the one
close to kickoff, so bet it near game time. Track it on 2026 before trusting it.

### Discounting games a player wasn't healthy for, or caught from a different QB
For WR/TE/RB, the rolling averages in `props.py` give half weight to (a) games where the player was
on the final injury report as Questionable/Doubtful without full practice, or was in his first
game back from Out, and (b) games with a different starting QB than today's. The discount only
applies when the player is healthy today. If he is still hurt, his hurt games are the best guide.
Example: Drake London's 2025 weeks 16-18 (knee, Cousins at QB) dragged his rolling receiving
yards to 63.8 going into 2026 week 3, Penix's first start. With the discount it is 75.4.

Same player-games, 2022-2025 (`python scripts/props.py HURT_WEIGHT OTHER_QB_WEIGHT` to try others):

| Weights (hurt, other QB) | Rec yds MAE | Rush yds MAE | Receptions MAE |
| --- | --- | --- | --- |
| 1, 1 (old) | 22.93 | 25.59 | 1.670 |
| 0.5, 0.5 (new default) | 22.84 | 25.45 | 1.671 |
| 0, 1 (drop hurt games) | 23.11 | 26.34 | 1.690 |

Small gain. Dropping hurt games entirely, or also discounting low-snap "left early" games, made
projections worse: a player who was limited often stays limited. The health discount also made
QB passing yards worse, so QBs are exempt.
