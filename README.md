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

## This week's projections vs. the logged lines
`python scripts/props_week.py [props_log/<snapshot>.csv] [--top N]` (run `python scripts/props.py` once
first; it saves the walk-forward predictions used for calibration). For every unplayed game in the
snapshot it builds stat-less rows for each team's skill players, computes the same pre-game
features as the backtest (wind from Open-Meteo's live forecast), refits each market on all completed
games and predicts the median. P(over) for each book's line comes from out-of-sample residuals of
similar predictions; each price gets an EV, and the best per player-market goes to
`data/props_edges_<snapshot>.csv` (every book and side in `_all.csv`).

Read the output with care:
- Rows marked `news` have a consensus line more than 35% away from the player's recent average.
  That is nearly always injury, QB or role news the books have priced in and the model can't see,
  so their large "edges" mean the model is out of date. The script also warns when the books list a
  different passing QB than the team's last game (e.g. a backup starting), because receiving
  projections don't account for it.
- QB rushing, and low-usage backs and receivers, have no projection (same eligibility rules as the backtest).
- On the 2026-10-04 snapshot (week 4) the model sides with the under 54-58% of the time in every
  market, against the books' 50% after removing the vig. Either the books shade toward overs or the
  model runs low against real lines; only grading these snapshots against results will tell which.
  Until then the EVs rank disagreements; they are not proven edges.

## Grading the lines and the model
`props_week.py` also writes `props_log/projections/<snapshot>.csv` (committed), so what the model said
before kickoff is kept. After the games, run `python scripts/pull_data.py` then
`python scripts/grade_props.py [snapshot prefix]`. For each snapshot it reports:
- **Lines alone:** over hit rate per market and the return from betting every over or every under
  at the listed price. This tests whether the books shade toward overs.
- **The model:** Brier score of `p_model` vs. the books' no-vig probability, and the record and units
  of the best bet per player-market above several EV cutoffs, split by news flag and by side.

A player who didn't play is a void; an active player with no stats grades as 0. Per-line results go
to `props_log/grades/`. One week is a few hundred correlated bets, so read a single week as noise;
the question is whether the under lean and the positive-EV picks hold up over many weeks.

## Injuries, roles and matchups in the prop model
Added to `props.py` (all pre-game, all tested walk-forward 2022-2025):
- **Player's own injury report** (Questionable, limited/no practice): MAE within 0.03 of before.
  Out/Doubtful players are dropped from the weekly card.
- **Teammate injuries by position** (starters' snap share out at WR, RB, TE) and **opponent's
  defensive injuries** (secondary, front seven). Raw effect, 2019-present: WRs produce 14% above
  their own average when 0.6-1.2 of a starting WR is out and 32% above when more is; RBs 42% above
  when a starting RB is out; TEs +5%. The opponent's missing DBs add only ~4% for receivers at the
  extreme, and missing front-seven starters do nothing measurable for rushers. The books know all
  this too. In the model: passing MAE 64.07 -> 63.72, rushing 25.33 -> 25.31, receiving 22.82 -> 22.81.
- **Role-based defense** (opponent's rolling record against WR1/WR2/WR3/RB1/TE1, raw and relative
  to each player's own average): a weak signal (correlation +0.03 for WR1s; the toughest fifth of
  defenses hold WR1s to 98% of their average, the softest 103%; RB1s +0.08 to +0.11) but a
  consistent gain: rushing 25.50 -> 25.33, receiving 22.90 -> 22.82, receptions 1.670 -> 1.664.
No slot/outside or coverage data (nflverse has none), so WR1 is "top target share", not alignment.

## Richer stats: Next Gen Stats, PFR advanced stats, expected yards
`pull_data.py` now also pulls nflverse's weekly Next Gen Stats (receiving/rushing/passing), PFR
advanced stats and `ff_opportunity` (expected yards from opportunity), all current through the
latest week. `props.py` rolls every one of them per player, but only the subsets that survived a
walk-forward ablation are model features (the rest overfit: rushing MAE rose 25.31 -> 25.49 with all
ten of its candidates):
- **Passing:** NGS time to throw, aggressiveness, intended air yards, CPOE, air yards to sticks.
  MAE 63.72 -> 63.31 (the biggest single gain so far). PFR pressure/bad-throw rates, expected
  passing yards and the defense's passing yards over expectation allowed all made it worse.
- **Rushing:** PFR broken tackles and rush yards over expectation: 25.31 -> 25.24.
- **Receiving yards / receptions:** NGS intended air yards and PFR drop rate: 22.81 -> 22.77 and
  1.665 -> 1.662. Separation, cushion, YAC over expectation and expected yards did not help.

NFL Savant (nflsavant.com) has no public API; the rebuilt site has an internal JSON API, but its
route/alignment data depends on participation data that ends in 2025 ("Receiver not found" for
2026 players) and its play-by-play explorer is built on the same nflverse pbp we already use.
