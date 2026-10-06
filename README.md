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

## Man/zone coverage (tested, not used)
nflverse's participation data has per-play coverage (man/zone, shell, route) for 2018-2025; the
feed ended after 2025, so for a current-season game only last season's tendencies are knowable.
`props.coverage_feats()` builds them: each defense's prior-season man rate and each receiver's
prior-season yards per target vs man and vs zone (shrunk), plus the matchup product. They add
nothing: a defense's man rate has only 0.37 year-to-year correlation, and the features' correlation
with a receiver beating his own average is ~0 (receiving MAE 22.77 -> 22.79-22.82 with them, receptions
unchanged). They stay computed in the feature frame for re-testing if a current-season source appears.

## Vacated volume: when a starter is out, the backup is projected on the starter's work
`props.vacated_volume()` takes the rolling carry share (RBs) and target share (WR/TE/RB) of every
player listed Out/Doubtful and redistributes it to the active players at the position in proportion
to their own usage. Only part of the vacated work actually reaches known backups (the rest goes to
call-ups or the team runs/throws less at the spot): fitted on 2018-2021, 35% of carries and 19% of
targets. The model's volume inputs are the adjusted ones (`e_carries_adj`, `e_targets_adj`, ...).
Check on 2022-2025: backups with a starter out got 13.4 carries vs 10.4 from their own history and
13.1 adjusted. Overall MAE is unchanged (differences < 0.03); on games with a large vacancy the
error drops (rushing 23.0 -> 22.1, receiving 22.4 -> 22.0, receptions 1.60 -> 1.52).

Two fixes came out of the same case: the role rank (WR1/RB1...) is now computed among active
players, so a listed-Out starter no longer holds the RB1 slot in an upcoming game, and per-unit
rates (yards per carry/target/attempt) are null rather than 0/0 = NaN in games with no volume,
which had been poisoning some players' rolling averages.
Example (2026 week 4): Seattle's two lead backs Out moved Emanuel Wilson's projection from 24.7 to
48.4 rushing yards against a 43.5-46.5 line.

## Past prop lines from prop-line.com (free tier)
`python scripts/propline_history.py [--only-week N] [--markets ...] [--scale S] [--budget B]`
pulls each player's main line per book per game from `/players/{name}/history` (lines are visible on
the free tier; prices and results are redacted, so actual stats come from nflverse) into
`props_log/propline/` (committed; one cached file per player-market, one request each, 1000/day).
`--no-fetch` grades from the cache. The report per week: over rate by market, biggest overs and
unders vs the consensus (median) line, and over rate by team. Coverage starts 2026-08-14 (Bovada),
DraftKings 08-27, FanDuel/Pinnacle 09-10, so weeks 1-4 of 2026 are available.
Mind the selection: a fetch restricted by usage *in the graded week* (e.g. `--only-week 3 --scale 2`)
keeps only players who had a big game, which inflates the over rate; fetch the full season list
before reading the hit rates.

## Injury type, playing through it, and coming back (scripts/injury_types.py)

Skill-player games 2018-2025, grouped from the weekly injury reports; yards vs the player's
pre-game average (median), snap share vs average, and how often he went under the walk-forward
model (2022-2025):

| group | games | yards vs avg | snaps vs avg | under the model |
|---|---|---|---|---|
| healthy | 35,446 | 0.84 | 1.02 | 50% |
| playing through (on the report, played) | 2,576 | 0.79 | 1.00 | 50% |
| first game back after 1 missed | 2,051 | 0.58 | 0.89 | 58% |
| first game back after 2+ missed | 1,779 | 0.45 | 0.76 | 66% |

Playing through an injury is already priced (the Questionable / practice features). The miss is
the return game: snaps are managed and the model, whose rolling averages only see games played,
overshot returners. By injury, playing through ankle/foot ran under the model ~56-59%, shoulder,
hamstring, knee and concussion about 50%; the report says "Ankle", not high vs low, and the
per-injury samples are small. In the walk-forward model `missed_prev` (games missed right before
this one, capped at 4) helped every market (MAE pass 62.46 -> 62.27, rush 25.04 -> 25.03, rec
22.72 -> 22.71, receptions 1.654 -> 1.653; returning QBs 85.1 -> 78.4, returning receivers
20.5 -> 19.8). Adding injury-type flags on top changed nothing, so only `missed_prev` is used.

## QB passing yards as attempts x yards per attempt (tested, not used)

`scripts/qb_two_stage.py` splits passing yards into an attempts model (game script, pace, PROE,
the QB's attempts, plus new features: attempts / plays / YPA the defense allows, the opponent
offense's pace and pass rate) and a yards-per-attempt model, and multiplies them. Walk-forward
MAE on passing yards, 2022-2025 (2,212 QB games):

| variant | MAE |
|---|---|
| current one-step model | **62.46** |
| one-step + the volume features | 63.15 |
| attempts x YPA | 65.52 (65.61 after a bias correction; the gap is not bias) |
| average of the two | 63.26 |

Attempts are close to unpredictable beyond the QB's own recent average (model MAE 7.85 vs 7.87):
game script and pace add almost nothing. The YPA model does beat the recent average (1.39 vs 1.47)
but not by enough. Against the 2026 prop-line passing lines (n=115) no variant beats the line
(line MAE 60.7), and the current model's 15+ yard disagreements with the line won only 16 of 44:
QB passing yards stay a market where we take the line, not the model.

## The model vs the market line (and the blend)
`python scripts/line_vs_model.py` joins every graded line we hold (prop-line history + our own
snapshots) to out-of-sample model projections (trained on earlier seasons) and measures what the
model adds. On 647 player-markets from 2026 weeks 1-4: the consensus line beats the model in every
yardage market (MAE passing 60.2 vs 62.3, rushing 23.3 vs 24.7, receiving 26.1 vs 26.7); receptions
favour the model (2.05 vs 2.14, n=68). Betting the model's side of the line won 51.5% (breakeven
52.4%). The best blend `line + k*(model - line)` has k = 0.0-0.2 for yardage, 1.0 for receptions.
Streaks go the other way: after an over, players went over 50% the next week; after an under, 62%.
`props_week.py` now bets on the blended projection (`BLEND`: 0.1/0.2/0.2/0.5), which shrinks the
model's disagreement with the market and the number of "edges" with it. Re-run `line_vs_model.py`
as graded weeks accumulate and update `BLEND`.

## Where the data lives
**Committed (browsable in the repo), under `props_log/`:**
- `<timestamp>.csv`: every prop line snapshot from The Odds API (book, player, market, side, point, price).
- `projections/<timestamp>.csv`: the model's pre-kickoff projection, blended projection, P(over) and EV
  for every book line in that snapshot.
- `grades/<timestamp>_lines.csv` / `_model.csv`: those lines and picks graded against actual stats.
- `propline/`: prop-line.com history, one file per player-market (every book's main line per game
  since August) and `graded_<season>.csv` with the consensus line vs the actual stat.
- `features_<season>.csv` (from `scripts/export_stats.py`): one row per player-game for QB/RB/WR/TE
  with the actual stats and every pre-game feature the model uses (usage, efficiency, Next Gen and PFR
  stats, injuries and redistributed volume, QB-specific usage, role, opponent defense, game
  environment). The compact, human-readable version of the model's input.

**Not committed (`data/`, gitignored, ~155 MB), regenerated by `python scripts/pull_data.py`:**
`data/raw/*.parquet` straight from nflverse: schedules, play-by-play (100 MB), player and team
stats, injuries, weekly rosters, snap counts, players, Next Gen Stats (receiving/rushing/passing),
PFR advanced stats (rec/rush/pass), ff_opportunity, participation (coverage, through 2025).
`data/props_features.parquet` is the full feature frame (all seasons, ~400 columns) and
`data/props_*_preds.parquet` the walk-forward predictions used for calibration.

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

### Discounting receiving games played hurt
For WR/TE/RB, the rolling receiving averages in `props.py` (yards, receptions, targets, target
share, air-yards share, yards per target, aDOT) count a past game at `HURT_WEIGHT = 0.25` if the
player played it hurt: Questionable/Doubtful without full practice, or his first game back from
Out. The discount only applies while he is healthy now. A player still on the report keeps his
hurt games at full weight, since they are the best guide to how he plays today. QBs and rushing
stats are untouched.

Same player-games, 2022-2025 (passing and rushing are identical to before):

| Hurt-game weight | Rec yds MAE | Receptions MAE |
| --- | --- | --- |
| 1 (none) | 22.712 | 1.653 |
| 0.5 | 22.693 | 1.651 |
| 0.25 (used) | 22.684 | 1.650 |
| 0 | 22.679 | 1.651 |

Applying it to every rolling stat made rushing worse, and a no-op version (weight 0.9999) showed
rushing MAE moves about 0.08 on tiny feature changes, so rushing gains under that are noise.
Example: Drake London going into 2026 week 4. His 2025 weeks 16-18 on a bad knee drop out, and
his recent receiving average goes 84.6 -> 93.2. The model's projection went 73.9 -> 69.9 anyway:
it weighs the average against everything else, so one player's number need not follow it.

### Defensive starters out in the weekly projections
`props_week.py` output now has `opp_starters_out` (starters out in the unit the player faces:
the front seven for rushing, the secondary otherwise) and `opp_def_out`, every defensive starter
Out/Doubtful with snap share, e.g. `Kaden Elliss (LB Out, 100%), Carl Granderson (LB Out, 66%)`.
A starter played at least half the snaps over his last 4 games. This is display only. The model
inputs stay the snap-share totals `opp_inj_front` / `opp_inj_db` (`opp_out`).

### Raw + ratio blend (top players were projected too low)
Tree models can't extrapolate, so the props model squeezed the best players toward the middle:
receivers averaging 95+ yards beat its projection 56% of the time (65-80: 49%). Each market now
averages two models, one on the raw stat and one on the stat as a ratio to the player's recent
average (`props.BlendModel`), used by `props.py`, `props_week.py` and `line_vs_model.py`.

| Market | MAE raw | MAE blend | Top-5% players beating the projection, raw -> blend |
| --- | --- | --- | --- |
| Passing yds | 62.27 | 62.17 | |
| Rushing yds | 25.03 | 24.98 | |
| Receiving yds | 22.70 | 22.67 | 53.9% -> 51.1% |
| Receptions | 1.651 | 1.646 | 55.4% -> 52.7% |

On the 2026 lines graded so far, the model's side wins 51.6% (was 50.6%; breakeven 52.4%). The
best line blend weights are unchanged, so `props_week.BLEND` stays. Projections still sit below
big recent averages, and that part is real: receivers averaging 95+ posted a median of 80.
Drake London for 2026 week 4: recent average 93.2, projection 69.9 -> 72.4, line 80.5.

### Special teams (tested, not used)
`props.special_teams()` adds each team's rolling net special-teams EPA (kickoffs, punts, field
goals, extra points), where its drives start, and the opponent's special-teams EPA and drive
start allowed (`e_st_epa`, `e_drive_start`, `opp_st_epa`, `opp_drive_start_alw`). None helped in
the walk-forward backtest, so they are not model inputs. The implied total and spread already
price special teams.

| Special-teams features | Pass yds | Rush yds | Rec yds | Receptions |
| --- | --- | --- | --- | --- |
| None (current) | 62.17 | 24.98 | 22.67 | 1.646 |
| All four | 62.42 | 25.00 | 22.67 | 1.648 |
| Field position only | 62.20 | 25.00 | 22.68 | 1.645 |
| ST EPA only | 62.23 | 25.03 | 22.71 | 1.648 |

## Game board (projected scores beside the spread)
```
python scripts/game_model.py    # walk-forward backtest, writes data/game_projections.parquet
python scripts/game_pages.py    # this week + last week -> site/nfl_board.html (or: 2026 4 5)
```
`game_model.py` rates every team each week on opponent-adjusted pass and rush EPA and success
rate (adjust.py's ridge, two years with a 180-day half-life), plus pass rate and plays per game.
It adds a QB adjustment (the listed starter's shrunk EPA/dropback minus the passers behind the
ratings), home field and rest. A linear points model, fit on earlier seasons, turns these into a
score for each side, so the margin splits exactly into each input's contribution.

`game_pages.py` renders a board (projected score, Vegas line, model line, model side, graded
result) and a breakdown per game: the number vs the market, where the margin comes from,
quarterbacks, percentile ratings and per-play team ratings.

Against the closing spread it does not beat the market:

| | ATS | Win % | Margin MAE (model / line) |
| --- | --- | --- | --- |
| 2022-2025, all games | 550-560-29 | 49.5% | 10.02 / 9.54 |
| 2022-2025, 3+ point gaps | 162-180-11 | 47.4% | 10.67 / 9.38 |
| 2026 through week 4 | 31-27-5 | 53.4% | 9.38 / 9.43 |

Lines here are nflverse's `spread_line` (closing for played games), not the opener a site would
show, so gaps are smaller than against an early line.
