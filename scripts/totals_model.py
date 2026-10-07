"""Game totals: a projected total for every game against the market total, with the drivers.

What the board uses is market_total(): the Vegas total plus an adjustment for weather and injuries,
fit on the market's past misses (total - closing total). Building a total from scratch out of all
the inputs below never matched the closing line (2023-2025 MAE 10.35 vs 10.12, 51.8% O/U); adding
pace, efficiency or situational spots to the market adjustment made it worse, so the market already
prices them. Walk-forward 2022-2025, over/under vs the closing total:
    market + dome/cold/injuries (no wind)        595-534 (52.7%), MAE 10.173 vs 10.189
    market + measured wind/dome/cold/injuries     608-521 (53.9%), MAE 10.161; adjustment 2+: 61-48
Measured wind is hindsight; the forecast a bettor had is noisier, so expect less than that.
2026 through week 4: 30-34. Breakeven at -110 is 52.4%.

Inputs (all known before kickoff; walk-forward by season):
- Efficiency: opponent-adjusted EPA/play (pass and run, offense vs the opposing defense, non-
  garbage plays, from game_model) combined by each offense's pass rate, plus success rates.
- Pace: neutral-situation seconds per snap (win probability 20-80%, quarters 1-3) and plays per
  game, recent games weighted most (half-life 3 games).
- QBs: both starters' adjustments (game_model.qb_adjustment).
- Weather: dome/closed roof, wind and temperature. Played games use nflverse's measured values;
  upcoming games use the live forecast. Measured wind is a little kinder to the model than the
  forecast a bettor had (see README, wind rule).
- Injuries: snap share lost on both offensive lines, both skill groups and both defenses.
- Situations: short week (Thursday), off a bye, a West Coast team at a 1 PM ET kickoff, late-season
  divisional game, prime time.

    python scripts/totals_model.py      # backtest; writes data/totals_projections.parquet
"""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

import game_model as gm

DATA = Path(__file__).resolve().parent.parent / "data"
RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]
WEST = {"SEA", "SF", "LA", "LAC", "LV", "ARI"}
PACE_HALF_LIFE = 3  # games
# feature groups: name -> (label, columns)
GROUPS = {
    "eff": ("Efficiency (adjusted EPA)", ["exp_epa", "exp_sr"]),
    "pace": ("Pace", ["sec_per_snap", "plays"]),
    "qb": ("Quarterbacks", ["qb_sum"]),
    "weather": ("Weather", ["dome", "wind", "wind15", "cold"]),
    "inj": ("Injuries", ["inj_ol_sum", "inj_skill_sum", "inj_def_sum"]),
    "spot": ("Situation", ["short_week", "off_bye", "west_early", "div_late", "prime"]),
}


def neutral_pace(sched: pl.DataFrame) -> pl.DataFrame:
    """Per team-game, rolled over prior games: seconds between snaps in neutral situations."""
    pbp = (pl.scan_parquet(DATA / "raw" / "pbp" / "*.parquet")
           .filter(pl.col("play_type").is_in(["pass", "run"]))
           .select("game_id", "posteam", "drive", "play_id", "game_seconds_remaining", "wp", "qtr").collect()
           .with_columns(pl.col("posteam").replace(RENAMES))
           .sort("game_id", "play_id")
           .with_columns((pl.col("game_seconds_remaining") - pl.col("game_seconds_remaining").shift(-1).over("game_id", "posteam", "drive"))
                         .alias("gap")))
    neutral = pl.col("wp").is_between(0.2, 0.8) & (pl.col("qtr") <= 3) & pl.col("gap").is_between(1, 60)
    g = pbp.filter(neutral).group_by("game_id", pl.col("posteam").alias("team")).agg(pl.col("gap").mean().alias("ssnap"))
    order = pl.concat([sched.select("game_id", "gameday", pl.col(f"{s}_team").alias("team")) for s in ("home", "away")])
    return (order.join(g, on=["game_id", "team"], how="left").sort("team", "gameday")
            .with_columns(pl.col("ssnap").shift(1).ewm_mean(half_life=PACE_HALF_LIFE, ignore_nulls=True).over("team").alias("e_ssnap"))
            .select("game_id", "team", "e_ssnap"))


LIVE_WIND = DATA.parent / "props_log" / "forecast_wind.csv"


def live_wind() -> dict:
    """Forecast wind for unplayed games, saved by fetch_wind() (props_week.forecast_wind)."""
    if not LIVE_WIND.exists():
        return {}
    w = pl.read_csv(LIVE_WIND)
    return dict(zip(w["game_id"], w["wind"]))


def fetch_wind(games: pl.DataFrame) -> dict:
    """Fetch the forecast for these games and merge it into props_log/forecast_wind.csv."""
    import datetime as dt
    import props_week
    new = props_week.forecast_wind(games)
    old = live_wind()
    old.update(new)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    pl.DataFrame({"game_id": list(old), "wind": list(old.values()), "fetched": [stamp] * len(old)}).write_csv(LIVE_WIND)
    return old


def build(wind: dict | None = None) -> pl.DataFrame:
    wind = live_wind() if wind is None else wind
    """One row per game: both teams' inputs, the market total and the actual total."""
    df = gm.build()
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
             .filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))
             .with_columns(pl.col("home_team", "away_team").replace(RENAMES)))
    df = df.join(neutral_pace(sched), on=["game_id", "team"], how="left")
    # each offense's expected EPA/play this game: its pass and run ratings plus the opposing defense's
    pr = pl.col("off_pass_rate")
    df = df.with_columns((pr * pl.col("pass") + (1 - pr) * pl.col("run")).alias("x_epa"),
                         (pr * pl.col("pass_sr") + (1 - pr) * pl.col("run_sr")).alias("x_sr"))
    h = df.filter(pl.col("is_home"))
    a = df.filter(~pl.col("is_home")).select("game_id", *[pl.col(c).alias(f"a_{c}") for c in
                  ("team", "x_epa", "x_sr", "e_ssnap", "off_plays", "def_plays", "qb_adj", "inj_ol", "inj_skill", "inj_def",
                   "rest_days", "points", "qb_name")])
    g = h.join(a, on="game_id")
    # unplayed games have no roof listed: use the stadium's usual setting (retractable roofs are
    # usually closed), from its past games
    usual = (sched.filter(pl.col("roof").is_not_null()).group_by("stadium_id")
             .agg(pl.col("roof").mode().first().alias("usual_roof")))
    s = (sched.join(usual, on="stadium_id", how="left").with_columns(pl.col("roof").fill_null(pl.col("usual_roof")))
         .select("game_id", "roof", "temp", "wind", "div_game", "weekday", "gametime"))
    if wind:  # forecast for unplayed games (nflverse fills wind only after kickoff)
        s = s.with_columns(pl.col("game_id").replace_strict(wind, default=None, return_dtype=pl.Float64)
                           .fill_null(pl.col("wind").cast(pl.Float64)).alias("wind"))
    g = g.join(s, on="game_id", how="left")
    indoor = pl.col("roof").is_in(["dome", "closed"])
    return g.with_columns(
        (pl.col("x_epa") + pl.col("a_x_epa")).alias("exp_epa"),
        (pl.col("x_sr") + pl.col("a_x_sr")).alias("exp_sr"),
        (pl.col("e_ssnap") + pl.col("a_e_ssnap")).alias("sec_per_snap"),
        (pl.col("off_plays") + pl.col("opp_def_plays") + pl.col("a_off_plays") + pl.col("def_plays")).alias("plays"),
        (pl.col("qb_adj") + pl.col("a_qb_adj")).alias("qb_sum"),
        indoor.cast(pl.Int8).alias("dome"),
        (~indoor & pl.col("wind").is_null()).alias("wind_missing"),
        pl.when(indoor).then(0.0).otherwise(pl.col("wind").cast(pl.Float64).fill_null(8.0)).alias("wind"),
        pl.when(indoor).then(0).otherwise((pl.col("wind").fill_null(0) >= 15).cast(pl.Int8)).alias("wind15"),
        pl.when(indoor).then(0).otherwise((pl.col("temp").fill_null(60) < 25).cast(pl.Int8)).alias("cold"),
        (pl.col("inj_ol") + pl.col("a_inj_ol")).alias("inj_ol_sum"),
        (pl.col("inj_skill") + pl.col("a_inj_skill")).alias("inj_skill_sum"),
        (pl.col("inj_def") + pl.col("a_inj_def")).alias("inj_def_sum"),
        (pl.min_horizontal("rest_days", "a_rest_days") <= 5).cast(pl.Int8).alias("short_week"),
        ((pl.col("rest_days") >= 13).cast(pl.Int8) + (pl.col("a_rest_days") >= 13).cast(pl.Int8)).alias("off_bye"),
        (pl.col("a_team").is_in(list(WEST)) & ~pl.col("team").is_in(list(WEST))
         & (pl.col("gametime") == "13:00")).cast(pl.Int8).alias("west_early"),
        ((pl.col("div_game") == 1) & (pl.col("week") >= 10)).cast(pl.Int8).alias("div_late"),
        ((pl.col("gametime") >= "19:00") | ~pl.col("weekday").is_in(["Sunday"])).cast(pl.Int8).alias("prime"),
        (pl.col("points") + pl.col("a_points")).alias("total"))


def feats(groups: list[str]) -> list[str]:
    return [c for g in groups for c in GROUPS[g][1]]


def project(g: pl.DataFrame, groups: list[str], alpha: float = 10.0) -> pl.DataFrame:
    """Walk-forward projected total; contributions per group are relative to the training mean."""
    cols = feats(groups)
    out = []
    for s in TEST_SEASONS:
        tr = g.filter((pl.col("season") < s) & pl.col("total").is_not_null()).drop_nulls(cols)
        te = g.filter(pl.col("season") == s).drop_nulls(cols)
        X, Xt = tr.select(cols).to_numpy(), te.select(cols).to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = Ridge(alpha=alpha).fit((X - mu) / sd, tr["total"].to_numpy())
        Z = (Xt - mu) / sd
        contrib = {gr: (Z[:, [cols.index(c) for c in GROUPS[gr][1]]] * m.coef_[[cols.index(c) for c in GROUPS[gr][1]]]).sum(1)
                   for gr in groups}
        out.append(te.with_columns(pl.Series("proj_total", m.predict(Z)), pl.lit(m.intercept_).alias("base_total"),
                                   *[pl.Series(f"t_{gr}", v) for gr, v in contrib.items()]))
    return pl.concat(out)


MARKET_GROUPS = {"weather": ["dome", "wind", "wind15", "cold"], "inj": ["inj_ol_sum", "inj_skill_sum", "inj_def_sum"]}
MARKET_ALPHA = 100.0


def market_total(g: pl.DataFrame) -> pl.DataFrame:
    """Walk-forward: Vegas total + a ridge fit of the market's past misses on weather and
    injuries. Returns proj_total, the adjustment and its weather / injury parts."""
    cols = [c for v in MARKET_GROUPS.values() for c in v]
    g = g.filter(pl.col("total_line").is_not_null()).with_columns((pl.col("total") - pl.col("total_line")).alias("resid"))
    out = []
    for s in TEST_SEASONS:
        tr = g.filter((pl.col("season") < s) & pl.col("resid").is_not_null()).drop_nulls(cols)
        te = g.filter(pl.col("season") == s).drop_nulls(cols)
        X, Xt = tr.select(cols).to_numpy(), te.select(cols).to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = Ridge(alpha=MARKET_ALPHA).fit((X - mu) / sd, tr["resid"].to_numpy())
        Z = (Xt - mu) / sd
        # an unplayed game with no Out/Doubtful players on either team is a report not out yet,
        # not a healthy game: hold its injury inputs at the average until it is
        inj = [cols.index(c) for c in MARKET_GROUPS["inj"]]
        pending = (te["total"].is_null() & (te.select(MARKET_GROUPS["inj"]).sum_horizontal() == 0)).to_numpy()
        Z[np.ix_(pending, inj)] = 0.0
        parts = {k: (Z[:, [cols.index(c) for c in v]] * m.coef_[[cols.index(c) for c in v]]).sum(1) for k, v in MARKET_GROUPS.items()}
        adj = m.intercept_ + Z @ m.coef_
        out.append(te.with_columns(pl.Series("inj_pending", pending), pl.Series("adj", adj), pl.Series("adj_weather", parts["weather"]),
                                   pl.Series("adj_inj", parts["inj"]),
                                   pl.Series("adj_base", adj - parts["weather"] - parts["inj"]),
                                   (pl.col("total_line") + pl.Series(adj)).alias("proj_total")))
    return pl.concat(out)


def grade(p: pl.DataFrame, min_gap: float = 0.0) -> dict:
    d = (p.filter(pl.col("total").is_not_null() & pl.col("total_line").is_not_null())
         .with_columns((pl.col("proj_total") - pl.col("total_line")).alias("edge"))
         .filter(pl.col("edge").abs() >= min_gap))
    res = (pl.col("total") - pl.col("total_line")) * pl.col("edge").sign()
    w, l = d.filter(res > 0).height, d.filter(res < 0).height
    return {"n": d.height, "w": w, "l": l, "pct": w / max(w + l, 1),
            "mae": (d["total"] - d["proj_total"]).abs().mean() if d.height else np.nan,
            "mae_line": (d["total"] - d["total_line"]).abs().mean() if d.height else np.nan}


if __name__ == "__main__":
    g = build()
    g.write_parquet(DATA / "totals_features.parquet")
    mt = market_total(g)
    mt.select("game_id", "season", "week", "total_line", "proj_total", "adj", "adj_weather", "adj_inj", "adj_base",
              "wind", "wind_missing", "inj_pending", "dome", "total").write_parquet(DATA / "totals_projections.parquet")
    for lab, f in (("2022-2025", pl.col("season") < 2026), ("2026", pl.col("season") == 2026)):
        for gap in (0, 1, 2):
            r = grade(mt.filter(f), gap)
            print(f"market total {lab} adj>={gap}: {r['w']}-{r['l']} ({r['pct']:.1%}) MAE {r['mae']:.3f} vs line {r['mae_line']:.3f}")
    print("from-scratch variants (for reference):")
    for groups in (["eff"], ["eff", "pace"], ["eff", "pace", "qb"], ["eff", "pace", "qb", "weather"],
                   ["eff", "pace", "qb", "weather", "inj"], list(GROUPS)):
        p = project(g, groups)
        b, b3, c = grade(p.filter(pl.col("season") < 2026)), grade(p.filter(pl.col("season") < 2026), 3), grade(p.filter(pl.col("season") == 2026))
        print(f"{'+'.join(groups):32s} 22-25: {b['w']}-{b['l']} ({b['pct']:.1%}) MAE {b['mae']:.2f} vs {b['mae_line']:.2f} | "
              f"3+: {b3['w']}-{b3['l']} ({b3['pct']:.1%}) | 2026: {c['w']}-{c['l']} ({c['pct']:.1%})")
