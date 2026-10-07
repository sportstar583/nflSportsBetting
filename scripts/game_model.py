"""Game projections: a score for every game, beside the market spread, with the margin broken down.

1. Ratings (walk-forward, every week): for each team, offense and defense, opponent-adjusted per-play
   pass EPA, rush EPA, pass success rate and rush success rate. Fit with adjust._fit (ridge over
   the prior two years, 180-day half-life, so early-season ratings lean on last season). Plus
   decayed pass rate and plays per game (not opponent-adjusted).
2. QB: the listed starter's shrunk career EPA/dropback (features.qb_form) minus the dropback-
   weighted rating of the passers who actually played in the games the ratings came from. Zero
   when the starter is the one already in the numbers.
3. Points: a linear model of a team's points on its offense + the opponent's defense, QB, home
   field, rest and pace, trained on earlier seasons. Linear, so the projected margin splits
   exactly into each input's contribution (the "where the margin comes from" bars).

    python scripts/game_model.py            # backtest 2022-2025 + 2026 so far, writes data/game_projections.parquet
"""
from pathlib import Path

import polars as pl
from sklearn.linear_model import Ridge

import adjust
import features

DATA = Path(__file__).resolve().parent.parent / "data"
RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]
SPLITS = {"pass": ("pass_epa", "n_pass"), "run": ("run_epa", "n_run"),
          "pass_sr": ("pass_sr", "n_pass"), "run_sr": ("run_sr", "n_run")}
# model inputs, each "own offense + opponent defense" except qb/home/rest; name -> label on the page
GROUPS = {"pass": "Passing game", "run": "Running game", "pass_sr": "Pass consistency",
          "run_sr": "Rush consistency", "qb": "Quarterback", "home": "Home field", "rest": "Rest",
          "pace": "Pace", "inj_ol": "Line injuries", "inj_skill": "Skill-player injuries",
          "inj_def": "Defensive injuries"}


DROP_GARBAGE = True


def team_games(sched: pl.DataFrame, drop_garbage: bool = DROP_GARBAGE) -> pl.DataFrame:
    """adjust.team_games plus success rates, pass rate and plays per team-game. With drop_garbage,
    efficiency and pass rate come from non-garbage plays only; plays per game counts every play."""
    path = DATA / "raw" / "pbp" / "*.parquet"
    tg = adjust.team_games(path, sched, drop_garbage)
    keep = ~adjust.GARBAGE.fill_null(False) if drop_garbage else pl.lit(True)
    pbp = (pl.scan_parquet(path)
           .filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null())
           .select("game_id", "posteam", "play_type", "success", keep.alias("keep")).collect())
    pt = pl.col("play_type")
    sr = pbp.group_by("game_id", "posteam").agg(
        pl.col("success").filter((pt == "pass") & pl.col("keep")).mean().alias("pass_sr"),
        pl.col("success").filter((pt == "run") & pl.col("keep")).mean().alias("run_sr"),
        pl.len().alias("plays"))
    return (tg.join(sr, on=["game_id", "posteam"], how="left")
            .with_columns((pl.col("n_pass") / pl.col("n")).alias("pass_rate")))


def ratings(tg: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Pre-game ratings for every team in every scheduled game (adjusted per-play splits,
    decayed pass rate and plays, offense and defense) and the passers behind them."""
    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    games = sched.select("game_id", "season", "week", "home_team", "away_team",
                         pl.col("gameday").str.to_date().alias("date"))
    qb_rows = qb_dropbacks(sched)
    out = []
    for (season, week), grp in games.group_by("season", "week", maintain_order=True):
        asof = grp["date"].min()
        train = tg.filter((pl.col("date") < asof) & ((asof - pl.col("date")).dt.total_days() <= adjust.MAX_AGE_DAYS))
        if train.height < 200:
            continue
        fits = {s: adjust._fit(train, teams, y, n, asof) for s, (y, n) in SPLITS.items()}
        w = (0.5 ** ((asof - pl.col("date")).dt.total_days() / adjust.HALF_LIFE_DAYS))
        dec = lambda by, cols: (train.with_columns(w.alias("w")).group_by(by)  # noqa: E731
                                .agg([((pl.col(c) * pl.col("w")).sum() / pl.col("w").sum()).alias(c) for c in cols]))
        off = dict((r[0], r[1:]) for r in dec("posteam", ["pass_rate", "plays"]).iter_rows())
        dfn = dict((r[0], r[1:]) for r in dec("defteam", ["pass_rate", "plays"]).iter_rows())
        # the passers in the rating window, weighted like the games: whose play the ratings reflect
        q = (qb_rows.filter((pl.col("date") < asof) & ((asof - pl.col("date")).dt.total_days() <= adjust.MAX_AGE_DAYS))
             .with_columns((pl.col("n") * w).alias("wn")).group_by("team", "qb_id").agg(pl.col("wn").sum()))
        for gid, h, a in grp.select("game_id", "home_team", "away_team").iter_rows():
            for tm in (h, a):
                row = {"game_id": gid, "team": tm,
                       "off_pass_rate": off.get(tm, (None, None))[0], "off_plays": off.get(tm, (None, None))[1],
                       "def_pass_rate": dfn.get(tm, (None, None))[0], "def_plays": dfn.get(tm, (None, None))[1]}
                for s, f in fits.items():
                    row[f"off_{s}"], row[f"def_{s}"] = f[tm]
                out.append(row)
        if q.height:
            out_q = q.with_columns(pl.lit(grp["game_id"].to_list()).alias("gids"))
            QB_WINDOW.append(out_q.explode("gids", empty_as_null=True).rename({"gids": "game_id"}))
    return pl.DataFrame(out)


QB_WINDOW: list[pl.DataFrame] = []  # filled by ratings(): (game_id, team, qb_id, weighted dropbacks)


def qb_dropbacks(sched: pl.DataFrame) -> pl.DataFrame:
    """Dropbacks per passer per team-game."""
    return (pl.scan_parquet(DATA / "raw" / "pbp" / "*.parquet")
            .filter((pl.col("qb_dropback") == 1) & pl.col("passer_player_id").is_not_null())
            .group_by("game_id", "posteam", "passer_player_id").agg(pl.len().alias("n")).collect()
            .with_columns(pl.col("posteam").replace(RENAMES))
            .rename({"posteam": "team", "passer_player_id": "qb_id"})
            .join(sched.select("game_id", pl.col("gameday").str.to_date().alias("date")), on="game_id"))


# EPA/dropback below league average that a QB with no history starts at (a rookie or backup, not an
# average starter). 2022-2025 margin MAE: 0 9.971, -0.05 9.949, -0.10 9.932, -0.15 9.923, -0.20 9.919.
QB_PRIOR_OFFSET = -0.15


def qb_adjustment(sched: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: listed starter's rating minus the rating of the passers in the window."""
    form, league = features.qb_form(sched, QB_PRIOR_OFFSET)
    # each QB's rating entering each game, as of that date (his latest prior game's running value)
    form = form.join(sched.select("game_id", pl.col("gameday").str.to_date().alias("date")), on="game_id")
    latest = form.sort("date").select("qb_id", "date", "qb_epa", "qb_n")
    starters = pl.concat([
        sched.select("game_id", pl.col("home_team").alias("team"), pl.col("home_qb_id").alias("qb_id"),
                     pl.col("home_qb_name").alias("qb_name"), pl.col("gameday").str.to_date().alias("date")),
        sched.select("game_id", pl.col("away_team").alias("team"), pl.col("away_qb_id").alias("qb_id"),
                     pl.col("away_qb_name").alias("qb_name"), pl.col("gameday").str.to_date().alias("date"))])

    def rate(df: pl.DataFrame) -> pl.DataFrame:  # rating strictly before the game date
        return (df.with_columns((pl.col("date") - pl.duration(days=1)).alias("asof")).sort("asof")
                .join_asof(latest.rename({"date": "asof"}).sort("asof"), on="asof", by="qb_id", strategy="backward",
                           check_sortedness=False)
                .with_columns(pl.col("qb_epa").fill_null(league + QB_PRIOR_OFFSET)).drop("asof"))
    st = rate(starters).rename({"qb_epa": "starter_epa"}).drop("qb_n")
    window = pl.concat(QB_WINDOW).join(sched.select("game_id", pl.col("gameday").str.to_date().alias("date")), on="game_id")
    window = rate(window).group_by("game_id", "team").agg(
        ((pl.col("qb_epa") * pl.col("wn")).sum() / pl.col("wn").sum()).alias("window_epa"))
    return (st.join(window, on=["game_id", "team"], how="left")
            .with_columns((pl.col("starter_epa") - pl.col("window_epa")).fill_null(0).alias("qb_adj"))
            .select("game_id", "team", "qb_id", "qb_name", "starter_epa", "window_epa", "qb_adj"))


def build(drop_garbage: bool = DROP_GARBAGE) -> pl.DataFrame:
    """One row per team-game: the team's inputs (own offense + opponent defense) and points."""
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
             .filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))
             .with_columns(pl.col("home_team", "away_team").replace(RENAMES)))
    QB_WINDOW.clear()
    r = ratings(team_games(sched, drop_garbage), sched)
    qb = qb_adjustment(sched)
    sides = []
    for home in (True, False):
        t, o = ("home", "away") if home else ("away", "home")
        sides.append(sched.select(
            "game_id", "season", "week", "gameday", "gametime", "stadium", "location", "spread_line", "total_line",
            pl.col(f"{t}_team").alias("team"), pl.col(f"{o}_team").alias("opp"),
            pl.col(f"{t}_score").alias("points"), pl.col(f"{o}_score").alias("opp_points"),
            pl.lit(home).alias("is_home"),
            pl.when(pl.col("location") == "Neutral").then(0).otherwise(1 if home else -1).alias("home"),
            (pl.col(f"{t}_rest") - pl.col(f"{o}_rest")).clip(-7, 7).alias("rest"),
            pl.col(f"{t}_rest").alias("rest_days")))
    df = pl.concat(sides)
    ro = r.select("game_id", "team", *[pl.col(c) for c in r.columns if c not in ("game_id", "team")])
    rd = r.select("game_id", pl.col("team").alias("opp"), *[pl.col(c).alias(f"opp_{c}") for c in r.columns if c not in ("game_id", "team")])
    # snap share lost to Out/Doubtful players (official report + props_log/manual_out.csv): the
    # team's offensive line and skill players, and the opponent's defense
    inj = features.injury_snaps().select("season", "week", "team", "inj_ol", "inj_skill", "inj_def")
    df = (df.with_columns(pl.col("season", "week").cast(pl.Int32))
          .join(inj.drop("inj_def"), on=["season", "week", "team"], how="left")
          .join(inj.select("season", "week", pl.col("team").alias("opp"), "inj_def"), on=["season", "week", "opp"], how="left")
          .with_columns(pl.col("inj_ol", "inj_skill", "inj_def").fill_null(0)))
    df = (df.join(ro, on=["game_id", "team"]).join(rd, on=["game_id", "opp"])
          .join(qb, on=["game_id", "team"], how="left").with_columns(pl.col("qb_adj").fill_null(0)))
    return df.with_columns(
        [(pl.col(f"off_{s}") + pl.col(f"opp_def_{s}")).alias(s) for s in SPLITS]
        + [pl.col("qb_adj").alias("qb"), (pl.col("off_plays") + pl.col("opp_def_plays")).alias("pace")])


FEATS = list(GROUPS)


# Seasons; weights recent seasons more so totals follow the scoring level. Tested 2, 1 and 0.5: totals
# MAE 10.62 -> 10.54-10.60 but the over/under record got worse (49.2% -> 46.5-48.4%), so none.
SEASON_HALF_LIFE = None


def project(df: pl.DataFrame, half_life: float | None = SEASON_HALF_LIFE, feats: list[str] | None = None) -> pl.DataFrame:
    """Walk-forward: each season projected from a model fit on earlier seasons' team-games."""
    FEATS = feats or globals()["FEATS"]  # noqa: N806
    out = []
    for s in TEST_SEASONS:
        tr = df.filter((pl.col("season") < s) & pl.col("points").is_not_null()).drop_nulls(FEATS)
        te = df.filter(pl.col("season") == s).drop_nulls(FEATS)
        w = None if half_life is None else 0.5 ** ((s - 1 - tr["season"].to_numpy()) / half_life)
        m = Ridge(alpha=1.0).fit(tr.select(FEATS).to_numpy(), tr["points"].to_numpy(), sample_weight=w)
        X = te.select(FEATS).to_numpy()
        te = te.with_columns(pl.Series("proj", m.predict(X)), pl.lit(m.intercept_).alias("intercept"),
                             *[pl.Series(f"c_{f}", X[:, i] * m.coef_[i]) for i, f in enumerate(FEATS)])
        out.append(te)
    return pl.concat(out)


def games(proj: pl.DataFrame) -> pl.DataFrame:
    """One row per game: projected score, model spread vs the market, margin contributions."""
    h = proj.filter(pl.col("is_home"))
    a = proj.filter(~pl.col("is_home")).select("game_id", *[pl.col(c).alias(f"a_{c}") for c in proj.columns if c != "game_id"])
    g = h.join(a, on="game_id")
    return g.with_columns(
        (pl.col("proj") - pl.col("a_proj")).alias("model_margin"),  # home minus away
        *[(pl.col(f"c_{f}") - pl.col(f"a_c_{f}")).alias(f"m_{f}") for f in FEATS])


def record(g: pl.DataFrame, min_gap: float = 0.0) -> dict:
    """Against the closing spread: take the side the model favours, when it differs by min_gap+."""
    d = g.filter(pl.col("points").is_not_null() & pl.col("spread_line").is_not_null()).with_columns(
        (pl.col("points") - pl.col("a_points")).alias("result"),
        (pl.col("model_margin") - pl.col("spread_line")).alias("edge"))  # + means model likes home more
    d = d.filter(pl.col("edge").abs() >= min_gap)
    cover = (pl.col("result") - pl.col("spread_line")) * pl.col("edge").sign()
    w, l = d.filter(cover > 0).height, d.filter(cover < 0).height
    return {"n": d.height, "w": w, "l": l, "p": d.height - w - l, "pct": w / max(w + l, 1),
            "units": w - 1.1 * l,
            "mae_model": (d["result"] - d["model_margin"]).abs().mean(),
            "mae_line": (d["result"] - d["spread_line"]).abs().mean()}


def total_record(g: pl.DataFrame, min_gap: float = 0.0) -> dict:
    """Against the closing total: over when the model's total is higher, under when lower."""
    d = (g.filter(pl.col("points").is_not_null() & pl.col("total_line").is_not_null())
         .with_columns((pl.col("proj") + pl.col("a_proj")).alias("mt"), (pl.col("points") + pl.col("a_points")).alias("at"))
         .filter((pl.col("mt") - pl.col("total_line")).abs() >= min_gap))
    res = (pl.col("at") - pl.col("total_line")) * (pl.col("mt") - pl.col("total_line")).sign()
    w, l = d.filter(res > 0).height, d.filter(res < 0).height
    return {"n": d.height, "w": w, "l": l, "p": d.height - w - l, "pct": w / max(w + l, 1),
            "mae_model": (d["at"] - d["mt"]).abs().mean(), "mae_line": (d["at"] - d["total_line"]).abs().mean()}


if __name__ == "__main__":
    df = build()
    g = games(project(df))
    g.write_parquet(DATA / "game_projections.parquet")
    for s in TEST_SEASONS:
        for gap in (0, 3):
            r = record(g.filter(pl.col("season") == s), gap)
            print(f"{s} gap>={gap}: {r['w']}-{r['l']}-{r['p']} ({r['pct']:.1%}, {r['units']:+.1f}u)  "
                  f"margin MAE model {r['mae_model']:.2f} vs line {r['mae_line']:.2f}")
    for gap in (0, 3):
        r = record(g.filter(pl.col("season").is_in(TEST_SEASONS[:-1])), gap)
        print(f"2022-2025 gap>={gap}: {r['w']}-{r['l']}-{r['p']} ({r['pct']:.1%}, {r['units']:+.1f}u)  "
              f"MAE model {r['mae_model']:.2f} vs line {r['mae_line']:.2f}")
