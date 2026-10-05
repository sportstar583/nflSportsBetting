"""Project this week's props and compare them to a props_log snapshot.

For each game in the snapshot that hasn't been played, every skill player who has played for
one of its teams this season gets a stat-less row, then props.build() computes his pre-game
features exactly as in the backtest. The model is refit on every completed game and predicts
the median of each market. Outdoor wind comes from Open-Meteo's live forecast (nflverse only
fills wind after kickoff).

The projection used for betting is a blend, line + k * (model - line), with k per market from
line_vs_model.py (the market line is more accurate than the model; k is 0.1-0.2 for yardage, 0.5
for receptions). `pred` is the raw model, `pred_blend` what the EVs use.

P(over a line) comes from the walk-forward residuals that props.py saved: among out-of-sample
player-games whose prediction was close to this one, how often did actual / predicted exceed
line / predicted. Each book's price is turned into an expected value per unit staked.

    python scripts/props.py                 # once: refreshes data/props_*_preds.parquet
    python scripts/props_week.py            # latest snapshot in props_log/
    python scripts/props_week.py props_log/20261004T045441Z.csv --top 30

Each row carries `team_spread` (positive = the player's team is favored; the model uses it, along
with the implied team total, for game script) and `inj`, the player's own final injury report
(Q = Questionable, ltd = limited/missed practice, both model features). Players listed Out or
Doubtful are dropped.

Rows whose consensus line is more than NEWS_GAP away from the player's recent average are
flagged `news`: that is usually an injury, a QB change or a new role the books know about and
the model doesn't, so their big "edges" are mostly the model being stale.

Caveat: the backtest only showed the model beats a player's recent average. Nothing here has
been tested against real closing lines yet, so treat the EVs as a ranking, not as edges.
"""
import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

import props
from weather_forecast import GAME_HOURS, STADIUMS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MARKET_KEYS = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds",
               "player_reception_yds": "rec_yds", "player_receptions": "receptions"}
TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
NEIGHBORS = 400  # historical predictions used to estimate each P(over)
NEWS_GAP = 0.35  # line this far from the recent average: books likely priced injury/role news we lack
# Weight of the model next to the consensus line: projection = line + k * (model - line).
# From line_vs_model.py on 647 graded 2026 player-markets (weeks 1-4): the line beat the model in
# every yardage market (best k 0.0-0.2); receptions favoured the model (k=1.0, n=68), shrunk here.
BLEND = {"pass_yds": 0.1, "rush_yds": 0.2, "rec_yds": 0.2, "receptions": 0.5}


def norm_name(s: str) -> str:
    s = re.sub(r"[.'’]", "", s.lower()).replace("-", " ")
    return " ".join(t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"})


def payout(price: float) -> float:
    """Profit per unit staked at American odds."""
    return price / 100 if price > 0 else 100 / -price


def load_snapshot(path: Path) -> pl.DataFrame:
    snap = pl.read_csv(path)
    return snap.with_columns(pl.col("home_team").replace_strict(TEAMS).alias("home"),
                             pl.col("away_team").replace_strict(TEAMS).alias("away"),
                             pl.col("player").map_elements(norm_name, return_dtype=pl.String).alias("key"),
                             pl.col("market").replace_strict(MARKET_KEYS).alias("mkt"))


def upcoming_games(snap: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    games = snap.select("home", "away", "commence_time").unique()
    return (sched.filter(pl.col("result").is_null())
            .join(games, left_on=["home_team", "away_team"], right_on=["home", "away"])
            .sort("gameday").unique(["home_team", "away_team"], keep="first"))


def forecast_wind(games: pl.DataFrame) -> dict:
    """game_id -> mean forecast wind (mph) over kickoff and the next GAME_HOURS hours."""
    out = {}
    for g in games.filter(~pl.col("roof").is_in(["dome", "closed"]).fill_null(False)).iter_rows(named=True):
        if g["stadium_id"] not in STADIUMS:
            continue
        lat, lon = STADIUMS[g["stadium_id"]]
        kick = dt.datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        params = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m", "wind_speed_unit": "mph",
                  "timezone": "GMT", "start_date": kick.date().isoformat(),
                  "end_date": (kick + dt.timedelta(hours=GAME_HOURS + 1)).date().isoformat()}
        r = None
        for attempt in range(3):
            try:
                r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params=params)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                r = None
                err = e
                time.sleep(2 ** attempt)
        if r is None:
            print(f"  no forecast for {g['game_id']} ({err}); wind left missing", file=sys.stderr)
            continue
        h = r.json()["hourly"]
        start = kick.replace(minute=0, tzinfo=None)
        vals = [w for t, w in zip(h["time"], h["wind_speed_10m"])
                if w is not None and 0 <= (dt.datetime.fromisoformat(t) - start).total_seconds() / 3600 <= GAME_HOURS]
        if vals:
            out[g["game_id"]] = float(np.mean(vals))
    return out


def upcoming_rows(games: pl.DataFrame) -> pl.DataFrame:
    """One stat-less row per skill player whose latest game this season was for a team in `games`."""
    season = games["season"].max()
    ps = pl.read_parquet(DATA / "raw" / "player_stats.parquet").filter(
        (pl.col("season") == season) & pl.col("position").is_in(props.SKILL))
    last = (ps.join(pl.read_parquet(DATA / "raw" / "schedules.parquet").select("game_id", "gameday"), on="game_id")
            .sort("gameday").group_by("player_id").last()
            .select("player_id", "player_name", "player_display_name", "position", "position_group", "team"))
    sides = pl.concat([
        games.select("game_id", "season", "week", "game_type", pl.col("home_team").alias("team"),
                     pl.col("away_team").alias("opponent_team")),
        games.select("game_id", "season", "week", "game_type", pl.col("away_team").alias("team"),
                     pl.col("home_team").alias("opponent_team"))])
    return (last.join(sides, on="team").rename({"game_type": "season_type"})
            .with_columns(pl.col("season", "week").cast(pl.Int32)))


def qb_changes(snap: pl.DataFrame, up: pl.DataFrame):
    """Warn when the QB the books list for passing yards isn't the team's QB in its last game.

    The model has no feature for a QB change, so that team's receiving projections assume the
    old QB's passing volume and efficiency."""
    season = up["season"].max()
    ps = pl.read_parquet(DATA / "raw" / "player_stats.parquet").filter(
        (pl.col("season") == season) & (pl.col("position") == "QB"))
    last = (ps.with_columns(pl.col("team").replace(props.RENAMES))
            .filter(pl.col("week") == pl.col("week").max().over("team"))
            .sort("attempts", descending=True).group_by("team").first()
            .select("team", pl.col("player_display_name").alias("last_qb")))
    keys = up.select("team", pl.col("player_display_name").map_elements(norm_name, return_dtype=pl.String)
                     .alias("key"))
    listed = (snap.filter(pl.col("mkt") == "pass_yds").select("key", "player").unique()
              .join(keys, on="key").join(last, on="team"))
    for r in listed.filter(pl.col("key") != pl.col("last_qb").map_elements(norm_name, return_dtype=pl.String)) \
            .iter_rows(named=True):
        print(f"  QB change? books list {r['player']} for {r['team']} (last game: {r['last_qb']}); "
              f"{r['team']} receiving projections don't know it")


def project(df: pl.DataFrame, game_ids: list[str]) -> pl.DataFrame:
    """Fit each market on all completed games and predict the upcoming rows."""
    df = props.add_flags(df)
    out = []
    for name, (target, _, _, feats) in props.MARKETS.items():
        m = props.eligible(df, name)
        cols = feats + props.COMMON
        tr = m.filter(pl.col(target).is_not_null() & ~pl.col("game_id").is_in(game_ids))
        te = m.filter(pl.col("game_id").is_in(game_ids))
        model = props.new_model().fit(tr.select(cols).to_numpy(), tr[target].to_numpy())
        out.append(te.select("game_id", "player_id", "player_display_name", "team", "opponent_team", "position",
                             pl.col(f"e_{target}").alias("recent_avg"), "wind", "team_spread",
                             # starters' snap share out at the player's own position (vacated volume)
                             # and on the opposing unit he attacks (DBs for passing game, front seven for rushing)
                             pl.when(pl.col("position") == "WR").then(pl.col("team_inj_wr"))
                             .when(pl.col("position") == "TE").then(pl.col("team_inj_te"))
                             .when(pl.col("position") == "RB").then(pl.col("team_inj_rb"))
                             .otherwise(pl.col("team_inj_wr")).round(2).alias("mates_out"),
                             pl.when(name == "rush_yds").then(pl.col("opp_inj_front"))
                             .otherwise(pl.col("opp_inj_db")).round(2).alias("opp_out"),
                             pl.when(pl.col("own_out") == 1).then(pl.lit("OUT/D"))
                             .when(pl.col("own_q") == 1).then(pl.lit("Q"))
                             .when(pl.col("own_dnp") == 1).then(pl.lit("ltd"))
                             .otherwise(pl.lit("")).alias("inj"))
                   .with_columns(pl.lit(name).alias("mkt"),
                                 pl.Series("pred", model.predict(te.select(cols).to_numpy())),
                                 pl.col("player_display_name").map_elements(norm_name, return_dtype=pl.String)
                                 .alias("key")))
    return pl.concat(out)


def p_over(hist: dict, mkt: str, pred: float, line: float) -> float:
    """Share of nearby out-of-sample predictions whose actual beat the line, scaled by pred."""
    p, ratio = hist[mkt]
    idx = np.argsort(np.abs(np.log(np.maximum(p, 0.5)) - np.log(max(pred, 0.5))))[:NEIGHBORS]
    return float((ratio[idx] * pred > line).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", nargs="?", help="props_log CSV (default: newest)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    path = Path(args.snapshot) if args.snapshot else max((ROOT / "props_log").glob("*.csv"))
    snap = load_snapshot(path)
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
             .with_columns(pl.col("home_team", "away_team").replace(props.RENAMES)))
    games = upcoming_games(snap, sched)
    print(f"{path.name}: {games.height} unplayed games, season {games['season'].max()} week(s) "
          f"{sorted(games['week'].unique().to_list())}")
    wind = forecast_wind(games)
    print("forecast wind (mph): " + ", ".join(f"{g.split('_', 2)[2]} {w:.0f}" for g, w in sorted(wind.items())))

    up = upcoming_rows(games)
    qb_changes(snap, up)
    preds = project(props.build(up, wind), games["game_id"].to_list())

    hist = {}
    for name, (target, *_) in props.MARKETS.items():
        h = pl.read_parquet(DATA / f"props_{name}_preds.parquet").filter(pl.col("pred") > 0)
        hist[name] = (h["pred"].to_numpy(), (h[target] / h["pred"]).to_numpy())

    # pair each book's over and under, attach the projection
    snap = snap.with_columns(pl.col("home"), pl.col("away"))
    book = (snap.pivot(on="side", index=["home", "away", "book", "mkt", "key", "player", "point"],
                       values="price", aggregate_function="first")
            .rename({"Over": "over_price", "Under": "under_price"}))
    # main line only: Bovada etc. also list alternate ladders; keep each book's point priced closest
    # to even on both sides, where the model's distribution is most trustworthy
    book = (book.filter(pl.col("over_price").is_not_null() & pl.col("under_price").is_not_null())
            .with_columns((pl.col("over_price") - pl.col("under_price")).abs().alias("_skew"))
            .sort("_skew").group_by("home", "away", "book", "mkt", "key", maintain_order=True).first()
            .drop("_skew"))
    g2 = games.select("game_id", pl.col("home_team").alias("home"), pl.col("away_team").alias("away"))
    book = book.join(g2, on=["home", "away"])
    matched = book.join(preds, on=["game_id", "mkt", "key"], how="inner")
    out_players = matched.filter(pl.col("inj") == "OUT/D")["player"].unique().to_list()
    if out_players:
        print("  listed Out/Doubtful, dropped (a bet on them is usually voided): " + ", ".join(sorted(out_players)))
        matched = matched.filter(pl.col("inj") != "OUT/D")
    lines = book.select("mkt", "key", "game_id").unique()
    n_match = matched.select("mkt", "key", "game_id").unique().height
    print(f"matched {n_match} of {lines.height} player-markets to a projection")
    miss = lines.join(preds, on=["game_id", "mkt", "key"], how="anti")
    if miss.height:
        print("  unmatched (no projection: too few games/usage, or a name mismatch): "
              + ", ".join(sorted(set(miss["key"].to_list()))[:40]))

    cons_line = matched.group_by("game_id", "mkt", "key").agg(pl.col("point").median().alias("cons_line"))
    matched = matched.join(cons_line, on=["game_id", "mkt", "key"]).with_columns(
        (pl.col("cons_line") + pl.col("mkt").replace_strict(BLEND, return_dtype=pl.Float64)
         * (pl.col("pred") - pl.col("cons_line"))).alias("pred_blend"))
    rows = []
    for r in matched.iter_rows(named=True):
        po = p_over(hist, r["mkt"], r["pred_blend"], r["point"])
        fair = None
        if r["over_price"] is not None and r["under_price"] is not None:
            io, iu = 1 / (1 + payout(r["over_price"])), 1 / (1 + payout(r["under_price"]))
            fair = io / (io + iu)
        for side, price, p in (("Over", r["over_price"], po), ("Under", r["under_price"], 1 - po)):
            if price is None:
                continue
            rows.append({**{k: r[k] for k in ("game_id", "player_id", "player", "team", "opponent_team", "mkt", "book",
                                              "point", "pred", "pred_blend", "recent_avg", "team_spread", "inj",
                                              "mates_out", "opp_out")},
                         "side": side, "price": int(price), "p_model": p,
                         "p_book_fair": None if fair is None else (fair if side == "Over" else 1 - fair),
                         "ev": p * payout(price) - (1 - p)})
    bets = pl.DataFrame(rows)
    cons = (matched.group_by("game_id", "mkt", "key")
            .agg(pl.col("player").first(), pl.col("point").median().alias("consensus_line"),
                 pl.col("pred").first(), pl.col("pred_blend").first(), pl.col("recent_avg").first(), pl.len().alias("books")))
    best = (bets.sort("ev", descending=True).group_by("player", "mkt", maintain_order=True).first()
            .join(cons.select("player", "mkt", "consensus_line", "books"), on=["player", "mkt"])
            .with_columns(((pl.col("consensus_line") / pl.col("recent_avg") - 1).abs() > NEWS_GAP)
                          .any().over("player").alias("news"))  # news about a player hits all his markets
            .sort("ev", descending=True))

    stamp = path.stem
    bets = bets.join(best.select("player", "mkt", "news"), on=["player", "mkt"], how="left")
    bets.write_csv(DATA / f"props_edges_{stamp}_all.csv")
    # committed copy, so grade_props.py can score what the model said before kickoff
    proj = ROOT / "props_log" / "projections"
    proj.mkdir(exist_ok=True)
    bets.write_csv(proj / f"{stamp}.csv")
    best.write_csv(DATA / f"props_edges_{stamp}.csv")
    with pl.Config(tbl_rows=args.top, tbl_cols=20, tbl_width_chars=200, float_precision=2,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(f"\nTop {args.top} by model EV (best book per player-market):")
        print(best.head(args.top).select("player", "team", "opponent_team", "team_spread", "inj", "mkt", "side", "book",
                                         "point", "price", "consensus_line", "pred", "pred_blend", "recent_avg", "p_model",
                                         "p_book_fair", "ev", "news"))
        clean = best.filter(~pl.col("news"))
        print(f"\nTop {args.top} without a news flag ({best.height - clean.height} flagged rows hidden):")
        print(clean.head(args.top).select("player", "team", "team_spread", "inj", "mates_out", "opp_out", "mkt", "side",
                                          "book", "point", "price", "consensus_line", "pred", "pred_blend", "recent_avg", "p_model",
                                          "p_book_fair", "ev"))
        boost = best.filter(pl.col("mates_out") >= 0.6).sort("mates_out", descending=True)
        print(f"\nInjury upgrades: a starter (>= 0.6 snap share) out at the player's own position, "
              f"so the model expects more volume ({boost.select('player').n_unique()} players):")
        print(boost.select("player", "team", "mates_out", "mkt", "side", "point", "pred", "recent_avg", "p_model", "ev", "news"))
        gap = cons.with_columns(((pl.col("pred") - pl.col("consensus_line")) / pl.col("consensus_line"))
                                .alias("gap_pct")).sort("gap_pct")
        print("\nLargest model-vs-consensus gaps (model below the line):")
        print(gap.head(10).select("player", "mkt", "consensus_line", "pred", "recent_avg", "books", "gap_pct"))
        print("Largest model-vs-consensus gaps (model above the line):")
        print(gap.tail(10).reverse().select("player", "mkt", "consensus_line", "pred", "recent_avg", "books", "gap_pct"))
        s = bets.group_by("mkt", "side").agg(pl.col("p_model").mean(), pl.col("p_book_fair").mean(),
                                             (pl.col("ev") > 0).mean().alias("share_pos_ev")).sort("mkt", "side")
        print("\nModel vs book, averaged over every offered price (a well-calibrated model should be near the book):")
        print(s)
    print(f"\nwrote data/props_edges_{stamp}.csv (best per player-market) and _all.csv (every book/side);"
          f" props_log/projections/{stamp}.csv (commit it before kickoff)")


if __name__ == "__main__":
    main()
