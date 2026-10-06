"""Snapshot Razzball's slot vs. wide tables into props_log/razzball/ and score next week's WR matchups.

- Defenses: fantasy PPG allowed to WRs in the slot vs out wide (season to date).
- Receivers: fantasy PPG scored from the slot vs out wide, and the share of points from each.
Matchup score for a receiver = slot_share x (opponent slot PPG allowed / league avg)
                              + wide_share x (opponent wide PPG allowed / league avg); 1.00 = neutral.
Razzball shows only the current season (no history), so this can't be backtested yet; saving the
snapshots weekly builds that history. Not a model feature.

    python scripts/razzball_slot.py            # next week = first week with unplayed games
    python scripts/razzball_slot.py 6          # score a specific week
"""
import datetime as dt
import html as html_lib
import re
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://football.razzball.com/defensive-slot-vs-wide-ppg-allowed/"
WR_URL = "https://football.razzball.com/wide-receiver-fantasy-points-scored-slot-vs-wide/"
TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX", "LAR": "LA"}  # Razzball -> nflverse codes


def table(url: str) -> pl.DataFrame:
    html = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).text
    tbl = html[html.find('id="neorazzstatstable"'):]
    tbl = tbl[:tbl.find("</table>")]
    rows = [[re.sub("<[^>]+>", "", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S | re.I)]
            for r in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S | re.I)]
    hdr, data = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
    return pl.DataFrame(data, schema=hdr, orient="row").drop("#")


pct = lambda c: pl.col(c).str.strip_chars("%").cast(pl.Float64) / 100


def fetch() -> pl.DataFrame:
    df = table(URL)
    return df.select(pl.col("Defense").replace(TEAM).alias("defense"),
                     pl.col("Total PPG Allowed").cast(pl.Float64).alias("wr_ppg_allowed"),
                     pl.col("Slot PPG Allowed").cast(pl.Float64).alias("slot_ppg_allowed"),
                     pl.col("Wide PPG Allowed").cast(pl.Float64).alias("wide_ppg_allowed"),
                     pct("Slot%").alias("slot_share"), pct("Wide%").alias("wide_share"),
                     pl.col("Next Opp").replace(TEAM).alias("next_opp"))


def fetch_wr() -> pl.DataFrame:
    df = table(WR_URL)
    return df.select(pl.col("Player").map_elements(html_lib.unescape, return_dtype=pl.String).alias("player"), pl.col("Team").replace(TEAM).alias("team"),
                     pl.col("Games").cast(pl.Int32).alias("games"), pl.col("Total Points").cast(pl.Float64).alias("points"),
                     pl.col("Slot PPG").cast(pl.Float64).alias("slot_ppg"), pct("Slot%").alias("slot_share"),
                     pl.col("Wide PPG").cast(pl.Float64).alias("wide_ppg"), pct("Wide%").alias("wide_share"))


def matchups(d: pl.DataFrame, w: pl.DataFrame, week: int | None) -> pl.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import props
    sched = (pl.read_parquet(ROOT / "data" / "raw" / "schedules.parquet")
             .with_columns(pl.col("home_team", "away_team").replace(props.RENAMES)))
    season = sched["season"].max()
    s = sched.filter(pl.col("season") == season)
    # default: the first week with no completed game (the next full slate)
    done = s.filter(pl.col("result").is_not_null())["week"].unique().to_list()
    week = week or s.filter(~pl.col("week").is_in(done))["week"].min()
    g = s.filter(pl.col("week") == week)
    opp = pl.concat([g.select(pl.col("home_team").alias("team"), pl.col("away_team").alias("opp")),
                     g.select(pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"))])
    avg = d.filter(pl.col("defense") == "NFL Average").row(0, named=True)
    dd = d.filter(pl.col("defense") != "NFL Average").select(
        pl.col("defense").alias("opp"), (pl.col("slot_ppg_allowed") / avg["slot_ppg_allowed"]).alias("opp_slot_idx"),
        (pl.col("wide_ppg_allowed") / avg["wide_ppg_allowed"]).alias("opp_wide_idx"),
        "slot_ppg_allowed", "wide_ppg_allowed")
    # quality of WRs faced: for each defense, the slot / wide fantasy PPG its past opponents' WRs score
    # (season to date, from the receiver table). Allowed / expected = adjusted index. The opponents'
    # season totals include their game against this defense, so the adjustment is slightly conservative.
    team_out = (w.with_columns((pl.col("slot_ppg") * pl.col("games")).alias("slot_pts"),
                               (pl.col("wide_ppg") * pl.col("games")).alias("wide_pts"))
                .group_by("team").agg(pl.col("slot_pts").sum(), pl.col("wide_pts").sum()))
    covered = int(w["games"].max())  # Razzball's season-to-date window (weeks 1..covered, no byes yet)
    played = s.filter(pl.col("result").is_not_null() & (pl.col("week") <= covered))
    tg = played.group_by("home_team").len().rename({"home_team": "team"}).vstack(
        played.group_by("away_team").len().rename({"away_team": "team"})).group_by("team").agg(pl.col("len").sum().alias("tg"))
    team_out = team_out.join(tg, on="team").with_columns((pl.col("slot_pts") / pl.col("tg")).alias("team_slot_pg"),
                                                         (pl.col("wide_pts") / pl.col("tg")).alias("team_wide_pg"))
    faced = pl.concat([played.select(pl.col("home_team").alias("defense"), pl.col("away_team").alias("faced")),
                       played.select(pl.col("away_team").alias("defense"), pl.col("home_team").alias("faced"))])
    exp = (faced.join(team_out.rename({"team": "faced"}), on="faced")
           .group_by("defense").agg(pl.col("team_slot_pg").mean().alias("exp_slot"), pl.col("team_wide_pg").mean().alias("exp_wide")))
    dd = (dd.join(exp.rename({"defense": "opp"}), on="opp", how="left")
          .with_columns((pl.col("slot_ppg_allowed") / pl.col("exp_slot")).alias("opp_slot_idx_adj"),
                        (pl.col("wide_ppg_allowed") / pl.col("exp_wide")).alias("opp_wide_idx_adj"))
          # the receiver table omits low-volume WRs, so 'expected' runs low: rescale to a league mean of 1
          .with_columns(pl.col("opp_slot_idx_adj") / pl.col("opp_slot_idx_adj").mean(),
                        pl.col("opp_wide_idx_adj") / pl.col("opp_wide_idx_adj").mean()))
    return (w.join(opp, on="team").join(dd, on="opp")
            .with_columns((pl.col("slot_share") * pl.col("opp_slot_idx") + pl.col("wide_share") * pl.col("opp_wide_idx"))
                          .alias("matchup"),
                          (pl.col("slot_share") * pl.col("opp_slot_idx_adj") + pl.col("wide_share") * pl.col("opp_wide_idx_adj"))
                          .alias("matchup_adj"), pl.lit(week).alias("week"))
            .sort("matchup_adj", descending=True))


def main():
    import sys
    week = int(sys.argv[1]) if len(sys.argv) > 1 else None
    d, w = fetch(), fetch_wr()
    out = ROOT / "props_log" / "razzball"
    out.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().isoformat()
    d.write_csv(out / f"slot_wide_{day}.csv")
    w.write_csv(out / f"wr_slot_wide_{day}.csv")
    m = matchups(d, w, week)
    m.write_csv(out / f"wr_matchups_wk{m['week'][0]}_{day}.csv", float_precision=3)
    print(f"saved {d.height} defenses, {w.height} receivers, {m.height} week-{m['week'][0]} matchups to props_log/razzball/")
    cols = ["player", "team", "opp", "points", "slot_share", "opp_slot_idx", "opp_slot_idx_adj", "opp_wide_idx_adj", "matchup", "matchup_adj"]
    with pl.Config(tbl_rows=15, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=2):
        top = m.filter(pl.col("points") >= 15)
        print("Best matchups (receivers with 15+ points so far):"); print(top.head(15).select(cols))
        print("Worst matchups:"); print(top.tail(15).reverse().select(cols))


if __name__ == "__main__":
    main()
