"""Does the kind of injury matter? Players who play through an injury, and players returning from one.

From the weekly injury reports (2018-), every skill-player game is put in one group:
  healthy   not on the week's report and didn't miss his team's previous game
  through   on the report (Questionable, or limited / no practice) with injury X, didn't miss
            the previous game: he played through it
  return    first game back after missing 1, or 2+, team games; X is the injury listed while out
The report only says "Ankle", not high vs low; a high-ankle sprain usually costs 2+ games, so
"ankle, return after 2+ missed" is the stand-in for it.

Per group and injury: the median ratio of the main stat (QB passing, RB rushing, WR/TE receiving
yards) to the player's pre-game average, the snap share ratio, and, for 2022-2025 where the
walk-forward model predictions exist, actual minus the model (does the model already price it?)
and how often the player went under the model.

    python scripts/injury_types.py       # writes props_log/injury_types.csv
"""
import re
from pathlib import Path

import numpy as np
import polars as pl

import props

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STAT = {"QB": ("passing_yards", "pass_yds"), "RB": ("rushing_yards", "rush_yds"),
        "WR": ("receiving_yards", "rec_yds"), "TE": ("receiving_yards", "rec_yds")}
MERGE = {"rib": "ribs", "quadricep": "quad", "quadriceps": "quad", "thigh": "quad", "hamstrings": "hamstring",
         "knees": "knee", "ankles": "ankle", "toe": "foot", "heel": "foot", "achilles": "achilles",
         "abdomen": "core", "oblique": "core", "pectoral": "chest", "thumb": "hand", "finger": "hand", "wrist": "hand"}


def body_part(s: str | None) -> str | None:
    if not s:
        return None
    s = s.lower()
    if "not injury" in s or "personal" in s or "rest" in s:
        return "not injury"
    s = re.sub(r"\b(left|right|l|r)\b", "", s).split(",")[0].split("/")[0].strip()
    return MERGE.get(s, s)


def tag(f: pl.DataFrame) -> pl.DataFrame:
    """Add missed (team games missed right before this one, same season), part_now (injury on this
    week's report) and part_out (injury listed while he was out) to a player-game frame."""
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet").filter(pl.col("game_type") == "REG")
             .with_columns(pl.col("home_team", "away_team").replace(props.RENAMES)))
    tg = (pl.concat([sched.select("game_id", "season", "week", pl.col("home_team").alias("team")),
                     sched.select("game_id", "season", "week", pl.col("away_team").alias("team"))])
          .sort("team", "season", "week").with_columns(pl.int_range(pl.len()).over("team", "season").alias("gidx")))

    inj = (pl.read_parquet(DATA / "raw" / "injuries.parquet")
           .filter(pl.col("gsis_id").is_not_null())
           .with_columns(pl.col("team").replace(props.RENAMES), pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
                         pl.coalesce("report_primary_injury", "practice_primary_injury")
                         .map_elements(body_part, return_dtype=pl.String).alias("part"),
                         (pl.col("report_status").is_in(["Questionable", "Doubtful", "Out"])
                          | pl.col("practice_status").str.contains("Did Not|Limited").fill_null(False)).alias("listed"))
           .filter(pl.col("listed") & pl.col("part").is_not_null())
           .unique(["season", "week", "gsis_id"], keep="last")
           .select("season", "week", pl.col("gsis_id").alias("player_id"), "part"))

    f = f.with_columns(pl.col("season", "week").cast(pl.Int32))
    f = (f.join(tg.select("game_id", "team", "gidx"), on=["game_id", "team"], how="left")
         .sort("player_id", "season", "gidx")
         .with_columns((pl.col("gidx") - pl.col("gidx").shift(1).over("player_id", "season") - 1)
                       .fill_null(0).clip(0).alias("missed")))

    # injury on this week's report (playing through), and the injury listed in the weeks he missed
    f = f.join(inj.rename({"part": "part_now"}), on=["season", "week", "player_id"], how="left")
    away = (f.filter(pl.col("missed") > 0)
            .select("player_id", "season", "week", "team", "gidx", "missed")
            .join(tg.select("team", "season", pl.col("week").alias("w_out"), pl.col("gidx").alias("g_out")), on=["team", "season"])
            .filter((pl.col("g_out") < pl.col("gidx")) & (pl.col("g_out") >= pl.col("gidx") - pl.col("missed")))
            .join(inj, left_on=["season", "w_out", "player_id"], right_on=["season", "week", "player_id"])
            .group_by("player_id", "season", "week").agg(pl.col("part").mode().first().alias("part_out")))
    return f.join(away, on=["player_id", "season", "week"], how="left")


def main():
    f = (pl.read_parquet(DATA / "props_features.parquet")
         .filter(pl.col("position").is_in(list(STAT)) & (pl.col("n_prior") >= 3) & (pl.col("season") >= 2018)))
    stat = pl.coalesce([pl.when(pl.col("position") == p).then(pl.col(c)) for p, (c, _) in STAT.items()])
    estat = pl.coalesce([pl.when(pl.col("position") == p).then(pl.col(f"e_{c}")) for p, (c, _) in STAT.items()])
    mkt = pl.coalesce([pl.when(pl.col("position") == p).then(pl.lit(m)) for p, (_, m) in STAT.items()])
    f = (tag(f).with_columns(stat.alias("y"), estat.alias("e_y"), mkt.alias("mkt")).filter(pl.col("e_y") > 5))
    f = f.with_columns(
        pl.when(pl.col("missed") >= 1).then(pl.when(pl.col("missed") >= 2).then(pl.lit("return after 2+ missed"))
                                            .otherwise(pl.lit("return after 1 missed")))
        .when(pl.col("part_now").is_not_null()).then(pl.lit("playing through"))
        .otherwise(pl.lit("healthy")).alias("group"),
        pl.when(pl.col("missed") >= 1).then(pl.col("part_out")).otherwise(pl.col("part_now")).alias("injury"),
        (pl.col("y") / pl.col("e_y")).alias("ratio"),
        (pl.col("offense_pct") / pl.col("e_offense_pct")).alias("snap_ratio"))

    preds = pl.concat([pl.read_parquet(DATA / f"props_{m}_preds.parquet")
                       .select("season", "week", "player_display_name", "team", "pred", pl.lit(m).alias("mkt"))
                       for m in ["pass_yds", "rush_yds", "rec_yds"]]).with_columns(pl.col("season", "week").cast(pl.Int32))
    f = f.join(preds, on=["season", "week", "player_display_name", "team", "mkt"], how="left")

    def summarize(d: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
        return (d.group_by(keys).agg(
            pl.len().alias("n"), pl.col("ratio").median().round(3).alias("yds_vs_avg"),
            pl.col("snap_ratio").median().round(3).alias("snaps_vs_avg"),
            pl.col("pred").is_not_null().sum().alias("n_model"),
            (pl.col("y") - pl.col("pred")).median().round(1).alias("vs_model_med"),
            ((pl.col("y") - pl.col("pred")) / pl.col("pred").clip(5)).median().round(3).alias("vs_model_pct"),
            (pl.col("y") < pl.col("pred")).filter(pl.col("pred").is_not_null()).mean().round(3).alias("under_model"))
            .sort(keys))

    with pl.Config(tbl_rows=200, tbl_width_chars=200, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print("All skill players, by group:")
        print(summarize(f, ["group"]))
        top = (f.filter(pl.col("group") != "healthy").group_by("injury").len().filter(pl.col("len") >= 60)["injury"])
        out = summarize(f.filter(pl.col("injury").is_in(top.implode()) & (pl.col("group") != "healthy")), ["group", "injury"])
        out = out.filter(pl.col("n") >= 25)
        for g in ["playing through", "return after 1 missed", "return after 2+ missed"]:
            print(f"\n{g}, by injury (n >= 25):")
            print(out.filter(pl.col("group") == g).sort("yds_vs_avg").drop("group"))
        print("\nBy position, playing through / returning 2+, key injuries:")
        key = ["ankle", "hamstring", "knee", "shoulder", "concussion", "groin", "foot", "calf"]
        pos = summarize(f.filter(pl.col("injury").is_in(key) & (pl.col("group") != "healthy"))
                        .with_columns(pl.when(pl.col("position") == "TE").then(pl.lit("WR")).otherwise(pl.col("position")).alias("pos")),
                        ["pos", "group", "injury"]).filter(pl.col("n") >= 20)
        print(pos.select("pos", "group", "injury", "n", "yds_vs_avg", "snaps_vs_avg", "n_model", "vs_model_pct", "under_model"))
    out.write_csv(ROOT / "props_log" / "injury_types.csv")
    pos.write_csv(ROOT / "props_log" / "injury_types_by_position.csv")
    print("\nwrote props_log/injury_types.csv and injury_types_by_position.csv")


if __name__ == "__main__":
    main()
