"""Team ratings from Madden player ratings (data/raw/madden_ratings.parquet, from madden_pull.py).

Each Madden player is matched to his nflverse id by birthdate and name. For every team-week the unit
ratings are the mean overall of the best healthy players at each unit on that week's active roster
(Out/Doubtful, IR and the manual out list removed), with UNIT_SIZE players per unit:
QB 1, OL 5, WR 3, TE 1, RB 1, DL 4, LB 3, DB 4. A rostered player Madden doesn't have (a late
signing, a rookie in a season Madden hasn't rated) counts as UNRATED.

Ratings in effect for a game: 2025 week W uses Madden 26's "Week W" update (the round's update in
the playoffs). 2026 games use Madden 26's last update, the latest ratings
EA's API serves, with players on their 2026 teams.

    python scripts/madden.py      # team unit ratings -> data/madden_team_ratings.parquet
"""
import datetime as dt
from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data"
RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
UNIT_SIZE = {"QB": 1, "OL": 5, "WR": 3, "TE": 1, "RB": 1, "DL": 4, "LB": 3, "DB": 4}
UNRATED = 60


def _date(s: str) -> dt.date | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%y"):
        try:
            d = dt.datetime.strptime(s, fmt).date()
            return d.replace(year=d.year - 100) if d.year > 2015 else d  # '%y' maps 98 -> 1998, 05 -> 2005
        except (TypeError, ValueError):
            continue
    return None


def norm(c: str) -> pl.Expr:
    return (pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z]", "")
            .str.replace_all(r"(jr|sr|ii|iii|iv)$", ""))


def ratings() -> pl.DataFrame:
    """Madden ratings with nflverse gsis ids (matched on birthdate + last name, else birthdate +
    first name)."""
    m = pl.read_parquet(DATA / "raw" / "madden_ratings.parquet")
    m = m.with_columns(pl.col("birthdate").map_elements(_date, return_dtype=pl.Date).alias("bdate"))
    p = (pl.read_parquet(DATA / "raw" / "players.parquet")
         .select("gsis_id", "first_name", "last_name", "football_name",
                 pl.col("birth_date").str.to_date(strict=False).alias("bdate"))
         .drop_nulls(["gsis_id", "bdate"]))
    ids = m.select("ea_id", "first_name", "last_name", "bdate").unique("ea_id")
    by_last = (ids.with_columns(norm("last_name").alias("k")).join(
        p.with_columns(norm("last_name").alias("k")).select("bdate", "k", "gsis_id"), on=["bdate", "k"], how="inner")
        .unique("ea_id", keep="none"))
    rest = ids.join(by_last.select("ea_id"), on="ea_id", how="anti")
    by_first = (rest.with_columns(norm("first_name").alias("k")).join(
        pl.concat([p.select("bdate", norm("first_name").alias("k"), "gsis_id"),
                   p.select("bdate", norm("football_name").alias("k"), "gsis_id")]).drop_nulls().unique(),
        on=["bdate", "k"], how="inner").unique("ea_id", keep="none"))
    link = pl.concat([by_last.select("ea_id", "gsis_id"), by_first.select("ea_id", "gsis_id")])
    return m.join(link, on="ea_id", how="left")


def team_ratings(seasons=(2025, 2026)) -> pl.DataFrame:
    """Per (season, week, team): mean overall of the top healthy players per unit, plus how many
    of them Madden had no rating for."""
    import props
    r = ratings().drop_nulls("gsis_id")
    last_order = r["iter_order"].max()
    ros = (pl.read_parquet(DATA / "raw" / "rosters_weekly.parquet")
           .filter(pl.col("season").is_in(list(seasons)) & (pl.col("status") == "ACT") & pl.col("position").is_in(list(UNIT_SIZE)))
           .select(pl.col("season", "week").cast(pl.Int32), pl.col("team").replace(RENAMES), "gsis_id", "position")
           .drop_nulls("gsis_id").unique(["season", "week", "team", "gsis_id"]))
    out = props.players_out().rename({"player_id": "gsis_id"})
    ros = ros.join(out, on=["season", "week", "team", "gsis_id"], how="anti")
    # the update in effect: "Week W" is update W+1 (base = 1; wild card = 20 ... Super Bowl = 23), so a
    # 2025 game in week W uses update W+1; 2026 games use the last one. A player missing from that
    # update (or an update that doesn't exist) takes his latest earlier rating, else UNRATED.
    ros = ros.with_columns(pl.when(pl.col("season") >= 2026).then(pl.lit(last_order))
                           .otherwise(pl.col("week") + 1).cast(pl.Int64).alias("iter_order")).sort("iter_order")
    rr = r.select("gsis_id", pl.col("iter_order").cast(pl.Int64), "overall").sort("iter_order")
    ros = ros.join_asof(rr, on="iter_order", by="gsis_id", strategy="backward")
    ros = ros.with_columns(pl.col("overall").is_null().alias("unrated"), pl.col("overall").fill_null(UNRATED))
    ranked = ros.with_columns(pl.col("overall").rank("ordinal", descending=True).over("season", "week", "team", "position").alias("rk"))
    ranked = ranked.filter(pl.col("rk") <= pl.col("position").replace_strict(UNIT_SIZE, return_dtype=pl.Int64))
    units = (ranked.group_by("season", "week", "team", "position")
             .agg(pl.col("overall").mean().alias("rating"), pl.col("unrated").sum().alias("n_unrated"))
             .pivot(on="position", index=["season", "week", "team"], values=["rating", "n_unrated"]))
    units = units.rename({f"rating_{u}": f"mad_{u.lower()}" for u in UNIT_SIZE} | {f"n_unrated_{u}": f"unrated_{u.lower()}" for u in UNIT_SIZE})
    return units.sort("season", "week", "team")


if __name__ == "__main__":
    r = ratings()
    n = r.select("ea_id").n_unique()
    print(f"matched {r.drop_nulls('gsis_id').select('ea_id').n_unique()} of {n} Madden players to nflverse ids")
    t = team_ratings()
    t.write_parquet(DATA / "madden_team_ratings.parquet")
    print(t.filter((pl.col("season") == 2026) & (pl.col("week") == 5)).sort("mad_qb", descending=True).head(10))
