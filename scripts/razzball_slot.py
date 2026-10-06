"""Snapshot Razzball's 'Defensive Slot vs. Wide PPG Allowed' table into props_log/razzball/.

Fantasy points per game each defense has allowed to WRs lined up in the slot vs out wide, this
season to date. Razzball shows only the current season (no history, no week selector), so the
table can't be backtested yet; saving it every week builds that history.

    python scripts/razzball_slot.py
"""
import datetime as dt
import re
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://football.razzball.com/defensive-slot-vs-wide-ppg-allowed/"
TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX", "LAR": "LA"}  # Razzball -> nflverse codes


def fetch() -> pl.DataFrame:
    html = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).text
    tbl = html[html.find('id="neorazzstatstable"'):]
    tbl = tbl[:tbl.find("</table>")]
    rows = [[re.sub("<[^>]+>", "", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S | re.I)]
            for r in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.S | re.I)]
    hdr, data = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
    df = pl.DataFrame(data, schema=hdr, orient="row").drop("#")
    pct = lambda c: pl.col(c).str.strip_chars("%").cast(pl.Float64) / 100
    return df.select(pl.col("Defense").replace(TEAM).alias("defense"),
                     pl.col("Total PPG Allowed").cast(pl.Float64).alias("wr_ppg_allowed"),
                     pl.col("Slot PPG Allowed").cast(pl.Float64).alias("slot_ppg_allowed"),
                     pl.col("Wide PPG Allowed").cast(pl.Float64).alias("wide_ppg_allowed"),
                     pct("Slot%").alias("slot_share"), pct("Wide%").alias("wide_share"),
                     pl.col("Next Opp").replace(TEAM).alias("next_opp"))


def main():
    df = fetch()
    out = ROOT / "props_log" / "razzball"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"slot_wide_{dt.date.today().isoformat()}.csv"
    df.write_csv(path)
    print(f"wrote {path.relative_to(ROOT)} ({df.height} defenses)")
    with pl.Config(tbl_rows=40, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=1):
        print(df.sort("slot_ppg_allowed", descending=True))


if __name__ == "__main__":
    main()
