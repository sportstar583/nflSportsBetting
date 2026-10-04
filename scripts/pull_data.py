"""Pull nflverse data and cache it as parquet under data/raw/."""
import sys
from pathlib import Path

import nflreadpy as nfl

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
START, END = 2018, 2026
SEASONS = list(range(START, END + 1))

DATASETS = {
    "schedules": lambda: nfl.load_schedules(SEASONS),
    "pbp": lambda: nfl.load_pbp(SEASONS),
    "team_stats": lambda: nfl.load_team_stats(SEASONS),
    "player_stats": lambda: nfl.load_player_stats(SEASONS),
    "injuries": lambda: nfl.load_injuries(SEASONS),
    "rosters": lambda: nfl.load_rosters(SEASONS),
    "snap_counts": lambda: nfl.load_snap_counts(SEASONS),
    "players": lambda: nfl.load_players(),
    # richer per-game stats for the prop model (NGS/PFR start 2018/2019; ff_opportunity 2006)
    "ngs_receiving": lambda: nfl.load_nextgen_stats(stat_type="receiving", seasons=SEASONS),
    "ngs_rushing": lambda: nfl.load_nextgen_stats(stat_type="rushing", seasons=SEASONS),
    "ngs_passing": lambda: nfl.load_nextgen_stats(stat_type="passing", seasons=SEASONS),
    "pfr_rec": lambda: nfl.load_pfr_advstats(seasons=SEASONS, stat_type="rec", summary_level="week"),
    "pfr_rush": lambda: nfl.load_pfr_advstats(seasons=SEASONS, stat_type="rush", summary_level="week"),
    "pfr_pass": lambda: nfl.load_pfr_advstats(seasons=SEASONS, stat_type="pass", summary_level="week"),
    "ff_opportunity": lambda: nfl.load_ff_opportunity(SEASONS),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]  # optional dataset names, e.g. `pull_data.py ngs_receiving`
    for name, fn in DATASETS.items():
        if only and name not in only:
            continue
        try:
            df = fn()
            df.write_parquet(OUT / f"{name}.parquet")
            print(f"{name}: {df.height} rows, {df.width} cols")
        except Exception as e:
            print(f"{name}: FAILED ({e})", file=sys.stderr)


if __name__ == "__main__":
    main()
