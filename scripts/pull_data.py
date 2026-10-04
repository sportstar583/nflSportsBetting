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
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in DATASETS.items():
        try:
            df = fn()
            df.write_parquet(OUT / f"{name}.parquet")
            print(f"{name}: {df.height} rows, {df.width} cols")
        except Exception as e:
            print(f"{name}: FAILED ({e})", file=sys.stderr)


if __name__ == "__main__":
    main()
