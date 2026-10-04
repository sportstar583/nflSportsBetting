"""Forecast (not observed) kickoff weather for outdoor games, from Open-Meteo's forecast archive.

nflverse's wind/temp are what was measured at kickoff, which a bettor didn't know when betting.
This pulls the archived short-range forecast for each stadium so the wind rule can be tested
on information that was available. Writes data/raw/forecast_weather.parquet.
"""
from pathlib import Path

import datetime as dt
import time

import polars as pl
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
STADIUMS = {  # stadium_id -> (lat, lon)
    "NYC01": (40.8135, -74.0745), "PHI00": (39.9008, -75.1675), "BAL00": (39.2780, -76.6227),
    "TAM00": (27.9759, -82.5033), "GNB00": (44.5013, -88.0622), "SFO01": (37.4030, -121.9700),
    "BOS00": (42.0909, -71.2643), "CHI98": (41.8623, -87.6167), "MIA00": (25.9580, -80.2389),
    "NAS00": (36.1665, -86.7713), "CAR00": (35.2258, -80.8528), "BUF00": (42.7738, -78.7870),
    "DEN00": (39.7439, -105.0201), "WAS00": (38.9076, -76.8645), "CLE00": (41.5061, -81.6995),
    "JAX00": (30.3239, -81.6373), "SEA00": (47.5952, -122.3316), "KAN00": (39.0489, -94.4839),
    "PIT00": (40.4468, -80.0158), "CIN00": (39.0955, -84.5161), "ATL97": (33.7554, -84.4008),
    "LAX99": (34.0141, -118.2879), "LAX97": (33.8644, -118.2611), "IND00": (39.7601, -86.1639),
    "LON02": (51.6043, -0.0664), "LON00": (51.5560, -0.2796), "PHO00": (33.5276, -112.2626),
    "OAK00": (37.7516, -122.2005), "DAL00": (32.7473, -97.0945), "HOU00": (29.6847, -95.4107),
    "FRA00": (50.0686, 8.6455), "GER00": (48.2188, 11.6247), "MEX00": (19.3029, -99.1505),
    "SAO00": (-23.5453, -46.4742), "RIO00": (-22.9122, -43.2302),
}
GAME_HOURS = 3  # average forecast over kickoff hour and the next 3


CACHE = DATA / "raw" / "forecast_cache"


def fetch(lat, lon, start, end) -> pl.DataFrame | None:
    """Hourly forecast for one stadium-season, cached on disk; None if the API keeps failing."""
    path = CACHE / f"{lat}_{lon}_{start}_{end}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    params = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,temperature_2m",
              "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
              "timezone": "America/New_York",  # nflverse gametime is Eastern
              "start_date": str(start), "end_date": str(end)}
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 5:
                print(f"  failed {start}..{end}: {type(e).__name__}", flush=True)
                return None
            time.sleep(min(60, 5 * 2 ** attempt))
    h = r.json()["hourly"]
    df = pl.DataFrame({"time": h["time"], "wind": h["wind_speed_10m"], "temp": h["temperature_2m"]},
                      schema={"time": pl.Utf8, "wind": pl.Float64, "temp": pl.Float64}
                      ).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M"))
    CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    time.sleep(1)  # be polite to the free API
    return df


def main():
    sched = (pl.read_parquet(DATA / "raw" / "schedules.parquet")
             .filter(pl.col("roof").is_in(["outdoors", "open"]) & pl.col("gametime").is_not_null())
             .with_columns(pl.col("gameday").str.to_date().alias("date")))
    rows = []
    for sid, games in sched.group_by("stadium_id"):
        sid = sid[0]
        if sid not in STADIUMS:
            print(f"skip {sid}: no coordinates")
            continue
        lat, lon = STADIUMS[sid]
        games = games.filter(pl.col("date") < dt.date.today())
        if games.is_empty():
            continue
        parts = [fetch(lat, lon, g["date"].min(), g["date"].max()) for _, g in games.group_by("season")]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        hourly = pl.concat(parts)
        for gid, date, gt in games.select("game_id", "date", "gametime").iter_rows():
            hh = int(gt.split(":")[0])
            start = pl.datetime(date.year, date.month, date.day, hh)
            w = hourly.filter((pl.col("time") >= start)
                              & (pl.col("time") <= start + pl.duration(hours=GAME_HOURS)))
            rows.append({"game_id": gid, "fc_wind": w["wind"].mean(), "fc_temp": w["temp"].mean()})
        print(f"{sid}: {games.height} games", flush=True)
    out = pl.DataFrame(rows)
    out.write_parquet(DATA / "raw" / "forecast_weather.parquet")
    print(f"forecast weather: {out.height} games, {out['fc_wind'].null_count()} missing")


if __name__ == "__main__":
    main()
