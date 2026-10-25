"""Bulk daily weather from Open-Meteo's ERA5 copy on AWS Open Data (no API limits).

Reads hourly point series straight from s3://openmeteo with python-omfiles and
aggregates them to local days. Temperature and dew point come from ERA5-Land
(0.1°, land only, much better for coastal and mountain cities); rain, snow,
cloud and wind come from ERA5 (0.25°). Cities with no ERA5-Land cell nearby
(tiny islands) use ERA5 for everything.

Writes one CSV per city to data/daily/. Precipitation and snowfall are written raw
(precipitation_era5, snowfall_era5); run rainfall.py afterwards to bias-correct them.

Usage: .venv/bin/python era5.py      # fetches every city without a daily CSV
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import fsspec
import numpy as np
import omfiles
import pandas as pd

import climate

BUCKET = "openmeteo/data"
CHUNK_HOURS = 504          # rolling chunk length for files after the yearly archives
LAPSE_RATE = 0.0065        # °C per metre, for grid-vs-city elevation differences
SNOW_CM_PER_MM = 0.7       # snow depth per mm water equivalent (Open-Meteo's factor)
WORKERS = 64
CELLS_PER_TASK = 24
CACHE_DIR = climate.DATA_DIR / "cache"  # hourly arrays; delete once data/daily/ is written

fs = fsspec.filesystem("s3", anon=True)


# ---------------------------------------------------------------- grids

@dataclass
class Grid:
    dataset: str
    step: float

    def __post_init__(self):
        self.land = self.static("soil_type") > 0
        self.hsurf = np.maximum(self.static("HSURF"), 0)
        self.nx = self.land.shape[1]

    def static(self, name: str) -> np.ndarray:
        return omfiles.OmFileReader.from_fsspec(fs, f"{BUCKET}/{self.dataset}/static/{name}.om")[...]

    def index(self, lat: float, lon: float) -> tuple[int, int]:
        return round((lat + 90) / self.step), round((lon + 180) / self.step) % self.nx

    def latlon(self, cell: tuple[int, int]) -> tuple[float, float]:
        return cell[0] * self.step - 90, cell[1] * self.step - 180

    def nearest_land(self, lat: float, lon: float, max_radius: int = 2) -> tuple[int, int] | None:
        """Nearest land cell within max_radius grid steps (ocean cells are not city weather)."""
        iy, ix = self.index(lat, lon)
        for radius in range(max_radius + 1):
            best = None
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = iy + dy, (ix + dx) % self.nx
                    if self.land[y, x]:
                        cy, cx = self.latlon((y, x))
                        d = (cy - lat) ** 2 + ((cx - lon) * np.cos(np.radians(lat))) ** 2
                        if best is None or d < best[0]:
                            best = (d, (y, x))
            if best:
                return best[1]
        return None


# ---------------------------------------------------------------- reading

def hour_of(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() // 3600)


def files_for(path: str, start_h: int, end_h: int) -> list[tuple[str, int]]:
    """(file, first hour) of every file covering [start_h, end_h): yearly archives, then 21-day chunks."""
    names = {p.rsplit("/", 1)[-1] for p in fs.ls(path, detail=False)}
    out, h = [], start_h
    while h < end_h:
        year = pd.Timestamp(h * 3600, unit="s").year
        if f"year_{year}.om" in names:
            out.append((f"{path}/year_{year}.om", hour_of(f"{year}-01-01")))
            h = hour_of(f"{year + 1}-01-01")
        else:
            n = h // CHUNK_HOURS
            if f"chunk_{n}.om" not in names:
                raise FileNotFoundError(f"{path}: no file for hour {h} (chunk_{n})")
            out.append((f"{path}/chunk_{n}.om", n * CHUNK_HOURS))
            h = (n + 1) * CHUNK_HOURS
    return out


def read_cells(path: str, cells: list[tuple[int, int]]) -> list[np.ndarray]:
    for attempt in range(5):
        try:
            reader = omfiles.OmFileReader.from_fsspec(fs, path)
            return [reader[y, x, :] for y, x in cells]
        except Exception as err:  # transient S3 errors
            if attempt == 4:
                raise
            print(f"  retry {path.rsplit('/', 2)[-2:]}: {err}", flush=True)
            time.sleep(5 * (attempt + 1))


def hourly_series(dataset: str, var: str, cells: list[tuple[int, int]],
                  start_h: int, end_h: int) -> np.ndarray:
    """Array [cell, hour] for hours [start_h, end_h), cached so an interrupted run can resume."""
    key = hashlib.sha1(repr((dataset, var, cells, start_h, end_h)).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"{dataset}_{var}_{key}.npy"
    if cache.exists():
        return np.load(cache)
    out = np.full((len(cells), end_h - start_h), np.nan, dtype=np.float32)
    files = files_for(f"{BUCKET}/{dataset}/{var}", start_h, end_h)
    batches = [list(range(i, min(i + CELLS_PER_TASK, len(cells))))
               for i in range(0, len(cells), CELLS_PER_TASK)]
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        jobs = {pool.submit(read_cells, path, [cells[i] for i in batch]): (first, batch)
                for path, first in files for batch in batches}
        for done, job in enumerate(as_completed(jobs), 1):
            first, batch = jobs[job]
            for i, series in zip(batch, job.result()):
                lo, hi = max(first, start_h), min(first + len(series), end_h)
                out[i, lo - start_h:hi - start_h] = series[lo - first:hi - first]
    print(f"  {dataset}/{var}: {len(files)} files x {len(cells)} cells in {time.time() - t0:.0f} s", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


# ---------------------------------------------------------------- daily aggregation

def to_local_days(hourly: np.ndarray, start_h: int, utc_offset: int, how: str) -> pd.Series:
    times = pd.to_datetime((np.arange(hourly.size) + start_h + utc_offset) * 3600, unit="s")
    s = pd.Series(hourly, index=times)
    days = s.groupby(s.index.floor("D"))
    full = days.count() == 24  # drop partial days at the edges of the range
    return getattr(days, how)()[full]


def fetch_missing(cities: pd.DataFrame, start: str = climate.START_DATE, end: str = climate.END_DATE) -> None:
    todo = cities[[not climate.daily_path(s).exists() for s in cities["slug"]]].reset_index(drop=True)
    if todo.empty:
        print("all cities already fetched")
        return

    era5 = Grid("copernicus_era5", 0.25)
    land = Grid("copernicus_era5_land", 0.1)
    cells = [era5.nearest_land(r.lat, r.lon) or era5.index(r.lat, r.lon) for r in todo.itertuples()]
    land_cells = [land.nearest_land(r.lat, r.lon) for r in todo.itertuples()]
    on_land = [i for i, c in enumerate(land_cells) if c is not None]
    offsets = [round(r.lon / 15) for r in todo.itertuples()]
    # Pad each side so every local day in [start, end] is complete.
    start_h, end_h = hour_of(start) - 24, hour_of(end) + 48
    print(f"{len(todo)} cities ({len(on_land)} on ERA5-Land), {start} to {end}", flush=True)

    def temperature_var(var: str) -> np.ndarray:
        """ERA5-Land where available, else ERA5; corrected to the city's elevation."""
        values = np.full((len(todo), end_h - start_h), np.nan, dtype=np.float32)
        grid_elev = np.array([era5.hsurf[c] for c in cells])
        if on_land:
            values[on_land] = hourly_series("copernicus_era5_land", var,
                                            [land_cells[i] for i in on_land], start_h, end_h)
            grid_elev[on_land] = [land.hsurf[land_cells[i]] for i in on_land]
        # Off-land cities, plus coastal cells the land mask calls land but ERA5-Land leaves empty.
        fallback = [i for i in range(len(todo)) if np.isnan(values[i]).all()]
        if fallback:
            values[fallback] = hourly_series("copernicus_era5", var,
                                             [cells[i] for i in fallback], start_h, end_h)
            grid_elev[fallback] = [era5.hsurf[cells[i]] for i in fallback]
        city_elev = todo["elevation"].fillna(pd.Series(grid_elev)).to_numpy()
        return values + ((grid_elev - city_elev) * LAPSE_RATE).astype(np.float32)[:, None]

    daily: dict[str, list[pd.Series]] = {c: [] for c in todo["slug"]}

    def add(column: str, values: np.ndarray, how: str) -> None:
        for i, slug in enumerate(todo["slug"]):
            daily[slug].append(to_local_days(values[i], start_h, offsets[i], how).rename(column))

    t = temperature_var("temperature_2m")
    add("temperature_2m_max", t, "max")
    add("temperature_2m_min", t, "min")
    add("dew_point_2m_mean", np.minimum(temperature_var("dew_point_2m"), t), "mean")
    del t
    add("precipitation_era5", hourly_series("copernicus_era5", "precipitation", cells, start_h, end_h), "sum")
    snow = hourly_series("copernicus_era5", "snowfall_water_equivalent", cells, start_h, end_h)
    add("snowfall_era5", snow * SNOW_CM_PER_MM, "sum")
    add("cloud_cover_mean", hourly_series("copernicus_era5", "cloud_cover", cells, start_h, end_h), "mean")
    u = hourly_series("copernicus_era5", "wind_u_component_10m", cells, start_h, end_h)
    v = hourly_series("copernicus_era5", "wind_v_component_10m", cells, start_h, end_h)
    add("wind_speed_10m_mean", np.hypot(u, v) * 3.6, "mean")  # m/s -> km/h

    climate.DAILY_DIR.mkdir(parents=True, exist_ok=True)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    for _, row in todo.iterrows():
        df = pd.concat(daily[row["slug"]], axis=1).loc[lo:hi].round(2)
        df.index.name = "time"
        df.to_csv(climate.daily_path(row["slug"]))

    log = todo[["slug", "lat", "lon", "elevation"]].assign(
        era5_cell=[era5.latlon(c) for c in cells],
        era5_land_cell=[land.latlon(c) if c else None for c in land_cells],
    )
    log_path = climate.DAILY_DIR / "_cells.csv"
    if log_path.exists():
        log = pd.concat([pd.read_csv(log_path), log]).drop_duplicates("slug", keep="last")
    log.to_csv(log_path, index=False)
    print(f"wrote {len(todo)} cities to {climate.DAILY_DIR}", flush=True)


if __name__ == "__main__":
    fetch_missing(climate.all_cities())
