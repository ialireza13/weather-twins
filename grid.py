"""Global 1° land grid of climate profiles, for looking up any city in the world.

Cells are 1°x1° blocks between 60°S and 84°N where at least LAND_SHARE of the
ERA5-Land cells are land. Each is represented by the ERA5-Land land cell nearest
the block's center, and runs through the same pipeline as the cities (era5.py),
with rain scaled to WorldClim normals (rain gauges per cell aren't available).
Temperatures are not elevation-corrected here: each cell keeps its own elevation,
so the website can adjust to a looked-up city's elevation.

Work happens in batches, so memory stays bounded and an interrupted run resumes:
  data/grid/cells.csv               the cells
  data/grid/parts/part_NNN.csv      monthly profiles, one file per finished batch
  data/grid/profiles.csv            all parts combined (written at the end)

Usage:
  .venv/bin/python grid.py --prepare                         # cells + WorldClim, once
  .venv/bin/python grid.py --aws --shard I --shards N        # run N of these in parallel
  .venv/bin/python grid.py --combine                         # merge when all shards finish
  (--limit N for a quick timing test; one process without --shards does everything)
"""

from __future__ import annotations

import argparse
import time

import fsspec
import numpy as np
import pandas as pd

import climate
import era5
import rainfall

GRID_DIR = climate.DATA_DIR / "grid"
PARTS = GRID_DIR / "parts"
STEP = 1.0          # degrees
LAND_SHARE = 0.25   # minimum land fraction of a block
LAT_MIN, LAT_MAX = -60.0, 84.0
BATCH = 1000


def make_cells() -> pd.DataFrame:
    land = era5.Grid("copernicus_era5_land", 0.1)
    per = int(round(STEP / land.step))
    rows = []
    for lat0 in np.arange(LAT_MIN, LAT_MAX, STEP):
        for lon0 in np.arange(-180.0, 180.0, STEP):
            iy0, ix0 = land.index(lat0, lon0)
            block = land.land[iy0:iy0 + per, ix0:ix0 + per]
            share = block.mean() if block.size else 0
            if share < LAND_SHARE:
                continue
            ys, xs = np.nonzero(block)
            cy, cx = (per - 1) / 2, (per - 1) / 2
            k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
            cell = (iy0 + ys[k], ix0 + xs[k])
            lat, lon = land.latlon(cell)
            rows.append({
                "slug": f"g{lat0 + STEP / 2:+06.1f}{lon0 + STEP / 2:+07.1f}",
                "block_lat": lat0 + STEP / 2, "block_lon": lon0 + STEP / 2,
                "lat": round(lat, 2), "lon": round(lon, 2),
                "cell_elevation": float(land.hsurf[cell]), "land_share": round(float(share), 2),
            })
    cells = pd.DataFrame(rows)
    # No elevation correction in the pipeline: era5.py fills NaN with the grid's own elevation.
    cells["elevation"] = np.nan
    cells["name"] = cells["slug"]
    return cells


def profile_rows(batch: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, c in batch.iterrows():
        daily = climate.load_daily(c["slug"])
        daily = daily.rename(columns={"precipitation_era5": "precipitation_raw", "snowfall_era5": "snowfall_raw"})
        model = daily.set_index("time")["precipitation_raw"]
        normal = normals.loc[c["slug"]].to_numpy()
        if not np.isnan(normal).any():
            corrected, factors = rainfall.scale_to_normals(model, normal)
        else:  # e.g. ice sheets, where WorldClim has no data
            corrected, factors = model, pd.Series(1.0, index=range(1, 13))
        daily["precipitation_sum"] = corrected.to_numpy()
        daily["snowfall_sum"] = daily["snowfall_raw"] * daily["time"].dt.month.map(factors.clip(0, 5))
        prof = climate.monthly_profile(daily)
        row = {"slug": c["slug"]}
        for f in climate.FEATURES:
            for m in range(1, 13):
                row[f"{f}_{m}"] = round(float(prof.loc[m, f]), 2)
        out.append(row)
    return pd.DataFrame(out)


def load_cells() -> pd.DataFrame:
    cells_path = GRID_DIR / "cells.csv"
    if not cells_path.exists():
        GRID_DIR.mkdir(parents=True, exist_ok=True)
        make_cells().to_csv(cells_path, index=False)
    return pd.read_csv(cells_path)


def tune_for_aws() -> None:
    """Next to the bucket, requests are cheap: many more in flight per process."""
    era5.WORKERS, era5.CELLS_PER_TASK = 256, 4
    era5.fs = fsspec.filesystem("s3", anon=True, skip_instance_cache=True,
                                config_kwargs={"max_pool_connections": 256})


def run_shard(cells: pd.DataFrame, shard: int, shards: int, test: bool) -> None:
    # Each shard keeps its own days and hourly cache, so parallel processes never collide.
    climate.DAILY_DIR = GRID_DIR / "daily" / f"shard{shard}"
    era5.CACHE_DIR = climate.DATA_DIR / "cache" / f"grid_shard{shard}"
    normals = rainfall.worldclim_normals(cells)
    batches = [cells.iloc[i:i + BATCH] for i in range(0, len(cells), BATCH)]
    mine = [n for n in range(len(batches)) if n % shards == shard]
    for k, n in enumerate(mine, 1):
        part = PARTS / f"part_{n:03d}{'_test' if test else ''}.csv"
        if part.exists():
            continue
        t0 = time.time()
        print(f"[shard {shard}] batch {n} ({k}/{len(mine)}): {len(batches[n])} cells", flush=True)
        era5.fetch_missing(batches[n])
        profile_rows(batches[n], normals).to_csv(part, index=False)
        for f in era5.CACHE_DIR.glob("*.npy"):  # hourly arrays are big; the days are on disk now
            f.unlink()
        print(f"[shard {shard}] batch {n} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def combine(cells: pd.DataFrame) -> None:
    batches = -(-len(cells) // BATCH)
    parts = sorted(PARTS.glob("part_[0-9][0-9][0-9].csv"))
    if len(parts) < batches:
        print(f"only {len(parts)} of {batches} batches done")
        return
    profiles = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    cells.merge(profiles, on="slug").to_csv(GRID_DIR / "profiles.csv", index=False)
    print(f"wrote {len(profiles)} cell profiles to {GRID_DIR / 'profiles.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="make cells.csv and fetch WorldClim, then stop")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--combine", action="store_true", help="merge finished batches into profiles.csv")
    parser.add_argument("--limit", type=int, help="only the first N cells (timing test)")
    parser.add_argument("--aws", action="store_true", help="high request concurrency (run next to the bucket)")
    args = parser.parse_args()

    PARTS.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    print(f"{len(cells)} land cells", flush=True)
    if args.prepare:
        rainfall.worldclim_normals(cells.head(1))  # downloads WorldClim once, before shards start
        return
    if args.combine:
        combine(cells)
        return
    if args.aws:
        tune_for_aws()
    run_shard(cells.head(args.limit) if args.limit else cells, args.shard, args.shards, bool(args.limit))
    if args.shards == 1 and not args.limit:
        combine(cells)


if __name__ == "__main__":
    main()
