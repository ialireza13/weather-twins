"""Climate profiles and similarity for the weather-twin prototype.

Data: ERA5 reanalysis (Copernicus) via Open-Meteo's open data on AWS, CC BY 4.0.
Daily CSVs in data/daily/ are produced by era5.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

# Features in a monthly profile, with the default weight each gets in the distance.
# Temperature dominates how weather feels, humidity next, then rain and sky.
FEATURES = {
    "tmax": 1.0,       # mean daily high, °C
    "tmin": 1.0,       # mean daily low, °C
    "dewpoint": 0.8,   # mean dew point, °C (mugginess)
    "precip": 0.5,     # monthly precipitation total, mm (log-scaled for distance)
    "rain_days": 0.5,  # days per month with >= 1 mm
    "cloud": 0.5,      # mean cloud cover, %
    "snow": 0.3,       # monthly snowfall total, cm (log-scaled for distance)
    "wind": 0.2,       # mean wind speed, km/h
    "tmax_sd": 0.3,    # day-to-day spread of highs within a month, °C
}
LOG_FEATURES = {"precip", "snow"}


# ---------------------------------------------------------------- cities

def load_cities(path: str | Path) -> pd.DataFrame:
    # keep_default_na=False so Namibia's country code "NA" stays a string.
    df = pd.read_csv(path, keep_default_na=False, na_values={"lat": [""], "lon": [""], "elevation": [""]})
    df["slug"] = [slugify(f"{n}_{c}") for n, c in zip(df["name"], df["country"])]
    return df


def all_cities() -> pd.DataFrame:
    return pd.concat([load_cities(DATA_DIR / "us_cities.csv"),
                      load_cities(DATA_DIR / "world_cities.csv")], ignore_index=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def daily_path(slug: str) -> Path:
    return DAILY_DIR / f"{slug}.csv"


def load_daily(slug: str) -> pd.DataFrame:
    return pd.read_csv(daily_path(slug), parse_dates=["time"])


# ---------------------------------------------------------------- profiles

def monthly_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Typical year: 12 rows (months 1-12), one column per feature."""
    d = daily.copy()
    d["year"] = d["time"].dt.year
    d["month"] = d["time"].dt.month
    d["wet"] = d["precipitation_sum"] >= 1.0

    # Per year-month first, then average across years, so totals stay monthly totals.
    ym = d.groupby(["year", "month"]).agg(
        tmax=("temperature_2m_max", "mean"),
        tmin=("temperature_2m_min", "mean"),
        dewpoint=("dew_point_2m_mean", "mean"),
        precip=("precipitation_sum", "sum"),
        rain_days=("wet", "sum"),
        cloud=("cloud_cover_mean", "mean"),
        snow=("snowfall_sum", "sum"),
        wind=("wind_speed_10m_mean", "mean"),
        tmax_sd=("temperature_2m_max", "std"),
    )
    return ym.groupby("month").mean()[list(FEATURES)]


def build_profiles(cities: pd.DataFrame) -> pd.DataFrame:
    """Long table indexed by (slug, month) for every city with cached data."""
    frames = {}
    for _, city in cities.iterrows():
        if daily_path(city["slug"]).exists():
            frames[city["slug"]] = monthly_profile(load_daily(city["slug"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, names=["slug", "month"])


# ---------------------------------------------------------------- similarity

def standardize(profiles: pd.DataFrame) -> pd.DataFrame:
    """z-score each feature over all city-months, so features share a scale."""
    z = profiles.copy()
    for col in LOG_FEATURES:
        z[col] = np.log1p(z[col])
    return (z - z.mean()) / z.std(ddof=0)


def align_seasons(z: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Shift Southern Hemisphere cities by 6 months so July lines up with January."""
    south = set(cities.loc[cities["lat"] < 0, "slug"])
    out = z.copy()
    for slug in south & set(z.index.get_level_values("slug")):
        out.loc[slug] = np.roll(z.loc[slug].to_numpy(), 6, axis=0)
    return out


def find_similar(query_slug: str, candidates: pd.DataFrame, profiles: pd.DataFrame,
                 cities: pd.DataFrame, weights: dict | None = None,
                 seasons: bool = False, top: int = 10,
                 max_per_country: int | None = None) -> pd.DataFrame:
    """Rank candidate cities by climate distance to the query city.

    distance = weighted RMS of z-score differences over 12 months x features.
    0 means identical; ~1 means about one standard deviation apart on average.
    max_per_country keeps results varied (e.g. not five Canadian cities in a row).
    """
    weights = {**FEATURES, **(weights or {})}
    cols = [f for f, w in weights.items() if w > 0]
    w = np.array([weights[f] for f in cols])

    z = standardize(profiles)[cols]
    if seasons:
        z = align_seasons(z, cities)

    q = z.loc[query_slug].to_numpy()
    rows = []
    for slug in candidates["slug"]:
        if slug == query_slug or slug not in z.index.get_level_values("slug"):
            continue
        diff = z.loc[slug].to_numpy() - q
        dist = np.sqrt((diff**2 * w).sum() / (12 * w.sum()))
        rows.append((slug, dist))

    ranked = pd.DataFrame(rows, columns=["slug", "distance"]).sort_values("distance")
    ranked = ranked.merge(cities[["slug", "name", "country"]], on="slug")
    if max_per_country:
        ranked = ranked[ranked.groupby("country").cumcount() < max_per_country]
    ranked = ranked.head(top).reset_index(drop=True)
    ranked["why"] = [explain(profiles.loc[query_slug], profiles.loc[s]) for s in ranked["slug"]]
    return ranked[["name", "country", "distance", "why", "slug"]]


def explain(a: pd.DataFrame, b: pd.DataFrame) -> str:
    """Short human summary of how city b's typical year differs from city a's."""
    notes = []
    dh = b["tmax"].mean() - a["tmax"].mean()
    dl = b["tmin"].mean() - a["tmin"].mean()
    if abs(dh) >= 1 or abs(dl) >= 1:
        notes.append(f"highs {dh:+.1f}°C, lows {dl:+.1f}°C")
    else:
        notes.append("same temps")
    pa, pb = a["precip"].sum(), b["precip"].sum()
    ratio = pb / pa if pa > 0 else np.inf
    if ratio >= 1.3:
        notes.append(f"{ratio:.1f}x the rain")
    elif ratio <= 0.77:
        notes.append(f"{1 / ratio:.1f}x drier" if ratio > 0 else "no rain")
    dd = b["dewpoint"].mean() - a["dewpoint"].mean()
    if dd >= 2:
        notes.append("more humid")
    elif dd <= -2:
        notes.append("drier air")
    dc = b["cloud"].mean() - a["cloud"].mean()
    if dc >= 10:
        notes.append("cloudier")
    elif dc <= -10:
        notes.append("sunnier")
    return ", ".join(notes)
