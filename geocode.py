"""Fill in lat/lon/elevation for any city row that is missing them.

Add a row with just name,country,cc (plus an optional `query`) and run:
    .venv/bin/python geocode.py

`query` overrides the search name; "Name|Region" also requires that admin region,
e.g. "Portland|Oregon". Uses the Open-Meteo Geocoding API (GeoNames data).
"""

import json
import urllib.parse
import urllib.request

import pandas as pd

import climate

URL = "https://geocoding-api.open-meteo.com/v1/search"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"


def geocode(name: str, cc: str, region: str | None) -> dict:
    params = {"name": name, "count": 20, "countryCode": cc, "language": "en", "format": "json"}
    with urllib.request.urlopen(f"{URL}?{urllib.parse.urlencode(params)}", timeout=30) as resp:
        results = json.load(resp).get("results", [])
    if region:
        results = [r for r in results if r.get("admin1", "").lower() == region.lower()]
    if not results:
        raise LookupError(f"no match for {name!r} in {cc} {region or ''}")
    # Results come ranked by relevance and population; prefer populated places.
    places = [r for r in results if r.get("feature_code", "").startswith("PPL")] or results
    return max(places[:5], key=lambda r: r.get("population") or 0)


def dem_elevation(lat: float, lon: float) -> float:
    """Ground elevation from Open-Meteo's 90 m DEM, for rows GeoNames lacks."""
    params = {"latitude": lat, "longitude": lon}
    with urllib.request.urlopen(f"{ELEVATION_URL}?{urllib.parse.urlencode(params)}", timeout=30) as resp:
        return json.load(resp)["elevation"][0]


def fill(path) -> None:
    # keep_default_na=False so Namibia's country code "NA" stays a string.
    df = pd.read_csv(path, dtype={"cc": str, "query": str}, keep_default_na=False,
                     na_values={c: [""] for c in ["query", "lat", "lon", "elevation"]})
    for i, row in df[df["lat"].isna()].iterrows():
        query = row["query"] if isinstance(row["query"], str) else row["name"]
        name, _, region = query.partition("|")
        try:
            hit = geocode(name, row["cc"], region or None)
        except LookupError as err:
            print(f"!! {row['name']}: {err} (set `query` and re-run)")
            continue
        df.loc[i, ["lat", "lon", "elevation"]] = hit["latitude"], hit["longitude"], hit.get("elevation")
        print(f"{row['name']:24s} -> {hit['name']}, {hit.get('admin1', '')} "
              f"({hit['latitude']:.3f}, {hit['longitude']:.3f}, {hit.get('elevation')} m, "
              f"pop {hit.get('population')})")
    for i, row in df[df["lat"].notna() & ~(df["elevation"] > -500)].iterrows():
        df.loc[i, "elevation"] = dem_elevation(row["lat"], row["lon"])
        print(f"{row['name']:24s} elevation from DEM: {df.loc[i, 'elevation']} m")
    df.to_csv(path, index=False)


if __name__ == "__main__":
    for f in ["us_cities.csv", "world_cities.csv"]:
        fill(climate.DATA_DIR / f)
