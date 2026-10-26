"""Export monthly climate profiles for the website.

docs/data/cities.json   the cities, with profiles and the standardization statistics
                        climate.standardize() uses (the site scores in the browser)
docs/data/regions.json  NOAA climate divisions: simplified shapes, neighbors and
                        profiles, for the twin-region map (needs regions.py first)

Usage: .venv/bin/python export_site.py
"""

import json

import numpy as np
import pandas as pd

import climate

OUT = climate.ROOT / "docs" / "data"

# Where a twin is, for coloring the region map. Three colored groups plus a neutral
# one: more categories than that can't be told apart on a map where any two touch.
GROUPS = {
    "Europe": ["Austria", "Belgium", "Bulgaria", "Croatia", "Czechia", "Denmark", "Estonia",
               "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
               "Latvia", "Lithuania", "Malta", "Netherlands", "Norway", "Poland", "Portugal",
               "Romania", "Russia", "Serbia", "Spain", "Sweden", "Switzerland", "Turkey",
               "Ukraine", "United Kingdom"],
    "Asia & Middle East": ["Afghanistan", "Azerbaijan", "Bangladesh", "Cambodia", "China",
                           "Georgia", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan",
                           "Jordan", "Kazakhstan", "Kuwait", "Lebanon", "Malaysia", "Maldives",
                           "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan",
                           "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea",
                           "Sri Lanka", "Syria", "Taiwan", "Thailand", "United Arab Emirates",
                           "Uzbekistan", "Vietnam"],
    "Americas": ["Argentina", "Aruba", "Bahamas", "Bolivia", "Brazil", "Canada", "Chile",
                 "Colombia", "Costa Rica", "Cuba", "Dominican Republic", "Ecuador", "Guatemala",
                 "Jamaica", "Mexico", "Panama", "Peru", "Uruguay", "Venezuela"],
    "Africa & Oceania": ["Algeria", "Angola", "Australia", "DR Congo", "Egypt", "Ethiopia", "Fiji",
                         "French Polynesia", "Ghana", "Ivory Coast", "Kenya", "Madagascar", "Mali",
                         "Mauritius", "Morocco", "Namibia", "New Zealand", "Nigeria", "Rwanda",
                         "Senegal", "South Africa", "Sudan", "Tanzania", "Tunisia", "Uganda"],
}
GROUP_OF = {country: group for group, countries in GROUPS.items() for country in countries}


def group_of(city: pd.Series) -> str:
    if city["country"] == "Russia" and city["lon"] > 60:  # Siberia and the Far East
        return "Asia & Middle East"
    return GROUP_OF[city["country"]]


def profile_json(profiles: pd.DataFrame, slug: str) -> dict:
    p = profiles.loc[slug]
    return {f: [round(float(v), 2) for v in p[f]] for f in climate.FEATURES}


def rain_label(log: pd.DataFrame, slug: str) -> str:
    if slug not in log.index:
        return "ERA5"
    name = log.loc[slug, "station_name"]
    return str(name).title() if isinstance(name, str) else "WorldClim normals"


def export_cities() -> None:
    cities = climate.all_cities()
    profiles = climate.build_profiles(cities)
    rain = pd.read_csv(climate.DATA_DIR / "rain_stations.csv").set_index("slug")
    us = set(climate.load_cities(climate.DATA_DIR / "us_cities.csv")["slug"])

    z = profiles.copy()
    for col in climate.LOG_FEATURES:
        z[col] = np.log1p(z[col])
    stats = {f: [round(float(z[f].mean()), 6), round(float(z[f].std(ddof=0)), 6)] for f in climate.FEATURES}

    out = []
    for _, c in cities.iterrows():
        if c["slug"] not in profiles.index.get_level_values("slug"):
            continue
        row = {
            "id": c["slug"],
            "name": c["name"],
            "country": c["country"],
            "lat": round(float(c["lat"]), 4),
            "lon": round(float(c["lon"]), 4),
            "us": c["slug"] in us,
            "rain": rain_label(rain, c["slug"]),
            "p": profile_json(profiles, c["slug"]),
        }
        if not row["us"]:
            row["group"] = group_of(c)
        out.append(row)

    data = {
        "period": f"{climate.START_DATE[:4]}–{climate.END_DATE[:4]}",
        "features": list(climate.FEATURES),
        "weights": climate.FEATURES,
        "log_features": sorted(climate.LOG_FEATURES),
        "stats": stats,
        "groups": list(GROUPS),
        "cities": out,
    }
    write(OUT / "cities.json", data, f"{len(out)} cities")


def export_regions() -> None:
    geo_path = climate.DATA_DIR / "us_regions.geojson"
    if not geo_path.exists():
        print("no regions (run regions.py first)")
        return
    regions = climate.load_regions().set_index("slug")
    profiles = climate.build_profiles(regions.reset_index())
    if profiles.empty:
        print("no region weather data yet (run era5.py / rainfall.py for regions first)")
        return
    have = set(profiles.index.get_level_values("slug"))
    log_path = climate.DATA_DIR / "rain_stations_regions.csv"
    rain = pd.read_csv(log_path).set_index("slug") if log_path.exists() else pd.DataFrame()

    features = []
    for f in json.loads(geo_path.read_text())["features"]:
        slug = f["properties"]["id"]
        if slug not in have:
            continue
        r = regions.loc[slug]
        f["properties"].update({
            "state": r["state"],
            "st": r["st"],
            "division": r["division"],
            "town": r["town"] if isinstance(r["town"], str) else None,
            "rain": rain_label(rain, slug),
            "p": profile_json(profiles, slug),
        })
        f["properties"]["neighbors"] = [n for n in f["properties"]["neighbors"] if n in have]
        features.append(f)
    write(OUT / "regions.json", {"type": "FeatureCollection", "features": features},
          f"{len(features)} regions")


def export_grid() -> None:
    """The global 1° grid (grid.py) as a compact binary file for looking up any city.

    grid.bin, little-endian: N x int16 lat*100, N x int16 lon*100, N x int16 elevation (m),
    then N x 108 uint8 values (feature-major, then month). Each feature is quantized to
    0-255 over its own range; rain and snow on the log1p scale the scoring uses, so dry
    months keep their precision. grid.json holds N and each feature's range.
    """
    path = climate.DATA_DIR / "grid" / "profiles.csv"
    if not path.exists():
        print("no grid (run grid.py first)")
        return
    g = pd.read_csv(path)
    n = len(g)
    ranges, values = {}, []
    for f in climate.FEATURES:
        v = g[[f"{f}_{m}" for m in range(1, 13)]].to_numpy(dtype=float)
        if f in climate.LOG_FEATURES:
            v = np.log1p(np.maximum(v, 0))
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        ranges[f] = [round(lo, 4), round(hi, 4)]
        values.append(np.clip(np.rint((v - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8))
    body = np.concatenate(values, axis=1)  # n x 108
    blob = b"".join([
        np.rint(g["lat"].to_numpy() * 100).astype("<i2").tobytes(),
        np.rint(g["lon"].to_numpy() * 100).astype("<i2").tobytes(),
        np.rint(g["cell_elevation"].to_numpy()).astype("<i2").tobytes(),
        body.tobytes(),
    ])
    (OUT / "grid.bin").write_bytes(blob)
    write(OUT / "grid.json", {"n": n, "step": 1.0, "features": list(climate.FEATURES),
                              "ranges": ranges, "log_features": sorted(climate.LOG_FEATURES)},
          "grid metadata")
    print(f"wrote {n} grid cells to docs/data/grid.bin ({len(blob) / 1024:.0f} KB)")


def write(path, data, what: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {what} to {path.relative_to(climate.ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    export_cities()
    export_regions()
    export_grid()
