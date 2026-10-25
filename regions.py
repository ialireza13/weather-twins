"""US climate regions for the twin-region map: NOAA's 344 CONUS climate divisions.

Each division is represented by the largest town (GeoNames, population >= 5,000)
in its central part, so the weather is where people live but a big city on the
edge doesn't speak for the whole region; divisions without a town use a point
inside the polygon. Writes:
  data/us_regions.csv       one row per division, same columns as the city lists,
                            so era5.py and rainfall.py treat divisions like cities
  data/us_regions.geojson   simplified shapes (shared borders kept exact) with
                            each division's neighbors, for the website

Usage: .venv/bin/python regions.py
"""

import json
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import shapefile
import shapely
from shapely.geometry import mapping, shape
from shapely.ops import polylabel

import climate

CACHE = climate.DATA_DIR / "cache"
DIVISIONS_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/CONUS_CLIMATE_DIVISIONS.shp.zip"
TOWNS_URL = "https://download.geonames.org/export/dump/cities5000.zip"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
SIMPLIFY_DEG = 0.05  # ~5 km; plenty for a national map, and keeps the file small
CORE_SHARE = 0.4     # look for the representative town in the central 40% of a division

REGIONS_CSV = climate.DATA_DIR / "us_regions.csv"
REGIONS_GEOJSON = climate.DATA_DIR / "us_regions.geojson"


def download(url: str, folder) -> None:
    if folder.exists():
        return
    folder.mkdir(parents=True)
    zip_path = folder / "download.zip"
    urllib.request.urlretrieve(url, zip_path)
    zipfile.ZipFile(zip_path).extractall(folder)
    zip_path.unlink()


def tidy_name(name: str) -> str:
    """NOAA's division names are upper-case and abbreviated ("SACRAMENTO DRNG.")."""
    words = {"Drng.": "Drainage", "Drng": "Drainage", "Inter.": "Interior", "Mtns": "Mountains",
             "Mtns.": "Mountains", "Mt.": "Mountain", "Cntrl": "Central", "Cent.": "Central"}
    out = []
    for w in name.title().split():
        if w.upper() in {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}:
            w = w.upper()
        out.append(words.get(w, w))
    return " ".join(out)


def load_divisions() -> tuple[pd.DataFrame, list]:
    folder = CACHE / "climdiv"
    download(DIVISIONS_URL, folder)
    path = next(folder.glob("*.shp"))
    reader = shapefile.Reader(str(path), encoding="cp1252")
    rows, geoms = [], []
    for rec, shp in zip(reader.records(), reader.shapes()):
        rows.append({"id": rec["FIPS_CD"], "division": tidy_name(rec["NAME"]),
                     "state": rec["STATE"], "st": rec["ST_ABBRV"]})
        geoms.append(shape(shp.__geo_interface__).buffer(0))
    return pd.DataFrame(rows), geoms


def load_towns() -> pd.DataFrame:
    folder = CACHE / "geonames"
    download(TOWNS_URL, folder)
    cols = ["id", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2",
            "admin1", "admin2", "admin3", "admin4", "population", "elevation", "dem", "tz", "mod"]
    towns = pd.read_csv(folder / "cities5000.txt", sep="\t", names=cols, usecols=range(19),
                        dtype={"admin1": str}, keep_default_na=False, na_values={"elevation": [""]},
                        quoting=3)
    # PPLX = a section of a city (e.g. Brooklyn), not a town of its own.
    return towns[(towns["cc"] == "US") & (towns["fcode"] != "PPLX")].reset_index(drop=True)


def core(geom, share: float = CORE_SHARE):
    """The division shrunk inward until it keeps `share` of its area.

    A big city on a division's edge (Manhattan in "Hudson Valley") shouldn't speak
    for the whole region, so the representative town comes from the middle.
    """
    lo, hi = 0.0, 2.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if geom.buffer(-mid).area > share * geom.area:
            lo = mid
        else:
            hi = mid
    shrunk = geom.buffer(-lo)
    return shrunk if not shrunk.is_empty else geom


def dem_elevations(points: list[tuple[float, float]]) -> list[float]:
    out = []
    for i in range(0, len(points), 100):
        chunk = points[i:i + 100]
        params = {"latitude": ",".join(f"{la:.4f}" for la, _ in chunk),
                  "longitude": ",".join(f"{lo:.4f}" for _, lo in chunk)}
        with urllib.request.urlopen(f"{ELEVATION_URL}?{urllib.parse.urlencode(params)}", timeout=60) as r:
            out += json.load(r)["elevation"]
    return out


def main() -> None:
    divisions, geoms = load_divisions()
    towns = load_towns()
    town_pts = shapely.points(towns["lon"], towns["lat"])

    reps = []
    for geom in geoms:
        inside = shapely.contains(core(geom), town_pts)
        if not inside.any():
            inside = shapely.contains(geom, town_pts)
        if inside.any():
            t = towns[inside].nlargest(1, "population").iloc[0]
            reps.append((t["name"], t["lat"], t["lon"], float(t["dem"])))
        else:
            p = geom.representative_point()
            reps.append((None, p.y, p.x, np.nan))
    divisions["town"] = [r[0] for r in reps]
    divisions["lat"] = [round(r[1], 5) for r in reps]
    divisions["lon"] = [round(r[2], 5) for r in reps]
    divisions["elevation"] = [r[3] for r in reps]
    missing = divisions["elevation"].isna() | (divisions["elevation"] < -500)
    if missing.any():
        divisions.loc[missing, "elevation"] = dem_elevations(
            list(zip(divisions.loc[missing, "lat"], divisions.loc[missing, "lon"])))

    divisions["name"] = divisions["state"] + ": " + divisions["division"]
    divisions["country"] = "United States"
    divisions["cc"] = "US"
    divisions["slug"] = "region-" + divisions["id"]
    divisions[["slug", "id", "name", "state", "st", "division", "town", "country", "cc",
               "lat", "lon", "elevation"]].to_csv(REGIONS_CSV, index=False)

    # Simplify all divisions together so neighbors keep identical shared borders.
    simple = shapely.coverage_simplify(np.array(geoms, dtype=object), SIMPLIFY_DEG)
    tree = shapely.STRtree(simple)
    features = []
    for i, geom in enumerate(simple):
        near = tree.query(geom, predicate="intersects")
        neighbors = [divisions["slug"][j] for j in near
                     if j != i and geom.intersection(simple[j]).length > 0]
        geom = shapely.set_precision(geom, 0.01)
        biggest = max(getattr(geom, "geoms", [geom]), key=lambda g: g.area)
        label = polylabel(biggest, tolerance=0.01)  # visual center, for map labels
        features.append({
            "type": "Feature",
            "properties": {"id": divisions["slug"][i], "neighbors": neighbors,
                           "label": [round(label.y, 3), round(label.x, 3)],
                           "area": round(geom.area, 3)},
            "geometry": mapping(geom),
        })
    REGIONS_GEOJSON.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                          separators=(",", ":")))
    print(f"{len(divisions)} divisions, {divisions['town'].notna().sum()} with a town; "
          f"geojson {REGIONS_GEOJSON.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
