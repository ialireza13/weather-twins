"""Bias-correct ERA5 precipitation with nearby rain gauges.

ERA5 drizzles too often and over-rains near mountains (Lima, Cusco, Lhasa,
Anchorage). For each city we find a rain gauge within 50 km with a mostly
complete 2016-2025 record:
  1. NOAA GHCN-Daily (quality-controlled; best coverage in the US/Europe), else
  2. NOAA GSOD (daily summaries of airport SYNOP reports; wider global coverage).

Then local intensity scaling (LOCI, Schmidli et al. 2006), per calendar month:
  - pick an ERA5 wet-day threshold so ERA5 has as many >= 1 mm days as the gauge,
  - scale wet-day amounts so the monthly mean matches the gauge.
The corrected series stays complete and daily even where the gauge has gaps.
Snowfall is scaled by the same monthly factor as total precipitation.

Gauges are sanity-checked first: obviously spurious days are dropped, and a gauge
whose total disagrees by >2x with both ERA5 and the WorldClim normal is rejected.
Cities without a usable gauge get ERA5 scaled month by month to WorldClim 2.1
normals (1970-2000, gauge-interpolated).

Reads precipitation_era5 / snowfall_era5 from data/daily/*.csv and writes
precipitation_sum / snowfall_sum next to them. Choices are logged in
data/rain_stations.csv (cities) and data/rain_stations_regions.csv (US climate divisions).

Usage: .venv/bin/python rainfall.py
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import climate

CACHE = climate.DATA_DIR / "cache" / "rain"
LOG_PATH = climate.DATA_DIR / "rain_stations.csv"
REGIONS_LOG_PATH = climate.DATA_DIR / "rain_stations_regions.csv"
GHCN = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ACCESS = "https://www.ncei.noaa.gov/access/services/data/v1"

MAX_KM = 50
MAX_ELEV_DIFF = 500        # m; a gauge far above/below the city sees different rain
MIN_COMPLETE = 0.6         # fraction of 2016-2025 days with a valid reading
GOOD_COMPLETE = 0.9        # take the first (WMO-first, nearest) gauge at least this complete
WET_MM = 1.0
SPURIOUS_MM = 25.0         # gauge days this big need some ERA5 rain nearby to be believed
MAX_RATIO = 2.0            # plausibility: gauge within 2x of ERA5 or of the WorldClim normal
ARID_MM = 100.0            # below this, annual totals are too small to compare by ratio
WORLDCLIM = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_5m_prec.zip"
GSOD_REPORT_FLAGS = set("ABCDEFGH")  # some 6/12/24 h report (H = 0 with a trace)
GSOD_MIN_REPORTS = 30                # over the whole period, to count as actively reporting
INCH = 25.4


# ---------------------------------------------------------------- catalogs

def fetch(url: str, path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


def ghcn_catalog(start: int, end: int) -> pd.DataFrame:
    fetch(f"{GHCN}/ghcnd-stations.txt", CACHE / "ghcnd-stations.txt")
    fetch(f"{GHCN}/ghcnd-inventory.txt", CACHE / "ghcnd-inventory.txt")
    st = pd.read_fwf(CACHE / "ghcnd-stations.txt",
                     colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (41, 71), (80, 85)],
                     names=["id", "lat", "lon", "elev", "name", "wmo_id"])
    inv = pd.read_fwf(CACHE / "ghcnd-inventory.txt", colspecs=[(0, 11), (31, 35), (36, 40), (41, 45)],
                      names=["id", "element", "first", "last"])
    ok = inv[(inv["element"] == "PRCP") & (inv["first"] <= start) & (inv["last"] >= end)]
    st = st[st["id"].isin(ok["id"])]
    return st.assign(source="ghcn", wmo=st.pop("wmo_id").notna())


def gsod_catalog(start: int, end: int) -> pd.DataFrame:
    fetch(ISD_HISTORY, CACHE / "isd-history.csv")
    isd = pd.read_csv(CACHE / "isd-history.csv", dtype={"USAF": str, "WBAN": str})
    isd = isd[(isd["BEGIN"] <= start * 10000 + 101) & (isd["END"] >= end * 10000 + 601)].dropna(subset=["LAT", "LON"])
    return pd.DataFrame({
        "id": isd["USAF"] + isd["WBAN"], "lat": isd["LAT"], "lon": isd["LON"],
        "elev": isd["ELEV(M)"], "name": isd["STATION NAME"], "source": "gsod", "wmo": True,
    }).query("not (lat == 0 and lon == 0)")


def nearby(catalog: pd.DataFrame, lat: float, lon: float, n: int = 3) -> pd.DataFrame:
    """The n closest WMO (synoptic, professionally maintained) stations, then the n closest
    others. Dense volunteer networks (e.g. CoCoRaHS in the US) would otherwise crowd out
    the airport station, and their gauges drift more."""
    la, lo = np.radians(lat), np.radians(lon)
    sla, slo = np.radians(catalog["lat"].to_numpy()), np.radians(catalog["lon"].to_numpy())
    km = 6371 * 2 * np.arcsin(np.sqrt(np.sin((sla - la) / 2) ** 2
                                      + np.cos(la) * np.cos(sla) * np.sin((slo - lo) / 2) ** 2))
    out = catalog.assign(km=km)
    out = out[out["km"] <= MAX_KM]
    return pd.concat([out[out["wmo"]].nsmallest(n, "km"), out[~out["wmo"]].nsmallest(n, "km")])


# ---------------------------------------------------------------- gauge series

def gauge_series(station: pd.Series, start: str, end: str) -> pd.Series:
    """Daily precipitation in mm, NaN where missing or failing quality checks."""
    path = CACHE / f"{station['source']}_{station['id']}.csv"
    if not path.exists():
        dataset = "daily-summaries" if station["source"] == "ghcn" else "global-summary-of-the-day"
        params = {"dataset": dataset, "stations": station["id"], "dataTypes": "PRCP",
                  "startDate": start, "endDate": end, "format": "csv", "includeAttributes": "true"}
        url = f"{ACCESS}?{urllib.parse.urlencode(params)}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=180) as resp:
                    text = resp.read().decode()
                break
            except (urllib.error.URLError, TimeoutError) as err:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    df = pd.read_csv(path, dtype={"PRCP_ATTRIBUTES": str}, keep_default_na=False)
    days = pd.date_range(start, end, freq="D")
    if df.empty or "PRCP" not in df:
        return pd.Series(np.nan, index=days)
    df["DATE"] = pd.to_datetime(df["DATE"])
    value = pd.to_numeric(df["PRCP"], errors="coerce")
    attrs = df["PRCP_ATTRIBUTES"].fillna("")
    if station["source"] == "ghcn":
        # Attributes are "measurement,quality,source[,time]"; any quality flag means it failed QC.
        quality = attrs.str.split(",").str[1].fillna("").str.strip()
        value = value.where(quality == "") / 10  # tenths of mm
    else:
        flag = attrs.str.strip()
        reported = (value < 99) & flag.isin(GSOD_REPORT_FLAGS)
        # "I" = no precipitation report that day. Many countries only report precipitation
        # when it falls, so for a station that reports regularly an "I" day is a dry day.
        # Years without a single report mean the station wasn't reporting at all.
        year = df["DATE"].dt.year
        active_year = reported.groupby(year).transform("sum") > 0
        dry = (flag == "I") & active_year & (reported.sum() >= GSOD_MIN_REPORTS)
        value = (value.where(reported) * INCH).mask(dry, 0.0)
    return pd.Series(value.to_numpy(), index=df["DATE"]).groupby(level=0).first().reindex(days)


def plausible(gauge_mm: float, references: list[float]) -> bool:
    """A gauge is rejected only if it disagrees with every reference by more than 2x.

    References are ERA5 and the WorldClim normal; each has its own failure modes
    (ERA5 drizzle and mountains, WorldClim cells that include nearby mountains), but
    a broken gauge (missing reports read as dry, garbage values) disagrees with both.
    Very dry places are always plausible: tiny totals have huge ratios.
    """
    if gauge_mm < ARID_MM and min(references) < ARID_MM:
        return True
    return any(1 / MAX_RATIO <= gauge_mm / ref <= MAX_RATIO for ref in references if ref > 0)


def choose_gauge(city: pd.Series, catalogs: list[pd.DataFrame], start: str, end: str,
                 model: pd.Series, references: list[float], rejected: list[str]):
    """First catalog with a gauge that is close, at a similar height, mostly complete
    and plausible.

    Candidates are tried WMO stations first, then by distance; the first one that is
    nearly complete wins, otherwise the most complete one above MIN_COMPLETE.
    """
    for catalog in catalogs:
        best = None
        for _, st in nearby(catalog, city["lat"], city["lon"]).iterrows():
            if pd.notna(city["elevation"]) and pd.notna(st["elev"]) and st["elev"] > -999 \
                    and abs(st["elev"] - city["elevation"]) > MAX_ELEV_DIFF:
                continue
            series = drop_spurious(gauge_series(st, start, end), model)
            complete = series.notna().mean()
            if complete >= MIN_COMPLETE and not plausible(series.mean() * 365.25, references):
                rejected.append(f"{st['source']}:{st['name'].strip()}")
                continue
            if complete >= GOOD_COMPLETE:
                return st, series, complete
            if complete >= MIN_COMPLETE and (best is None or complete > best[2]):
                best = (st, series, complete)
        if best:
            return best
    return None


# ---------------------------------------------------------------- correction

def drop_spurious(gauge: pd.Series, model: pd.Series) -> pd.Series:
    """Blank out big gauge totals on days ERA5 saw essentially nothing (±1 day).

    GSOD in particular has occasional garbage values (e.g. 99 mm days in Cairo);
    a real storm of that size nearly always shows up in ERA5 at least lightly.
    """
    model_near = model.reindex(gauge.index).rolling(3, center=True, min_periods=1).max()
    return gauge.mask((gauge >= SPURIOUS_MM) & (model_near < 0.5))


def loci(model: pd.Series, gauge: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Corrected daily precipitation, and the per-month total scaling factor (for snow)."""
    corrected = pd.Series(0.0, index=model.index)
    factors = {}
    for month in range(1, 13):
        m = model[model.index.month == month]
        both = pd.concat([m, gauge.reindex(m.index)], axis=1, keys=["m", "g"]).dropna()
        g_wet = both["g"][both["g"] >= WET_MM]
        freq = len(g_wet) / len(both)
        if freq == 0:
            threshold, scale = np.inf, 0.0
        else:
            threshold = max(float(np.quantile(both["m"], 1 - freq)), 0.1)
            m_wet = both["m"][both["m"] >= threshold]
            excess = m_wet.mean() - threshold
            scale = (g_wet.mean() - WET_MM) / excess if excess > 0 else 1.0
            scale = float(np.clip(scale, 0.1, 10))
        fixed = np.where(m >= threshold, WET_MM + scale * (m - threshold), 0.0)
        corrected[m.index] = fixed
        factors[month] = fixed.sum() / m.sum() if m.sum() > 0 else 1.0
    return corrected, pd.Series(factors)


def scale_to_normals(model: pd.Series, normals: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Fallback without a gauge: scale each calendar month to the WorldClim normal."""
    years = model.index.year.nunique()
    monthly = model.groupby(model.index.month).sum() / years
    factors = pd.Series(normals, index=range(1, 13)) / monthly.replace(0, np.nan)
    factors = factors.fillna(1.0).clip(0.1, 10)
    return model * model.index.month.map(factors).to_numpy(), factors


# ---------------------------------------------------------------- WorldClim

def worldclim_normals(cities: pd.DataFrame) -> pd.DataFrame:
    """WorldClim 2.1 monthly precipitation normals (1970-2000, 5 arc-minutes), mm.

    Nearest valid cell; WorldClim is land-only, so coastal cities may need a neighbor.
    """
    import tifffile  # only needed here

    folder = CACHE.parent / "worldclim"
    if not (folder / "wc2.1_5m_prec_01.tif").exists():
        zip_path = folder / "wc2.1_5m_prec.zip"
        fetch(WORLDCLIM, zip_path)
        zipfile.ZipFile(zip_path).extractall(folder)
        zip_path.unlink()
    out = np.full((len(cities), 12), np.nan)
    for m in range(12):
        grid = tifffile.imread(folder / f"wc2.1_5m_prec_{m + 1:02d}.tif")
        res = 180 / grid.shape[0]
        for i, (lat, lon) in enumerate(zip(cities["lat"], cities["lon"])):
            iy, ix = int((90 - lat) / res), int((lon + 180) / res) % grid.shape[1]
            for r in range(4):
                win = grid[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1]
                vals = win[win >= 0]
                if vals.size:
                    out[i, m] = vals.mean()
                    break
    return pd.DataFrame(out, index=cities["slug"].to_numpy(), columns=range(1, 13))


def correct_all(cities: pd.DataFrame, log_path) -> None:
    start, end = climate.START_DATE, climate.END_DATE
    y0, y1 = int(start[:4]), int(end[:4])
    catalogs = [ghcn_catalog(y0, y1), gsod_catalog(y0, y1)]
    cities = cities[[climate.daily_path(s).exists() for s in cities["slug"]]]
    normals = worldclim_normals(cities)

    def work(city: pd.Series) -> dict:
        daily = climate.load_daily(city["slug"])
        if "precipitation_era5" not in daily:  # files from before this step: keep the raw values
            daily = daily.rename(columns={"precipitation_sum": "precipitation_era5",
                                          "snowfall_sum": "snowfall_era5"})
        model = daily.set_index("time")["precipitation_era5"]
        era5_mm = model.sum() / (y1 - y0 + 1)
        normal = normals.loc[city["slug"]].to_numpy()
        rejected: list[str] = []
        found = choose_gauge(city, catalogs, start, end, model, [era5_mm, normal.sum()], rejected)
        row = {"slug": city["slug"], "name": city["name"], "era5_mm": era5_mm,
               "worldclim_mm": normal.sum(), "rejected": "; ".join(rejected)}
        if found is not None:
            st, gauge, complete = found
            corrected, factors = loci(model, gauge)
            row.update(source=st["source"], station=st["id"], station_name=st["name"].strip(),
                       km=round(st["km"], 1), station_elev=st["elev"], complete=round(complete, 2),
                       gauge_mm=gauge.mean() * 365.25)
        elif not np.isnan(normal).any():
            corrected, factors = scale_to_normals(model, normal)
            row.update(source="worldclim")
        else:
            corrected, factors = model, pd.Series(1.0, index=range(1, 13))
            row.update(source="era5")
        row["corrected_mm"] = corrected.sum() / (y1 - y0 + 1)
        daily["precipitation_sum"] = corrected.round(2).to_numpy()
        snow_factor = daily["time"].dt.month.map(factors.clip(0, 5))
        daily["snowfall_sum"] = (daily["snowfall_era5"] * snow_factor).round(2)
        daily.to_csv(climate.daily_path(city["slug"]), index=False)
        return row

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(work, [c for _, c in cities.iterrows()]))
    log = pd.DataFrame(rows)
    log.round(1).to_csv(log_path, index=False)
    print(log["source"].value_counts().to_string())


if __name__ == "__main__":
    correct_all(climate.all_cities(), LOG_PATH)
    if (climate.DATA_DIR / "us_regions.csv").exists():
        correct_all(climate.load_regions(), REGIONS_LOG_PATH)
