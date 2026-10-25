# Weather twins (prototype)

Find the famous world cities whose typical year of weather is most like your US city's.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy matplotlib jupyter omfiles fsspec s3fs tifffile
.venv/bin/python era5.py           # bulk-reads 10 years of hourly ERA5 data into data/daily/ (~1.5 h for 268 cities)
.venv/bin/python rainfall.py       # bias-corrects rain/snow against nearby rain gauges (~15 min)
.venv/bin/jupyter lab explore.ipynb
```

## Adding cities

Add a row with `name,country,cc` to `data/us_cities.csv` or `data/world_cities.csv` (`cc` = ISO country code; optional `query` overrides the search name, `"Name|Region"` pins the admin region), then:

```bash
.venv/bin/python geocode.py        # fills lat/lon/elevation
.venv/bin/python era5.py           # fetches only cities without data
.venv/bin/python rainfall.py       # re-runs the rain correction (cached gauges are reused)
```

## Files

- `data/us_cities.csv`, `data/world_cities.csv` — the cities, with coordinates and elevation.
- `geocode.py` — fills in coordinates via the Open-Meteo Geocoding API (GeoNames).
- `era5.py` — bulk ERA5 / ERA5-Land reader from Open-Meteo's AWS open data (`s3://openmeteo`, no API limits).
- `rainfall.py` — rain-gauge bias correction of ERA5 precipitation; choices logged in `data/rain_stations.csv`.
- `climate.py` — monthly profiles, similarity (`find_similar`) and default feature weights (`FEATURES`).
- `explore.ipynb` — results tables, match explanations, side-by-side climate charts.
- `data/daily/` — one daily CSV per city; `_cells.csv` records which grid cells were used.

## Method

- 2016–2025 hourly reanalysis, aggregated to local days. Temperature and dew point from ERA5-Land (0.1°, nearest land cell); rain, snow, cloud and wind from ERA5 (0.25°). Temperatures are corrected for grid-vs-city elevation (6.5 °C/km).
- ERA5 rain drizzles too often and over-rains near mountains, so it is corrected per calendar month with local intensity scaling (LOCI) against the nearest trustworthy rain gauge: NOAA GHCN-Daily first, then GSOD, WMO stations preferred, within 50 km and 500 m of elevation. Gauges disagreeing >2x with both ERA5 and WorldClim are rejected; cities without a usable gauge (26 of 268) are scaled to WorldClim 2.1 monthly normals. Snowfall gets the same monthly factor.
- Each city becomes a "typical year": 12 monthly averages of high, low, dew point, rain total, rainy days, cloud cover, snowfall, wind and day-to-day swing in highs.
- Features are z-scored over all city-months, weighted, and compared month by month (weighted RMS). Lower distance = more similar.
- `seasons=True` shifts Southern Hemisphere cities by 6 months so seasons line up.

Data: Copernicus ERA5 / ERA5-Land via Open-Meteo (CC BY 4.0); NOAA GHCN-Daily and GSOD; WorldClim 2.1 (Fick & Hijmans 2017).
