"""Export monthly climate profiles for the website (docs/data/cities.json).

The site recomputes similarity in the browser, so this ships the raw monthly
profiles plus the standardization statistics climate.standardize() uses.

Usage: .venv/bin/python export_site.py
"""

import json

import numpy as np

import climate

OUT = climate.ROOT / "docs" / "data" / "cities.json"


def main() -> None:
    cities = climate.all_cities()
    profiles = climate.build_profiles(cities)
    rain = climate.pd.read_csv(climate.DATA_DIR / "rain_stations.csv").set_index("slug")
    us = set(climate.load_cities(climate.DATA_DIR / "us_cities.csv")["slug"])

    z = profiles.copy()
    for col in climate.LOG_FEATURES:
        z[col] = np.log1p(z[col])
    stats = {f: [round(float(z[f].mean()), 6), round(float(z[f].std(ddof=0)), 6)] for f in climate.FEATURES}

    out = []
    for _, c in cities.iterrows():
        if c["slug"] not in profiles.index.get_level_values("slug"):
            continue
        p = profiles.loc[c["slug"]]
        r = rain.loc[c["slug"]] if c["slug"] in rain.index else None
        out.append({
            "id": c["slug"],
            "name": c["name"],
            "country": c["country"],
            "lat": round(float(c["lat"]), 4),
            "lon": round(float(c["lon"]), 4),
            "us": c["slug"] in us,
            "rain": (str(r["station_name"]).title() if isinstance(r["station_name"], str) else "WorldClim normals")
                    if r is not None else "ERA5",
            "p": {f: [round(float(v), 2) for v in p[f]] for f in climate.FEATURES},
        })

    data = {
        "period": f"{climate.START_DATE[:4]}–{climate.END_DATE[:4]}",
        "features": list(climate.FEATURES),
        "weights": climate.FEATURES,
        "log_features": sorted(climate.LOG_FEATURES),
        "stats": stats,
        "cities": out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {len(out)} cities to {OUT.relative_to(climate.ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
