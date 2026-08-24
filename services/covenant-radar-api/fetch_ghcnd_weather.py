"""Fetch the GHCN-Daily station files behind the weather_tmax corpus.

One-shot vendoring script in the style of fetch_mckinnon.py: it runs
once, the raw files it produces are pinned by sha256 in a committed
manifest, and the script is kept so the provenance of those files is
recorded as code rather than as prose.

Station selection is MECHANICAL, so nobody hand-picked favorable
stations: from NOAA's ghcnd-inventory.txt, keep TMAX records with
firstyear <= 1950 and lastyear >= 2024, restrict to IDs starting "USW"
(US first-order/ASOS stations, the best-instrumented tier), sort by
station ID, take the first 24. The raw per-station files come from
NOAA's by_station mirror; VALUEs are TMAX in tenths of degrees C.

Outputs (under data/external/weather_tmax/):
    raw/<ID>.csv.gz     one per selected station (gitignored; ~1-3 MB each)
    raw/MANIFEST.json   the selection rule, station list, URLs, sha256s
"""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
BY_STATION_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz"

FIRST_YEAR_AT_MOST = 1950
LAST_YEAR_AT_LEAST = 2024
ID_PREFIX = "USW"
N_STATIONS = 24

OUTPUT_DIR = Path(__file__).parent / "data" / "external" / "weather_tmax" / "raw"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "cleargbm-corpus-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body: bytes = response.read()
    return body


def select_stations(inventory_text: str) -> list[str]:
    """Apply the mechanical selection rule to the inventory."""
    chosen: list[str] = []
    for line in inventory_text.splitlines():
        # Fixed-width-ish: ID LAT LON ELEMENT FIRSTYEAR LASTYEAR
        parts = line.split()
        if len(parts) != 6:
            continue
        station, _lat, _lon, element, first_year, last_year = parts
        if element != "TMAX":
            continue
        if not station.startswith(ID_PREFIX):
            continue
        if int(first_year) > FIRST_YEAR_AT_MOST or int(last_year) < LAST_YEAR_AT_LEAST:
            continue
        chosen.append(station)
    chosen.sort()
    return chosen[:N_STATIONS]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("fetching inventory...")
    inventory = _fetch(INVENTORY_URL).decode("utf-8")
    stations = select_stations(inventory)
    print(f"selected {len(stations)} stations: {stations[0]}..{stations[-1]}")

    files: list[dict[str, object]] = []
    for station in stations:
        url = BY_STATION_URL.format(station=station)
        body = _fetch(url)
        # Sanity: the payload must be a real gzip holding this station's rows.
        head = gzip.decompress(body)[:200]
        if station.encode("utf-8") not in head:
            raise RuntimeError(f"{station}: payload does not open with the station's own rows")
        out_path = OUTPUT_DIR / f"{station}.csv.gz"
        out_path.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        files.append({"station": station, "url": url, "sha256": digest, "bytes": len(body)})
        print(f"  {station}: {len(body)} bytes sha256={digest[:16]}...")

    manifest = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "selection_rule": (
            f"ghcnd-inventory.txt TMAX rows with firstyear<={FIRST_YEAR_AT_MOST}, "
            f"lastyear>={LAST_YEAR_AT_LEAST}, id prefix {ID_PREFIX!r}, sorted by id, "
            f"first {N_STATIONS}"
        ),
        "inventory_url": INVENTORY_URL,
        "value_units": "TMAX tenths of degrees C",
        "files": files,
    }
    (OUTPUT_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {OUTPUT_DIR / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
