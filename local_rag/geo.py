"""Offline country lookup from a GPS point.

Uses the Natural Earth-style country polygons in data/countries.geojson
(world.geo.json, public domain). No network call at query time.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

GEOJSON = Path(__file__).resolve().parent / "data" / "countries.geojson"


def _bbox(ring: list) -> tuple[float, float, float, float]:
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    return min(lons), min(lats), max(lons), max(lats)


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            # Avoid a zero division on a horizontal edge.
            denom = (yj - yi) or 1e-15
            x_cross = (xj - xi) * (lat - yi) / denom + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def _point_in_polygon(lon: float, lat: float, rings: list) -> bool:
    if not rings or not _point_in_ring(lon, lat, rings[0]):
        return False
    for hole in rings[1:]:
        if hole and _point_in_ring(lon, lat, hole):
            return False
    return True


@lru_cache(maxsize=1)
def _countries() -> list[dict]:
    if not GEOJSON.exists():
        raise FileNotFoundError(f"Missing country polygons: {GEOJSON}")
    collection = json.loads(GEOJSON.read_text(encoding="utf-8"))
    prepared = []
    for feature in collection["features"]:
        name = (feature.get("properties") or {}).get("name")
        geometry = feature.get("geometry") or {}
        gtype = geometry.get("type")
        coords = geometry.get("coordinates") or []
        polygons = []
        if gtype == "Polygon":
            polygons = [coords]
        elif gtype == "MultiPolygon":
            polygons = coords
        else:
            continue
        boxes = []
        for rings in polygons:
            if rings and rings[0]:
                boxes.append(_bbox(rings[0]))
        if name and polygons:
            prepared.append({"name": name, "polygons": polygons, "boxes": boxes})
    return prepared


def country_at(lat: float | None, lon: float | None) -> str | None:
    """Return the country name containing this point, or None."""
    if lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    for country in _countries():
        for box, rings in zip(country["boxes"], country["polygons"]):
            min_lon, min_lat, max_lon, max_lat = box
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                continue
            if _point_in_polygon(lon, lat, rings):
                return country["name"]
    return None
