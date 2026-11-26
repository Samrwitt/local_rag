"""Turn OCR and document text into place and trip signals.

Raw OCR is not stored alone. Cafe names, street lines, city names, and
document cues are interpreted into country, city, language, and roles
(receipt, hotel, booking, memo) so a query like "my trip to Italy in 2019"
can find related photos, videos, receipts, and notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from local_rag.evidence import attribute, pack_attributes

# City → country. Enough for travel demos; extend as the corpus grows.
CITY_COUNTRY = {
    "rome": "Italy",
    "roma": "Italy",
    "milan": "Italy",
    "milano": "Italy",
    "florence": "Italy",
    "firenze": "Italy",
    "venice": "Italy",
    "venezia": "Italy",
    "naples": "Italy",
    "napoli": "Italy",
    "turin": "Italy",
    "torino": "Italy",
    "bologna": "Italy",
    "genoa": "Italy",
    "genova": "Italy",
    "pisa": "Italy",
    "verona": "Italy",
    "paris": "France",
    "lyon": "France",
    "marseille": "France",
    "nice": "France",
    "london": "United Kingdom",
    "manchester": "United Kingdom",
    "edinburgh": "United Kingdom",
    "berlin": "Germany",
    "munich": "Germany",
    "münchen": "Germany",
    "hamburg": "Germany",
    "frankfurt": "Germany",
    "miltenberg": "Germany",
    "musterstadt": "Germany",
    "madrid": "Spain",
    "barcelona": "Spain",
    "lisbon": "Portugal",
    "lisboa": "Portugal",
    "amsterdam": "Netherlands",
    "brussels": "Belgium",
    "bruxelles": "Belgium",
    "vienna": "Austria",
    "wien": "Austria",
    "zurich": "Switzerland",
    "zürich": "Switzerland",
    "geneva": "Switzerland",
    "athens": "Greece",
    "istanbul": "Turkey",
    "cairo": "Egypt",
    "addis ababa": "Ethiopia",
    "addis": "Ethiopia",
    "nairobi": "Kenya",
    "lagos": "Nigeria",
    "new york": "United States of America",
    "los angeles": "United States of America",
    "san francisco": "United States of America",
    "chicago": "United States of America",
    "boston": "United States of America",
    "tokyo": "Japan",
    "osaka": "Japan",
    "seoul": "South Korea",
    "beijing": "China",
    "shanghai": "China",
    "hong kong": "China",
    "singapore": "Singapore",
    "bangkok": "Thailand",
    "dubai": "United Arab Emirates",
    "sydney": "Australia",
    "melbourne": "Australia",
    "toronto": "Canada",
    "vancouver": "Canada",
    "montreal": "Canada",
}

# Cities that are also English words / common names — only trust path or travel context.
AMBIGUOUS_CITIES = {
    "nice",
    "florence",
    "firenze",
    "mobile",
    "durham",
    "reading",
    "bath",
    "orange",
    "buffalo",
    "springfield",
}

COUNTRY_ALIASES = {
    "italy": "Italy",
    "italia": "Italy",
    "france": "France",
    "germany": "Germany",
    "deutschland": "Germany",
    "spain": "Spain",
    "españa": "Spain",
    "portugal": "Portugal",
    "uk": "United Kingdom",
    "england": "United Kingdom",
    "britain": "United Kingdom",
    "united kingdom": "United Kingdom",
    "usa": "United States of America",
    "united states": "United States of America",
    "ethiopia": "Ethiopia",
    "kenya": "Kenya",
    "japan": "Japan",
    "china": "China",
    "turkey": "Turkey",
    "greece": "Greece",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "belgium": "Belgium",
    "austria": "Austria",
    "switzerland": "Switzerland",
    "egypt": "Egypt",
    "thailand": "Thailand",
    "australia": "Australia",
    "canada": "Canada",
    "uae": "United Arab Emirates",
}

# Used only when parsing the user's question, not document text.
QUERY_COUNTRY_ALIASES = {
    **COUNTRY_ALIASES,
    "italian": "Italy",
    "french": "France",
    "german": "Germany",
    "spanish": "Spain",
    "portuguese": "Portugal",
    "american": "United States of America",
    "america": "United States of America",
}

STREET_HINTS = (
    r"\bvia\b",
    r"\bviale\b",
    r"\bpiazza\b",
    r"\brue\b",
    r"\bavenue\b",
    r"\bave\.?\b",
    r"\bstreet\b",
    r"\bst\.?\b",
    r"\broad\b",
    r"\brd\.?\b",
    r"\bstrasse\b",
    r"\bstraße\b",
    r"\bstr\.?\b",
    r"\bcalle\b",
    r"\bcorso\b",
    r"\blane\b",
    r"\bboulevard\b",
    r"\bblvd\.?\b",
)

VENUE_PATTERNS = [
    (r"\b(cafe|café|caffe|coffee|trattoria|osteria|pizzeria|bistro|bakery)\b", "cafe"),
    (r"\b(hotel|hostel|resort|bnb|bed\s*&\s*breakfast)\b", "hotel"),
    (r"\b(restaurant|ristorante|diner)\b", "restaurant"),
    (r"\b(museum|galleria|gallery|cathedral|basilica)\b", "attraction"),
    (r"\b(airport|terminal)\b", "transit"),
]

ROLE_PATTERNS = [
    (r"\b(invoice|receipt|vat\s*no|total\s*(due|amount)|gross\s*amount)\b", "receipt"),
    (r"\b(booking|reservation|confirmation\s*number|itinerary|boarding\s*pass)\b", "booking"),
    (r"\b(check[- ]?in|check[- ]?out|room\s*rate|hotel\s*(bill|folio|invoice))\b", "hotel"),
    # Avoid "journal" — academic papers match it constantly.
    (r"\b(travel\s*diary|trip\s*(memo|note|journal)|vacation\s*journal)\b", "memo"),
]

# Bibliography / paper locations are not trip evidence.
CITATION_NEAR = re.compile(
    r"\b(proceedings|proc\.|pp\.|arxiv|doi|vol\.|conference|workshop|symposium|"
    r"association for|acl|neurips|icml|cvpr|ieee|acm|springer)\b",
    re.I,
)

TRAVEL_NEAR = re.compile(
    r"\b(trip|travel|vacation|holiday|hotel|hostel|airport|flight|passport|"
    r"itinerary|booking|check[- ]?in|souvenir|tour|visited|visiting)\b",
    re.I,
)

LANG_HINTS = [
    # Avoid bare "via" — it is common English for "through".
    ("Italian", (r"\b(piazza|ristorante|grazie|prego|albergo|caffe|espresso)\b", r"\bvia\s+[A-ZÀ-Ö]")),
    ("French", (r"\b(rue|merci|hôtel|bonjour)\b",)),
    ("German", (r"\b(strasse|straße|rechnung|danke|bitte|gmbh)\b",)),
    ("Spanish", (r"\b(calle|gracias|factura|españa)\b",)),
    ("English", (r"\b(invoice|receipt|booking|confirmation|check-in|reservation)\b",)),
]


@dataclass
class PlaceInfo:
    country: str | None = None
    city: str | None = None
    language: str | None = None
    venues: list[str] = field(default_factory=list)
    streets: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    place_source: str | None = None  # gps | text | both | path
    summary_lines: list[str] = field(default_factory=list)
    enrichment: dict = field(default_factory=dict)


def _searchable(text: str) -> str:
    """Treat underscores and dashes like spaces so Italy_2019_rome matches."""
    return re.sub(r"[_\-.]+", " ", (text or "").lower())


def interpret_place(
    text: str,
    *,
    path_name: str = "",
    gps_country: str | None = None,
) -> PlaceInfo:
    """Interpret OCR/document text into place and trip metadata."""
    info = PlaceInfo()
    path_lower = _searchable(path_name)
    body = text or ""
    lower = _searchable(body)
    # Prefer path + body, but score path cues separately so filenames win.
    blob_lower = f"{path_lower}\n{lower}".strip()

    if gps_country:
        info.country = gps_country
        info.place_source = "gps"

    city = _find_city(path_lower, prefer_path=True) or _find_trusted_city(lower, path_lower)
    if city:
        info.city = _pretty_city(city)
        mapped = CITY_COUNTRY.get(city)
        if mapped and not info.country:
            info.country = mapped
            info.place_source = "path" if city in path_lower else "text"
        elif mapped and info.country == mapped:
            info.place_source = "both" if info.place_source == "gps" else info.place_source

    if not info.country:
        country = _find_country(path_lower) or _find_trusted_country(lower)
        if country:
            info.country = country
            info.place_source = info.place_source or ("path" if _find_country(path_lower) else "text")

    info.streets = _find_streets(body)
    for pattern, label in VENUE_PATTERNS:
        if re.search(pattern, blob_lower, re.I):
            # Skip lone hotel/transit hits that only appear in citations.
            if label in {"hotel", "transit"} and _mostly_citation(lower) and label not in path_lower:
                continue
            if label not in info.venues:
                info.venues.append(label)

    for pattern, role in ROLE_PATTERNS:
        if re.search(pattern, blob_lower, re.I) and role not in info.roles:
            if role in {"hotel", "memo"} and _mostly_citation(lower) and role not in path_lower:
                continue
            # Bare "hotel" in long papers is weak; need travel/path support.
            if role == "hotel" and "hotel" not in path_lower and not TRAVEL_NEAR.search(lower):
                if not re.search(r"\b(check[- ]?in|check[- ]?out|room\s*rate|hotel\s*(bill|folio|invoice))\b", lower):
                    continue
            info.roles.append(role)

    if re.search(r"\b(trip|travel|vacation|holiday)\b", path_lower) and "memo" not in info.roles:
        info.roles.append("memo")
    if "receipt" in path_lower and "receipt" not in info.roles:
        info.roles.append("receipt")
    if "booking" in path_lower and "booking" not in info.roles:
        info.roles.append("booking")
    if "hotel" in path_lower and "hotel" not in info.roles:
        info.roles.append("hotel")

    # Academic PDFs with only a conference city should not become trip docs.
    if info.place_source == "text" and _mostly_citation(lower) and not info.streets and not info.roles:
        if not any(city_name in path_lower for city_name in CITY_COUNTRY):
            info.country = gps_country
            info.city = None
            info.venues = []
            info.language = None
            info.place_source = "gps" if gps_country else None

    info.language = _guess_language(lower, info.country)
    info.summary_lines = _summary(info)
    info.enrichment = _enrichment_bundle(info, path_name=path_name, gps_country=gps_country)
    return info


def _enrichment_bundle(info: PlaceInfo, *, path_name: str, gps_country: str | None) -> dict:
    """Attach evidence, method, confidence, and observed/inferred flags."""
    attrs = []
    if info.country:
        observed = info.place_source == "gps" or (gps_country and info.country == gps_country)
        conf = 0.92 if info.place_source in {"gps", "both"} else 0.72 if info.place_source == "path" else 0.58
        attrs.append(
            attribute(
                "country",
                info.country,
                method=f"place:{info.place_source or 'text'}",
                confidence=conf,
                observed=bool(observed),
                evidence=path_name or (info.streets[0] if info.streets else ""),
                alternatives=[],
            )
        )
    if info.city:
        conf = 0.8 if info.place_source in {"path", "both", "gps"} else 0.55
        attrs.append(
            attribute(
                "city",
                info.city,
                method=f"place:{info.place_source or 'text'}",
                confidence=conf,
                observed=False,
                evidence=path_name,
            )
        )
    if info.language:
        # Country→language defaults are inferred and weaker than lexical cues.
        blob = _searchable(f"{path_name}\n" + " ".join(info.summary_lines))
        lexical_hits = 0
        for name, patterns in LANG_HINTS:
            if name != info.language:
                continue
            lexical_hits = sum(1 for pat in patterns if re.search(pat, blob, re.I))
        from_country_only = lexical_hits == 0
        attrs.append(
            attribute(
                "language",
                info.language,
                method="lang_hints" if not from_country_only else "country_default",
                confidence=0.45 if from_country_only else 0.65,
                observed=False,
                evidence="language cues in text/path",
            )
        )
    for role in info.roles:
        attrs.append(
            attribute(
                f"role:{role}",
                role,
                method="role_pattern",
                confidence=0.7,
                observed=True,
                evidence=path_name,
            )
        )
    for street in info.streets[:3]:
        attrs.append(
            attribute(
                "street",
                street,
                method="street_pattern",
                confidence=0.75,
                observed=True,
                evidence=street,
            )
        )
    return {
        "attributes": pack_attributes(attrs),
        "place_source": info.place_source,
        "venues": list(info.venues),
        "streets": list(info.streets),
    }


def parse_trip_query(question: str) -> dict:
    """Pull year / country / city hints from a natural trip question."""
    lower = _searchable(question or "")
    year = None
    match = re.search(r"\b((?:19|20)\d{2})\b", lower)
    if match:
        year = match.group(1)
    country = None
    for alias, name in QUERY_COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            country = name
            break
    city = _find_city(lower, prefer_path=False)
    trip = bool(re.search(r"\b(trip|travel|vacation|holiday|visit|itinerary)\b", lower))
    return {
        "year": year,
        "country": country,
        "city": _pretty_city(city) if city else None,
        "trip": trip,
    }


def _pretty_city(city: str) -> str:
    if city in {"addis ababa", "addis"}:
        return "Addis Ababa"
    if city == "roma":
        return "Rome"
    return city.title()


def _find_city(lower: str, *, prefer_path: bool) -> str | None:
    best = None
    best_pos = None
    for city in sorted(CITY_COUNTRY, key=len, reverse=True):
        if not prefer_path and city in AMBIGUOUS_CITIES:
            continue
        match = re.search(rf"\b{re.escape(city)}\b", lower)
        if not match:
            continue
        if best_pos is None or match.start() < best_pos:
            best = city
            best_pos = match.start()
    return best


def _find_trusted_city(lower: str, path_lower: str) -> str | None:
    """City from document body.

    Unique multi-word cities (Addis Ababa, New York) are trusted when not in a
    citation. Ambiguous single words (Nice, Florence) need travel context.
    """
    has_travel = bool(TRAVEL_NEAR.search(lower) or _find_streets_quick(lower))
    best = None
    best_pos = None
    for city in sorted(CITY_COUNTRY, key=len, reverse=True):
        ambiguous = city in AMBIGUOUS_CITIES or " " not in city
        if ambiguous and not has_travel and city not in path_lower:
            continue
        for match in re.finditer(rf"\b{re.escape(city)}\b", lower):
            window = lower[max(0, match.start() - 80) : match.end() + 80]
            if CITATION_NEAR.search(window):
                continue
            if city in AMBIGUOUS_CITIES and not TRAVEL_NEAR.search(window):
                continue
            if best_pos is None or match.start() < best_pos:
                best = city
                best_pos = match.start()
            break
    return best


def _find_streets_quick(lower: str) -> bool:
    return any(re.search(pat, lower, re.I) for pat in STREET_HINTS)


def _find_country(lower: str) -> str | None:
    for alias, country in sorted(COUNTRY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return country
    return None


def _find_trusted_country(lower: str) -> str | None:
    # Prefer country that accompanies a trusted city; otherwise need travel cues
    # or a clear non-citation country mention paired with a known city name.
    has_travel = bool(TRAVEL_NEAR.search(lower))
    for alias, country in sorted(COUNTRY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(rf"\b{re.escape(alias)}\b", lower):
            window = lower[max(0, match.start() - 80) : match.end() + 80]
            if CITATION_NEAR.search(window):
                continue
            if has_travel or _city_for_country(lower, country):
                return country
    return None


def _city_for_country(lower: str, country: str) -> bool:
    for city, mapped in CITY_COUNTRY.items():
        if mapped == country and " " in city and re.search(rf"\b{re.escape(city)}\b", lower):
            return True
    return False


def _mostly_citation(lower: str) -> bool:
    if not lower:
        return False
    hits = len(CITATION_NEAR.findall(lower))
    # Dense citation markers ⇒ academic paper, not a trip memo.
    return hits >= 3


def _find_streets(text: str) -> list[str]:
    streets = []
    for line in (text or "").splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip(" ,.;")
        if len(cleaned) < 6 or len(cleaned) > 60:
            continue
        if not any(re.search(pat, cleaned, re.I) for pat in STREET_HINTS):
            continue
        # Prefer address-like lines: street word near a number, not full sentences.
        if not re.search(r"\d", cleaned):
            continue
        if cleaned.count(" ") > 8:
            continue
        if cleaned not in streets:
            streets.append(cleaned)
        if len(streets) >= 5:
            break
    return streets


def _guess_language(lower: str, country: str | None) -> str | None:
    scores = []
    for name, patterns in LANG_HINTS:
        hits = sum(1 for pat in patterns if re.search(pat, lower, re.I))
        if hits:
            scores.append((hits, name))
    if scores:
        scores.sort(reverse=True)
        # Need a clear cue; a single shared word is not enough.
        if scores[0][0] >= 1 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            return scores[0][1]
    if country == "Italy":
        return "Italian"
    if country == "France":
        return "French"
    if country == "Germany":
        return "German"
    if country == "Spain":
        return "Spanish"
    if country in {"United Kingdom", "United States of America", "Canada", "Australia"}:
        return "English"
    return None


def _summary(info: PlaceInfo) -> list[str]:
    lines = []
    if info.country:
        lines.append(f"Country: {info.country}")
    if info.city:
        lines.append(f"City: {info.city}")
    if info.language:
        lines.append(f"Language cues: {info.language}")
    if info.venues:
        lines.append("Venues: " + ", ".join(info.venues))
    if info.streets:
        lines.append("Streets: " + "; ".join(info.streets[:3]))
    if info.roles:
        lines.append("Document roles: " + ", ".join(info.roles))
    if info.place_source:
        lines.append(f"Place from: {info.place_source}")
    return lines
