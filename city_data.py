"""
City-level housing data for college towns.
==========================================
Person B will expand this with real data feeds or a larger dataset.
For now, these are reasonable 2024-2025 estimates sourced from
Zillow, Apartment List, and local MLS averages.

Each city dict contains:
    name              : Display name
    state             : Two-letter state code
    median_home_price : Median single-family home price ($)
    median_rent       : Median monthly rent for a comparable unit ($)
    property_tax_rate : Effective annual property tax rate (decimal)
    insurance_rate    : Annual homeowner's insurance as % of home value
    notes             : Any quirks about the local market
"""

CITIES: dict[str, dict] = {
    "state_college_pa": {
        "name": "State College, PA",
        "state": "PA",
        "median_home_price": 340_000,
        "median_rent": 1_500,
        "property_tax_rate": 0.0195,
        "insurance_rate": 0.004,
        "notes": "Penn State. Tight rental market near campus inflates rents.",
    },
    "ann_arbor_mi": {
        "name": "Ann Arbor, MI",
        "state": "MI",
        "median_home_price": 450_000,
        "median_rent": 1_800,
        "property_tax_rate": 0.0180,
        "insurance_rate": 0.004,
        "notes": "University of Michigan. High demand pushes prices above MI averages.",
    },
    "chapel_hill_nc": {
        "name": "Chapel Hill, NC",
        "state": "NC",
        "median_home_price": 520_000,
        "median_rent": 1_650,
        "property_tax_rate": 0.0105,
        "insurance_rate": 0.005,
        "notes": "UNC. Research Triangle proximity adds price premium.",
    },
    "ames_ia": {
        "name": "Ames, IA",
        "state": "IA",
        "median_home_price": 250_000,
        "median_rent": 1_100,
        "property_tax_rate": 0.0155,
        "insurance_rate": 0.004,
        "notes": "Iowa State. Affordable market relative to peers.",
    },
    "austin_tx": {
        "name": "Austin, TX",
        "state": "TX",
        "median_home_price": 540_000,
        "median_rent": 1_750,
        "property_tax_rate": 0.0180,
        "insurance_rate": 0.006,
        "notes": "UT Austin. No state income tax but high property taxes.",
    },
    "madison_wi": {
        "name": "Madison, WI",
        "state": "WI",
        "median_home_price": 380_000,
        "median_rent": 1_450,
        "property_tax_rate": 0.0195,
        "insurance_rate": 0.004,
        "notes": "UW-Madison. Isthmus geography constrains supply.",
    },
    "boulder_co": {
        "name": "Boulder, CO",
        "state": "CO",
        "median_home_price": 780_000,
        "median_rent": 2_200,
        "property_tax_rate": 0.0060,
        "insurance_rate": 0.005,
        "notes": "CU Boulder. Growth boundary keeps prices high.",
    },
    "tucson_az": {
        "name": "Tucson, AZ",
        "state": "AZ",
        "median_home_price": 310_000,
        "median_rent": 1_200,
        "property_tax_rate": 0.0100,
        "insurance_rate": 0.004,
        "notes": "University of Arizona. Affordable Sun Belt market.",
    },
    # ── Major cities (where rent vs buy actually matters) ──
    "new_york_ny": {
        "name": "New York City, NY",
        "state": "NY",
        "median_home_price": 708_000,
        "median_rent": 3_200,
        "property_tax_rate": 0.0120,
        "insurance_rate": 0.005,
        "notes": "Median condo/co-op. Renting is the default — does buying ever win?",
    },
    "san_francisco_ca": {
        "name": "San Francisco, CA",
        "state": "CA",
        "median_home_price": 1_305_000,
        "median_rent": 3_500,
        "property_tax_rate": 0.0073,
        "insurance_rate": 0.004,
        "notes": "Prop 13 keeps property tax low. Prices dipped slightly in 2025.",
    },
    "denver_co": {
        "name": "Denver, CO",
        "state": "CO",
        "median_home_price": 570_000,
        "median_rent": 1_900,
        "property_tax_rate": 0.0055,
        "insurance_rate": 0.005,
        "notes": "Prices declined 3.2% in 2025. Market cooling from pandemic peak.",
    },
    "chicago_il": {
        "name": "Chicago, IL",
        "state": "IL",
        "median_home_price": 336_000,
        "median_rent": 1_700,
        "property_tax_rate": 0.0210,
        "insurance_rate": 0.005,
        "notes": "Affordable vs coasts but high property taxes. Led price gains in 2025.",
    },
    "phoenix_az": {
        "name": "Phoenix, AZ",
        "state": "AZ",
        "median_home_price": 420_000,
        "median_rent": 1_550,
        "property_tax_rate": 0.0065,
        "insurance_rate": 0.004,
        "notes": "Sun Belt. Prices declined 2.3% in 2025 after pandemic surge.",
    },
    "miami_fl": {
        "name": "Miami, FL",
        "state": "FL",
        "median_home_price": 580_000,
        "median_rent": 2_650,
        "property_tax_rate": 0.0100,
        "insurance_rate": 0.012,
        "notes": "No state income tax but very high insurance costs. Rents up 53% since 2020.",
    },
}


def get_city(city_key: str) -> dict:
    """Retrieve city data by key. Raises KeyError if not found."""
    return CITIES[city_key]


def list_cities() -> list[dict]:
    """Return list of {key, name} for all available cities."""
    return [{"key": k, "name": v["name"]} for k, v in CITIES.items()]