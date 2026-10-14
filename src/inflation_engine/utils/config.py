COUNTRIES = {
    "US": {
        "name": "United States",
        "region": "North America",
        "source": "fred",
    },
    # US PCE price indexes (BEA) under their own geography code, so they sit
    # beside US CPI in the panel without ever sharing its unique key.
    "US_PCE": {
        "name": "United States (PCE)",
        "region": "North America",
        "source": "bea",
    },
    "GB": {
        "name": "United Kingdom",
        "region": "Europe",
        "source": "ons",
    },
    "DE": {
        "name": "Germany",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "FR": {
        "name": "France",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "IT": {
        "name": "Italy",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "ES": {
        "name": "Spain",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "AT": {
        "name": "Austria",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "BE": {
        "name": "Belgium",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "CY": {
        "name": "Cyprus",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "EE": {
        "name": "Estonia",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "FI": {
        "name": "Finland",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "GR": {
        "name": "Greece",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "HR": {
        "name": "Croatia",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "IE": {
        "name": "Ireland",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "LT": {
        "name": "Lithuania",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "LU": {
        "name": "Luxembourg",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "LV": {
        "name": "Latvia",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "MT": {
        "name": "Malta",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "NL": {
        "name": "Netherlands",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "PT": {
        "name": "Portugal",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "SI": {
        "name": "Slovenia",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "SK": {
        "name": "Slovakia",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "EA20": {
        "name": "Euro Area",
        "region": "Euro Area",
        "source": "eurostat",
    },
    "DK": {
        "name": "Denmark",
        "region": "Nordics",
        "source": "statbank",
    },
    "NO": {
        "name": "Norway",
        "region": "Nordics",
        "source": "ssb",
    },
    "SE": {
        "name": "Sweden",
        "region": "Nordics",
        "source": "scb",
    },
}

# Geography codes that are an alternative price measure for a country
# already in the panel rather than a country of their own; excluded from
# cross-country rankings so they never displace a real country's rank.
ALTERNATIVE_MEASURE_CODES = frozenset({"US_PCE"})

PRIMARY_SERIES = {
    "AT": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "BE": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "CY": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "DE": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "EE": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "FI": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "GR": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "HR": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "IE": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "LT": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "LU": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "LV": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "MT": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "NL": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "PT": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "SI": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "SK": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "DK": {
        "headline": "all_items",
        "core": "all_items_less_energy_unprocessed_food",
    },
    "EA20": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "ES": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "FR": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "GB": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "IT": {
        "headline": "all_items",
        "core": "all_items_less_food_energy_alcohol_tobacco",
    },
    "NO": {
        "headline": "all_items",
        "core": "cpi_adjusted_taxes_excluding_energy",
    },
    "SE": {
        "headline": "cpif_fixed_interest_rate",
        "core": "cpif_fixed_interest_rate_excluding_energy",
    },
    "US": {
        "headline": "all_items",
        "core": "all_items_less_food_and_energy",
    },
    "US_PCE": {
        "headline": "all_items",
        "core": "all_items_less_food_and_energy",
    },
}


def primary_series_keys() -> list[tuple[str, str, str]]:
    """Return canonical country, type, and component keys for report series."""
    return [
        (country_code, series_type, component)
        for country_code, series in PRIMARY_SERIES.items()
        for series_type, component in series.items()
    ]


def series_role(country_code: str, series_type: str, component: str) -> str:
    """Classify canonical headline/core indexes separately from components."""
    primary_component = PRIMARY_SERIES.get(country_code, {}).get(series_type)
    return "aggregate" if component == primary_component else "component"
