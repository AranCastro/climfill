"""
Reanalysis data handler — daily series from the Open-Meteo archive API.
"""

import pandas as pd


def get_open_meteo_data(lat, lon, start_date, end_date, variables=None,
                        model="era5"):
    """
    Download daily reanalysis data from the Open-Meteo archive API.
    No API key needed.

    Parameters
    ----------
    lat, lon : float
        Station coordinates in decimal degrees (WGS 84, EPSG:4326)
    start_date, end_date : str
        Date range (YYYY-MM-DD)
    variables : list, optional
        Variables to fetch. Defaults to common climate vars.
    model : str or None
        Open-Meteo ``models`` value. The default ``"era5"`` requests ERA5
        (0.25 degree grid), so that the series is the product named in the
        documentation. ``None`` leaves the choice to Open-Meteo, which then
        blends several products.

    Returns
    -------
    pd.DataFrame
        Daily reanalysis data with 'date' column and prefixed variable columns.
    """
    import requests

    if variables is None:
        variables = [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "surface_pressure_mean",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
        "timezone": "auto",
    }
    if model:
        params["models"] = model

    response = requests.get(url, params=params, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        # Open-Meteo explains rejected requests in a JSON "reason" field
        try:
            reason = response.json().get("reason")
        except ValueError:
            reason = None
        if response.status_code == 429:
            reason = reason or "rate limit reached; try again in a minute"
        raise RuntimeError(
            f"Open-Meteo request failed (HTTP {response.status_code})"
            + (f": {reason}" if reason else "")
        ) from exc
    data = response.json()

    if "daily" not in data:
        raise ValueError(f"No data returned from Open-Meteo. Response: {data}")

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"})

    # Add prefix to distinguish from station data
    rename_map = {}
    for col in df.columns:
        if col != "date":
            rename_map[col] = f"reanalysis_{col}"
    df = df.rename(columns=rename_map)

    return df


# Keyword rules for matching station variables to Open-Meteo daily series.
# Rules are tried in order; the first rule whose keywords all appear in the
# normalised station column name wins.
_MATCH_RULES = [
    (("max",), ("temp", "tmax", "tx"), "temperature_2m_max"),
    (("min",), ("temp", "tmin", "tn"), "temperature_2m_min"),
    ((), ("tmax",), "temperature_2m_max"),
    ((), ("tmin",), "temperature_2m_min"),
    ((), ("temp", "tmean", "tavg", "airt"), "temperature_2m_mean"),
    ((), ("precip", "rain", "prcp", "ppt"), "precipitation_sum"),
    ((), ("humid", "rh"), "relative_humidity_2m_mean"),
    ((), ("wind", "ws"), "wind_speed_10m_max"),
    ((), ("press", "slp", "mslp"), "surface_pressure_mean"),
    ((), ("solar", "radiation", "srad", "rad", "sw"), "shortwave_radiation_sum"),
    ((), ("evapo", "et0", "pet"), "et0_fao_evapotranspiration"),
]


def suggest_reanalysis_mapping(station_cols, reanalysis_cols):
    """Guess which reanalysis column corresponds to each station variable.

    Parameters
    ----------
    station_cols : list of str
        Station variable names, e.g. ``["Temperature_Max", "Rainfall"]``.
    reanalysis_cols : list of str
        Available ``reanalysis_``-prefixed columns.

    Returns
    -------
    dict
        ``{station_col: reanalysis_col}`` for the variables that matched.
    """
    available = set(reanalysis_cols)
    mapping = {}
    for col in station_cols:
        name = "".join(ch for ch in str(col).lower() if ch.isalnum())
        for required, any_of, target in _MATCH_RULES:
            if all(k in name for k in required) and any(k in name for k in any_of):
                candidate = f"reanalysis_{target}"
                if candidate in available:
                    mapping[col] = candidate
                break
    return mapping

