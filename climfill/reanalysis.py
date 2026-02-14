"""
Reanalysis data handler — Open-Meteo (free) and ERA5 (CDS API).
"""

import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


def get_open_meteo_data(lat, lon, start_date, end_date, variables=None):
    """
    Download free weather data from Open-Meteo archive API.
    No API key needed.

    Parameters
    ----------
    lat, lon : float
        Station coordinates
    start_date, end_date : str
        Date range (YYYY-MM-DD)
    variables : list, optional
        Variables to fetch. Defaults to common climate vars.

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

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
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


# ERA5 variable mapping
ERA5_VARIABLE_MAP = {
    "temperature": "2m_temperature",
    "temp": "2m_temperature",
    "tmax": "maximum_2m_temperature_since_previous_post_processing",
    "tmin": "minimum_2m_temperature_since_previous_post_processing",
    "precipitation": "total_precipitation",
    "precip": "total_precipitation",
    "humidity": "2m_dewpoint_temperature",
    "wind_speed": "10m_u_component_of_wind",
    "pressure": "surface_pressure",
    "solar": "surface_solar_radiation_downwards",
    "sst": "sea_surface_temperature",
}
