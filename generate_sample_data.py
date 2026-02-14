#!/usr/bin/env python3
"""Generate realistic sample climate station data with controlled gaps."""

import numpy as np
import pandas as pd


def generate_sample_station(
    start_date="2019-01-01",
    end_date="2023-12-31",
    gap_pct=0.03,
    n_sensor_failures=8,
    n_maintenance=2,
    seed=42,
):
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    doy = dates.dayofyear

    # Temperature with seasonal cycle + autocorrelation
    temp = 28 + 6 * np.sin(2 * np.pi * (doy - 105) / 365)
    noise = np.random.normal(0, 1.8, n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + 0.4 * noise[i]
    temp += noise

    tmax = temp + np.abs(np.random.normal(3, 1, n))
    tmin = temp - np.abs(np.random.normal(3, 1, n))

    # Precipitation
    prob = 0.25 + 0.2 * np.sin(2 * np.pi * (doy - 200) / 365)
    occur = np.random.random(n) < prob
    precip = np.maximum(0, np.random.exponential(8, n) * occur)

    # Humidity
    humidity = 62 + 18 * np.sin(2 * np.pi * (doy - 210) / 365)
    humidity += np.random.normal(0, 6, n) + precip * 0.3
    humidity = np.clip(humidity, 20, 100)

    # Wind
    wind = 4 + 2.5 * np.cos(2 * np.pi * doy / 365) + np.random.exponential(1.2, n)
    wind = np.maximum(0, wind)

    # Solar radiation
    solar = 18 + 6 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 3, n)
    solar *= 1 - 0.5 * occur
    solar = np.maximum(0, solar)

    # Pressure
    pressure = np.full(n, 1013.25)
    for i in range(1, n):
        pressure[i] = 0.85 * pressure[i - 1] + 0.15 * (1013.25 + np.random.normal(0, 3))

    df = pd.DataFrame({
        "Date": dates,
        "Temperature_Mean": np.round(temp, 1),
        "Temperature_Max": np.round(tmax, 1),
        "Temperature_Min": np.round(tmin, 1),
        "Precipitation": np.round(precip, 1),
        "Relative_Humidity": np.round(humidity, 1),
        "Wind_Speed": np.round(wind, 1),
        "Solar_Radiation": np.round(solar, 1),
        "Pressure": np.round(pressure, 1),
    })

    variables = list(df.columns[1:])

    # Random single-point gaps
    for col in variables:
        mask = np.random.random(n) < gap_pct
        df.loc[mask, col] = np.nan

    # Sensor failures (5-15 days)
    for _ in range(n_sensor_failures):
        start = np.random.randint(0, n - 20)
        length = np.random.randint(5, 15)
        cols = np.random.choice(variables, size=np.random.randint(1, 3), replace=False)
        for col in cols:
            df.loc[start : start + length, col] = np.nan

    # Maintenance outages (20-50 days, all variables)
    for _ in range(n_maintenance):
        start = np.random.randint(0, n - 60)
        length = np.random.randint(20, 50)
        for col in variables:
            df.loc[start : start + length, col] = np.nan

    # Sentinel values
    for _ in range(10):
        idx = np.random.randint(0, n)
        col = np.random.choice(variables)
        df.loc[idx, col] = np.random.choice([-999, -9999])

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="sample_station_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_sample_station(seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} records → {args.output}")
    for col in df.columns[1:]:
        missing = df[col].isna().sum()
        pct = missing / len(df) * 100
        print(f"  {col}: {missing} missing ({pct:.1f}%)")
