import requests
import pandas as pd
import numpy as np

from data_pipeline import add_cyclical_time_features, add_pv_physics_features


def fetch_weather_forecast(latitude, longitude, days=2, past_days=2):
    """
    Pulls hourly forecast from Open-Meteo (free, no API key required).
    past_days includes recent actual weather before "now" in the same
    response, which is required so lag features (e.g. same time yesterday)
    have real values instead of being empty for the first day of data.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation",
        "forecast_days": days,
        "past_days": past_days,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "outdoor_temp_C": data["temperature_2m"],
        "humidity_pct": data["relative_humidity_2m"],
        "solar_irradiance_Wm2": data["shortwave_radiation"],
    })
    return df


def fetch_weather_display(latitude, longitude, days=3):
    """
    Pulls a richer hourly weather set purely for display on the Weather
    Forecast page (adds cloud cover and precipitation, which the ML models
    never see -- this is display-only, no risk of feature mismatch).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,cloudcover,precipitation",
        "forecast_days": days,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "outdoor_temp_C": data["temperature_2m"],
        "humidity_pct": data["relative_humidity_2m"],
        "solar_irradiance_Wm2": data["shortwave_radiation"],
        "cloudcover_pct": data["cloudcover"],
        "precipitation_mm": data["precipitation"],
    })
    return df


def summarize_weather_by_day(df):
    """
    Collapses hourly weather into per-day summary cards: temp range,
    avg cloud cover, total rainfall, peak solar irradiance.
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    summaries = []
    for i, (date, group) in enumerate(df.groupby("date", sort=True)):
        summaries.append({
            "date": str(date),
            "label": "Today" if i == 0 else ("Tomorrow" if i == 1 else f"Day {i + 1}"),
            "min_temp_C": round(float(group["outdoor_temp_C"].min()), 1),
            "max_temp_C": round(float(group["outdoor_temp_C"].max()), 1),
            "avg_cloudcover_pct": round(float(group["cloudcover_pct"].mean()), 0),
            "total_precipitation_mm": round(float(group["precipitation_mm"].sum()), 1),
            "peak_irradiance_Wm2": round(float(group["solar_irradiance_Wm2"].max()), 0),
        })
    return summaries


def fetch_historical_weather(latitude, longitude, start_date, end_date):
    """
    Pulls actual historical weather from Open-Meteo's archive API.
    Useful for backtesting the app against a real past date without
    needing the original Kaggle CSV at all. Dates in YYYY-MM-DD format.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "outdoor_temp_C": data["temperature_2m"],
        "humidity_pct": data["relative_humidity_2m"],
        "solar_irradiance_Wm2": data["shortwave_radiation"],
    })
    return df


def upsample_to_15min(df_hourly):
    """Interpolates hourly weather down to 15 minute resolution."""
    df_hourly = df_hourly.set_index("timestamp")
    df_15min = df_hourly.resample("15min").interpolate(method="linear")
    return df_15min.reset_index()


def add_calendar_columns(df):
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["minute_of_hour"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_forecast_row_set(latitude, longitude, pv_capacity_kW, days=1, historical_range=None):
    """
    Full pipeline: fetch weather (forecast or historical archive), add
    calendar + PV physics features. Uses the exact same feature engineering
    functions as training, so the output is guaranteed to match what the
    models expect -- no manual feature list duplication, no mismatch risk.

    historical_range: optional (start_date, end_date) strings 'YYYY-MM-DD'.
    If given, pulls real historical weather for that range instead of a
    forward-looking forecast. Use this for any past date.
    """
    if historical_range:
        start_date, end_date = historical_range
        weather = fetch_historical_weather(latitude, longitude, start_date, end_date)
        cutoff = None
    else:
        weather = fetch_weather_forecast(latitude, longitude, days=days, past_days=2)
        cutoff = pd.Timestamp.now(tz=weather["timestamp"].dt.tz).normalize()

    weather = upsample_to_15min(weather)
    weather = add_calendar_columns(weather)
    weather["pv_capacity_kW"] = pv_capacity_kW
    weather["house_id"] = 0

    weather = add_cyclical_time_features(weather)
    weather = add_pv_physics_features(weather)

    if cutoff is not None:
        weather = weather[weather["timestamp"] >= cutoff].reset_index(drop=True)

    return weather