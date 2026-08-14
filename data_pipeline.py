import pandas as pd
import numpy as np
import config


def load_raw_data(path=None):
    path = path or config.DATA_PATH
    if not path:
        raise ValueError(
            "No data path set. Fill in DATA_PATH in config.py or pass a path to load_raw_data()."
        )
    df = pd.read_csv(path)
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL], format="%d-%m-%Y %H:%M")
    df = df.sort_values([config.HOUSE_COL, config.TIMESTAMP_COL]).reset_index(drop=True)
    return df


def add_cyclical_time_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute_of_hour"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_of_hour"] / 60)
    return df


def add_pv_physics_features(df):
    """
    Physics-informed PV features, built only from weather + panel spec.
    Fully computable for any date past or future given a weather source.
    """
    df = df.copy()
    df["clear_sky_pv_kW"] = df["pv_capacity_kW"] * (df["solar_irradiance_Wm2"] / 1000.0)

    for lag in [1, 4, 96]:
        df[f"solar_irradiance_lag_{lag}"] = df.groupby(config.HOUSE_COL)["solar_irradiance_Wm2"].shift(lag)

    df["irradiance_roll_mean_4"] = df.groupby(config.HOUSE_COL)["solar_irradiance_Wm2"].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )

    df["temp_irradiance_interaction"] = df["outdoor_temp_C"] * df["solar_irradiance_Wm2"]
    df["humidity_irradiance_interaction"] = df["humidity_pct"] * df["solar_irradiance_Wm2"]

    return df


def add_nonflexible_load_targets(df):
    """
    non_flexible_load_kW = base_load_kW + hvac_power_kW, i.e. household demand
    MINUS the appliances we're trying to schedule (washing machine, dishwasher,
    EV charging). This is what the surplus calculation actually needs to know
    ahead of time. We predict it directly instead of assuming we can read it
    live, since a real deployment has no way to know "current base load" for
    a house that hasn't reported anything yet.
    """
    df = df.copy()
    df["non_flexible_load_kW"] = df["base_load_kW"] + df["hvac_power_kW"]
    df["target_nonflex_kW_15"] = df.groupby(config.HOUSE_COL)["non_flexible_load_kW"].shift(-1)
    df["target_nonflex_kW_1D"] = df.groupby(config.HOUSE_COL)["non_flexible_load_kW"].shift(-96)
    return df

def augment_pv_capacity(df, scale_factors=(0.15, 0.3, 0.5, 0.75, 1.5, 2.0, 2.5, 3.0, 3.5)):
    """
    Creates synthetic copies of the dataset with both smaller and larger
    panel capacities, scaling pv_capacity_kW and pv_generation_kW (and its
    forecast targets) proportionally -- PV output scales roughly linearly
    with capacity at the same irradiance.

    Original dataset range: ~2.1kW to ~7.8kW.
    scale_factors below 1.0 (0.15, 0.3, 0.5, 0.75) extend the covered range
    down to roughly 0.3kW - 5.9kW, covering small setups like a 1-3 panel
    system (e.g. 3 x 400W = 1.2kW).
    scale_factors above 1.0 (1.5 - 3.5) extend the range up to ~27kW,
    covering larger residential and small-commercial systems.

    Each synthetic copy gets a new house_id offset so lag/rolling features
    computed per-house don't mix real and synthetic rows together.
    """
    max_house_id = df[config.HOUSE_COL].max()
    augmented_frames = [df]

    for i, factor in enumerate(scale_factors, start=1):
        scaled = df.copy()
        scaled[config.HOUSE_COL] = scaled[config.HOUSE_COL] + i * (max_house_id + 1)
        scaled["pv_capacity_kW"] = scaled["pv_capacity_kW"] * factor
        scaled["pv_generation_kW"] = scaled["pv_generation_kW"] * factor
        scaled[config.PV_TARGET_15] = scaled[config.PV_TARGET_15] * factor
        scaled[config.PV_TARGET_1D] = scaled[config.PV_TARGET_1D] * factor
        augmented_frames.append(scaled)

    return pd.concat(augmented_frames, ignore_index=True)


def engineer_features(df):
    """
    Forecast-safe feature engineering: every feature this produces is
    something you can know ahead of time from weather + calendar + panel
    spec alone. No live sensor readings (appliance flags, battery state,
    grid import/export, current demand) are used as inputs, so the trained
    model works identically on historical rows or on a freshly fetched
    weather forecast for any date.
    """
    df = add_cyclical_time_features(df)
    df = add_pv_physics_features(df)
    df = add_nonflexible_load_targets(df)

    # Only the irradiance-based lag/rolling features need history to exist,
    # and that's a maximum of 1 day back -- much less restrictive than the
    # old 1-week self-referential lags.
    physics_feature_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c or c == "clear_sky_pv_kW"]
    df = df.dropna(subset=physics_feature_cols).reset_index(drop=True)

    return df


def time_based_split(df, test_fraction=config.TEST_FRACTION, group_col=config.HOUSE_COL):
    """
    Splits each house's time series into train/test by time,
    so the model is always tested on the most recent period per house.
    """
    train_parts, test_parts = [], []
    for house_id, group in df.groupby(group_col):
        group = group.sort_values(config.TIMESTAMP_COL)
        split_idx = int(len(group) * (1 - test_fraction))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


# Explicit allow-list: every feature here must be computable from weather +
# calendar + panel spec alone, for ANY date, with no live meter required.
# This is deliberately an inclusion list rather than "everything except
# targets" -- that exclusion approach is what caused the live forecast to
# silently depend on sensor columns like battery_soc_kWh or washing_machine_on.
FORECAST_SAFE_FEATURES = [
    "hour_of_day", "minute_of_hour", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "minute_sin", "minute_cos",
    "outdoor_temp_C", "solar_irradiance_Wm2", "humidity_pct",
    "pv_capacity_kW", "clear_sky_pv_kW",
    "solar_irradiance_lag_1", "solar_irradiance_lag_4", "solar_irradiance_lag_96",
    "irradiance_roll_mean_4",
    "temp_irradiance_interaction", "humidity_irradiance_interaction",
]


def get_feature_columns(df):
    return [c for c in FORECAST_SAFE_FEATURES if c in df.columns]


if __name__ == "__main__":
    df = load_raw_data()
    df = engineer_features(df)
    print(f"Shape after feature engineering: {df.shape}")
    train_df, test_df = time_based_split(df)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    feats = get_feature_columns(df)
    print(f"Feature count: {len(feats)}")
    print(feats)
