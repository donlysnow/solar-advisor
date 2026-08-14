import joblib
import numpy as np
import pandas as pd

import config
from data_pipeline import load_raw_data, engineer_features
from battery import simulate_battery

def load_model(model_name):
    path = f"{config.MODEL_DIR}/{model_name}.joblib"
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_day(df_day):
    nonflex_model_obj, nonflex_feats = load_model("nonflex_model_15min")
    pv_model_obj, pv_feats = load_model("pv_model_15min")

    df_day = df_day.copy()
    df_day["predicted_nonflex_kW"] = nonflex_model_obj.predict(df_day[nonflex_feats])
    df_day["predicted_pv_kW"] = pv_model_obj.predict(df_day[pv_feats])
    return df_day


def compute_surplus(df_day):
    # Backward compatibility for any old calls
    df_day = df_day.copy()
    df_day["surplus_kW"] = df_day["predicted_pv_kW"] - df_day["predicted_nonflex_kW"]
    return df_day


def best_window_for_appliance(df_day, duration_slots, typical_power_kW, battery=None):
    """
    Finds the best window by actually simulating the battery (if present) for each possible window
    and finding the window that results in the minimum total grid import.
    """
    df = df_day.copy()
    n = len(df)
    if n < duration_slots:
        return None

    best_start = -1
    best_grid_import = float('inf')
    best_sim_df = None

    for start_idx in range(n - duration_slots + 1):
        # Create a test load profile
        test_load = df["total_load_kW"].values.copy()
        test_load[start_idx : start_idx + duration_slots] += typical_power_kW
        
        test_df = df.copy()
        test_df["total_load_kW"] = test_load
        
        # Simulate battery or grid
        if battery:
            sim_df = simulate_battery(
                test_df,
                capacity_kWh=battery["capacity_kWh"],
                charge_rate_kW=battery["charge_rate_kW"],
                discharge_rate_kW=battery["discharge_rate_kW"],
                initial_soc_pct=battery.get("initial_soc_pct", config.DEFAULT_BATTERY_INITIAL_SOC_PCT)
            )
        else:
            # Without a battery, grid import is just max(0, load - pv)
            sim_df = test_df.copy()
            sim_df["grid_import_kW"] = np.maximum(0, sim_df["total_load_kW"] - sim_df["predicted_pv_kW"])
            sim_df["solar_used_kW"] = np.minimum(sim_df["predicted_pv_kW"], sim_df["total_load_kW"])
            sim_df["battery_discharge_kW"] = 0.0

        total_grid = sim_df["grid_import_kW"].sum()
        
        if total_grid < best_grid_import:
            best_grid_import = total_grid
            best_start = start_idx
            best_sim_df = sim_df

    if best_start == -1:
        return None

    best_end = best_start + duration_slots - 1
    
    # Calculate coverage percentages for this specific appliance window
    # We look at the delta in energy sources between the chosen sim_df and the base df
    window_sim = best_sim_df.iloc[best_start : best_start + duration_slots]
    window_base = df.iloc[best_start : best_start + duration_slots]
    
    # The appliance load added in this window
    appliance_load = typical_power_kW * duration_slots
    
    # How much extra solar was used?
    if "solar_used_kW" in window_base.columns:
        extra_solar = (window_sim["solar_used_kW"].sum() - window_base["solar_used_kW"].sum())
        extra_battery = (window_sim["battery_discharge_kW"].sum() - window_base["battery_discharge_kW"].sum())
    else:
        # fallback if base didn't have these columns (first appliance scheduled)
        extra_solar = window_sim["solar_used_kW"].sum()
        extra_battery = window_sim["battery_discharge_kW"].sum()

    solar_pct = max(0, min(100, round((extra_solar / appliance_load) * 100)))
    battery_pct = max(0, min(100, round((extra_battery / appliance_load) * 100)))
    grid_pct = max(0, 100 - solar_pct - battery_pct)

    start_time = df.iloc[best_start][config.TIMESTAMP_COL]
    end_time = df.iloc[best_end][config.TIMESTAMP_COL] + pd.Timedelta(minutes=15)

    return {
        "start_idx": best_start,
        "end_idx": best_end,
        "start_time": start_time,
        "end_time": end_time,
        "solar_covered": grid_pct == 0 and battery_pct == 0,
        "solar_pct": solar_pct,
        "battery_pct": battery_pct,
        "grid_pct": grid_pct,
        "coverage_pct": solar_pct, # legacy field for backward compat
        "sim_df_if_chosen": best_sim_df # We return the resulting df to lock it in
    }


def recommend_schedule_multiday(forecast_df, appliances, battery=None, current_time=None):
    """
    Schedules appliances across the entire forecast period (1 to 3 days),
    maintaining continuous battery state. 
    appliances are scheduled once PER DAY.
    """
    if appliances is None:
        appliances = [
            {"name": name, "duration_minutes": cfg["duration_slots"] * 15, "typical_power_kW": cfg["typical_power_kW"]}
            for name, cfg in config.FLEXIBLE_APPLIANCES.items()
        ]

    # Initialize the base load profile
    df = forecast_df.copy()
    df["total_load_kW"] = df["predicted_nonflex_kW"]
    if not battery:
        df["solar_used_kW"] = np.minimum(df["predicted_pv_kW"], df["total_load_kW"])
        df["battery_discharge_kW"] = 0.0
        df["grid_import_kW"] = np.maximum(0, df["total_load_kW"] - df["predicted_pv_kW"])
    else:
        df = simulate_battery(
            df,
            capacity_kWh=battery["capacity_kWh"],
            charge_rate_kW=battery["charge_rate_kW"],
            discharge_rate_kW=battery["discharge_rate_kW"],
            initial_soc_pct=battery.get("initial_soc_pct", config.DEFAULT_BATTERY_INITIAL_SOC_PCT)
        )

    # We must group by day so we schedule the appliance ONCE per day
    df["cal_date"] = df["timestamp"].dt.date
    daily_groups = [d for _, d in df.groupby("cal_date", sort=True)]
    
    all_recommendations = {} # date_str -> recommendations dict

    for df_day in daily_groups:
        date_str = str(df_day["cal_date"].iloc[0])
        daily_recs = {}
        
        # Apply current_time constraint if this day is today
        df_for_scheduling = df_day
        if current_time is not None and df_day["cal_date"].iloc[0] == pd.to_datetime(current_time).date():
            df_for_scheduling = df_day[df_day[config.TIMESTAMP_COL] >= current_time]
            
        for appliance in appliances:
            duration_slots = max(1, round(appliance["duration_minutes"] / 15))
            
            # The search operates on the whole `df` (all 3 days) so battery carryover works perfectly!
            # Wait, no. We only want to search within `df_for_scheduling`.
            # To preserve battery state, we must pass the FULL `df` into `best_window_for_appliance`,
            # but restrict its search space to indices belonging to `df_for_scheduling`.
            # For simplicity, we just slice `df` to `df_for_scheduling`, run the search, and then apply it back to `df`.
            # This means battery carry-over DURING the search is localized, but then we lock it into the global `df`.
            
            window = best_window_for_appliance(
                df_for_scheduling,
                duration_slots=duration_slots,
                typical_power_kW=appliance["typical_power_kW"],
                battery=battery
            )
            
            if window:
                daily_recs[appliance["name"]] = window
                # Lock it into the global df
                global_start_idx = df.index[df[config.TIMESTAMP_COL] == window["start_time"]][0]
                df.loc[global_start_idx : global_start_idx + duration_slots - 1, "total_load_kW"] += appliance["typical_power_kW"]
                
                # Re-simulate the full `df` to update battery/grid for the next appliance
                if battery:
                    df = simulate_battery(
                        df,
                        capacity_kWh=battery["capacity_kWh"],
                        charge_rate_kW=battery["charge_rate_kW"],
                        discharge_rate_kW=battery["discharge_rate_kW"],
                        initial_soc_pct=battery.get("initial_soc_pct", config.DEFAULT_BATTERY_INITIAL_SOC_PCT)
                    )
                else:
                    df["grid_import_kW"] = np.maximum(0, df["total_load_kW"] - df["predicted_pv_kW"])
                    df["solar_used_kW"] = np.minimum(df["predicted_pv_kW"], df["total_load_kW"])
                    df["battery_discharge_kW"] = 0.0
                    
                # Update df_for_scheduling for the next appliance in this day loop
                df_for_scheduling = df.loc[df_for_scheduling.index]
            else:
                daily_recs[appliance["name"]] = None
                
        all_recommendations[date_str] = daily_recs

    return df, all_recommendations


def summarize_day(df_day, recommendations, appliances):
    total_pv_kWh = float(df_day["predicted_pv_kW"].sum()) * 0.25
    peak_pv = float(df_day["predicted_pv_kW"].max())
    peak_time = df_day.loc[df_day["predicted_pv_kW"].idxmax(), config.TIMESTAMP_COL]

    if peak_pv <= 1.0:
        quality = "poor"
        headline = "Low solar output expected today -- most appliances will need battery or grid support."
    elif peak_pv < 3.0:
        quality = "fair"
        headline = f"Modest solar expected today, peaking around {peak_time.strftime('%H:%M')}."
    else:
        quality = "good"
        headline = f"Good solar day -- peak output of {peak_pv:.1f} kW expected around {peak_time.strftime('%H:%M')}."

    fully_covered_count = 0
    estimated_kwh_saved = 0.0
    
    appliance_lookup = {a["name"]: a for a in appliances}

    for appliance_name, window in recommendations.items():
        if not window:
            continue
        if window.get("grid_pct", 100) == 0:
            fully_covered_count += 1
            
        cfg = appliance_lookup.get(appliance_name)
        if cfg:
            run_hours = cfg["duration_minutes"] / 60.0
            full_energy = cfg["typical_power_kW"] * run_hours
            
            # Money saved is from Solar AND Battery usage
            coverage_fraction = (window.get("solar_pct", 0) + window.get("battery_pct", 0)) / 100.0
            estimated_kwh_saved += full_energy * coverage_fraction

    estimated_cost_saved = estimated_kwh_saved * config.DEFAULT_GRID_PRICE_PER_KWH

    return {
        "solar_quality": quality,
        "headline": headline,
        "total_pv_kWh_today": round(total_pv_kWh, 2),
        "appliances_covered": fully_covered_count,
        "appliances_total": len(recommendations),
        "estimated_kwh_saved": round(estimated_kwh_saved, 2),
        "estimated_cost_saved": round(estimated_cost_saved, 0),
    }

def get_recommendations_for_house_day(house_id, date_str, appliances=None):
    df = load_raw_data()
    df = engineer_features(df)

    df_house = df[df[config.HOUSE_COL] == house_id].copy()
    df_house["date"] = df_house[config.TIMESTAMP_COL].dt.date
    target_date = pd.to_datetime(date_str).date()
    df_day = df_house[df_house["date"] == target_date].sort_values(config.TIMESTAMP_COL)

    if df_day.empty:
        raise ValueError(f"No data for house_id={house_id} on {date_str}")

    df_day = predict_day(df_day)
    df_day, all_recs = recommend_schedule_multiday(df_day, appliances=appliances, battery=None)
    recs = all_recs[str(target_date)]
    summary = summarize_day(df_day, recs, appliances)

    return {
        "house_id": house_id,
        "date": date_str,
        "recommendations": recs,
        "summary": summary,
        "day_data": df_day.to_dict(orient="records"),
    }