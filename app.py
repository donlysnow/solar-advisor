from flask import Flask, jsonify, request, render_template, session, send_from_directory
import pandas as pd
from datetime import datetime

import config
from recommender import (
    get_recommendations_for_house_day, load_model, compute_surplus,
    recommend_schedule_multiday, summarize_day,
)
from data_pipeline import load_raw_data
from live_forecast import (
    build_forecast_row_set, fetch_weather_display, summarize_weather_by_day,
)
import live_logs
import advisor
from battery import simulate_battery

app = Flask(__name__)
app.secret_key = "solar_secret_key"
DATA_DIR = "data"

@app.route('/sw.js')
def serve_sw():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')

DEFAULT_APPLIANCES = [
    {"name": "washing_machine", "duration_minutes": 60, "typical_power_kW": 0.8},
    {"name": "dishwasher", "duration_minutes": 90, "typical_power_kW": 1.2},
    {"name": "ev_charging", "duration_minutes": 180, "typical_power_kW": 3.5},
]


def get_session_appliances():
    appliances = session.get("appliances", DEFAULT_APPLIANCES)
    return [a for a in appliances if a.get("active", True)]


def compute_multiday(latitude, longitude, pv_capacity_kW, background_load_kW,
                      appliances, forecast_days=3, system_loss_pct=0, battery=None):
    forecast_df = build_forecast_row_set(latitude, longitude, pv_capacity_kW, days=forecast_days)
    forecast_df = forecast_df.dropna(subset=["solar_irradiance_lag_96"])
    if forecast_df.empty:
        raise ValueError("Not enough forecast data returned to build a full forecast")

    pv_model_obj, pv_feats = load_model("pv_model_15min")
    missing_pv = [c for c in pv_feats if c not in forecast_df.columns]
    if missing_pv:
        raise RuntimeError(f"Feature mismatch: missing_pv={missing_pv}")

    forecast_df["predicted_nonflex_kW"] = float(background_load_kW)
    predicted_pv = pv_model_obj.predict(forecast_df[pv_feats])
    if system_loss_pct:
        predicted_pv = predicted_pv * (1 - system_loss_pct / 100.0)
    forecast_df["predicted_pv_kW"] = predicted_pv

    battery_enabled = False
    if battery:
        battery_enabled = True

    now = pd.Timestamp.now()
    
    # Run the multiday scheduler which handles battery carry-over and grid optimization
    final_df, all_recs = recommend_schedule_multiday(
        forecast_df, appliances=appliances, battery=battery, current_time=now
    )
    
    final_df["cal_date"] = final_df["timestamp"].dt.date
    days_out = []
    
    for i, (cal_date, df_day) in enumerate(final_df.groupby("cal_date", sort=True)):
        date_str = str(cal_date)
        recs = all_recs.get(date_str, {})
        summary = summarize_day(df_day, recs, appliances)

        serializable_recs = {
            appliance: (
                {
                    "start_time": str(w["start_time"]),
                    "end_time": str(w["end_time"]),
                    "solar_covered": w["solar_covered"],
                    "solar_pct": w.get("solar_pct", 0),
                    "battery_pct": w.get("battery_pct", 0),
                    "grid_pct": w.get("grid_pct", 0),
                    "coverage_pct": w.get("coverage_pct", 0),
                } if w else None
            )
            for appliance, w in recs.items()
        }

        day_data = [
            {
                "timestamp": str(row["timestamp"]),
                "predicted_nonflex_kW": round(float(row["predicted_nonflex_kW"]), 3),
                "predicted_pv_kW": round(float(row["predicted_pv_kW"]), 3),
                "surplus_kW": round(float(row.get("surplus_kW", 0)), 3),
                "battery_soc_kWh": round(float(row.get("battery_soc_kWh", 0)), 2) if battery_enabled else None,
                "grid_import_kW": round(float(row.get("grid_import_kW", 0)), 3),
                "battery_discharge_kW": round(float(row.get("battery_discharge_kW", 0)), 3),
                "solar_used_kW": round(float(row.get("solar_used_kW", 0)), 3),
            }
            for _, row in df_day.iterrows()
        ]

        days_out.append({
            "date": date_str,
            "label": "Today" if i == 0 else ("Tomorrow" if i == 1 else f"Day {i + 1}"),
            "summary": summary,
            "recommendations": serializable_recs,
            "day_data": day_data,
            "background_load_kW": background_load_kW,
            "battery_enabled": battery_enabled,
        })

    return days_out


# ---------------- Pages ----------------

@app.route("/")
def home():
    return render_template("home.html", active="home")


@app.route("/historical")
def historical_page():
    return render_template("historical.html", active="historical")


@app.route("/live")
def live_page():
    return render_template("live.html", active="live")


@app.route("/weather")
def weather_page():
    return render_template("weather.html", active="weather")


@app.route("/advisor")
def advisor_page():
    return render_template("advisor.html", active="advisor")


@app.route("/appliances")
def appliances_page():
    return render_template("appliances.html", active="appliances")


@app.route("/about")
def about_page():
    return render_template("about.html", active="about")


# ---------------- Appliance storage (session-based, no login) ----------------

@app.route("/api/appliances", methods=["GET"])
def get_appliances():
    return jsonify({"appliances": session.get("appliances", DEFAULT_APPLIANCES)})


@app.route("/api/appliances", methods=["POST"])
def save_appliances():
    body = request.get_json(silent=True) or {}
    appliances = body.get("appliances", [])
    if not appliances:
        appliances = DEFAULT_APPLIANCES
    session["appliances"] = appliances
    return jsonify({"appliances": appliances})


# ---------------- Live Mode settings (persist last-used values) ----------------

@app.route("/api/live-settings", methods=["GET"])
def get_live_settings():
    settings = session.get("live_settings", {
        "lat": config.DEFAULT_LATITUDE,
        "lon": config.DEFAULT_LONGITUDE,
        "pv_capacity": 4.0,
        "background_load_kW": config.DEFAULT_BACKGROUND_LOAD_KW,
        "system_loss_pct": config.DEFAULT_SYSTEM_LOSS_PCT,
        "battery_enabled": False,
        "battery_capacity_kWh": config.DEFAULT_BATTERY_CAPACITY_KWH,
        "battery_charge_rate_kW": config.DEFAULT_BATTERY_CHARGE_RATE_KW,
        "battery_discharge_rate_kW": config.DEFAULT_BATTERY_DISCHARGE_RATE_KW,
        "battery_initial_soc_pct": config.DEFAULT_BATTERY_INITIAL_SOC_PCT,
    })
    return jsonify(settings)


@app.route("/api/live-settings", methods=["POST"])
def save_live_settings():
    body = request.get_json(silent=True) or {}
    session["live_settings"] = body
    return jsonify(body)


# ---------------- App Settings (API Keys, etc) ----------------
# Removed Settings Modal for now


# ---------------- Data lookups (sample dataset) ----------------

@app.route("/api/houses")
def list_houses():
    df = load_raw_data()
    house_ids = sorted(df[config.HOUSE_COL].unique().tolist())
    return jsonify({"house_ids": house_ids})


@app.route("/api/dates")
def list_dates():
    house_id = request.args.get("house_id", type=int, default=1)
    df = load_raw_data()
    df_house = df[df[config.HOUSE_COL] == house_id]
    dates = sorted(pd.to_datetime(df_house[config.TIMESTAMP_COL]).dt.date.astype(str).unique().tolist())
    return jsonify({"dates": dates})


# ---------------- My Live Log (your own logged days) ----------------

@app.route("/api/my-log/dates")
def my_log_dates():
    return jsonify({"dates": live_logs.get_logged_dates()})


@app.route("/api/my-log/day")
def my_log_day():
    date_str = request.args.get("date")
    entry = live_logs.get_logged_day(date_str)
    if not entry:
        return jsonify({"error": f"No logged live forecast for {date_str}"}), 404
    return jsonify(entry)


# ---------------- Recommendations (sample dataset, historical mode -- model-based) ----------------

@app.route("/api/recommendations", methods=["POST"])
def recommendations():
    body = request.get_json(silent=True) or {}
    house_id = body.get("house_id", 1)
    date_str = body.get("date")
    appliances = get_session_appliances()

    if not date_str:
        return jsonify({"error": "date is required, format YYYY-MM-DD"}), 400

    try:
        result = get_recommendations_for_house_day(house_id, date_str, appliances=appliances)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except FileNotFoundError:
        return jsonify({"error": "Model files not found. Run train_models.py first."}), 500

    serializable = {
        "house_id": result["house_id"],
        "date": result["date"],
        "summary": result["summary"],
        "recommendations": {
            appliance: (
                {
                    "start_time": str(w["start_time"]),
                    "end_time": str(w["end_time"]),
                    "avg_surplus_kW": w["avg_surplus_kW"],
                    "solar_covered": w["solar_covered"],
                    "coverage_pct": w.get("coverage_pct", 0),
                }
                if w else None
            )
            for appliance, w in result["recommendations"].items()
        },
        "day_data": [
            {
                "timestamp": str(row[config.TIMESTAMP_COL]),
                "predicted_nonflex_kW": round(row["predicted_nonflex_kW"], 3),
                "predicted_pv_kW": round(row["predicted_pv_kW"], 3),
                "surplus_kW": round(row["surplus_kW"], 3),
            }
            for row in result["day_data"]
        ],
    }
    return jsonify(serializable)


# ---------------- Live recommendations, single day ----------------

@app.route("/api/live-recommendations", methods=["POST"])
def live_recommendations():
    body = request.get_json(silent=True) or {}
    latitude = body.get("lat", config.DEFAULT_LATITUDE)
    longitude = body.get("lon", config.DEFAULT_LONGITUDE)
    pv_capacity_kW = body.get("pv_capacity", 4.0)
    background_load_kW = body.get("background_load_kW", config.DEFAULT_BACKGROUND_LOAD_KW)
    system_loss_pct = body.get("system_loss_pct", 0)
    battery = body.get("battery")
    appliances = get_session_appliances()

    try:
        days_out = compute_multiday(
            latitude, longitude, pv_capacity_kW, background_load_kW, appliances,
            forecast_days=1, system_loss_pct=system_loss_pct, battery=battery,
        )
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Weather fetch failed: {e}"}), 502

    today = days_out[0]
    today_str = datetime.now().strftime("%Y-%m-%d")
    live_logs.log_day(
        date_str=today_str, latitude=latitude, longitude=longitude,
        pv_capacity_kW=pv_capacity_kW, summary=today["summary"],
        recommendations=today["recommendations"], day_data=today["day_data"],
    )

    return jsonify({
        "latitude": latitude, "longitude": longitude, "pv_capacity_kW": pv_capacity_kW,
        "background_load_kW": background_load_kW,
        "summary": today["summary"], "recommendations": today["recommendations"],
        "day_data": today["day_data"], "battery_enabled": today["battery_enabled"],
    })


# ---------------- Live recommendations, multi-day ----------------

@app.route("/api/live-recommendations-multiday", methods=["POST"])
def live_recommendations_multiday():
    body = request.get_json(silent=True) or {}
    latitude = body.get("lat", config.DEFAULT_LATITUDE)
    longitude = body.get("lon", config.DEFAULT_LONGITUDE)
    pv_capacity_kW = body.get("pv_capacity", 4.0)
    background_load_kW = body.get("background_load_kW", config.DEFAULT_BACKGROUND_LOAD_KW)
    system_loss_pct = body.get("system_loss_pct", 0)
    battery = body.get("battery")
    forecast_days = body.get("days", 3)
    appliances = get_session_appliances()

    try:
        days_out = compute_multiday(
            latitude, longitude, pv_capacity_kW, background_load_kW, appliances,
            forecast_days=forecast_days, system_loss_pct=system_loss_pct, battery=battery,
        )
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Weather fetch failed: {e}"}), 502

    today = days_out[0]
    live_logs.log_day(
        date_str=today["date"], latitude=latitude, longitude=longitude,
        pv_capacity_kW=pv_capacity_kW, summary=today["summary"],
        recommendations=today["recommendations"], day_data=today["day_data"],
    )

    return jsonify({
        "latitude": latitude, "longitude": longitude, "pv_capacity_kW": pv_capacity_kW,
        "background_load_kW": background_load_kW, "days": days_out,
    })


# ---------------- Weather Forecast page ----------------

@app.route("/api/weather-forecast", methods=["POST"])
def weather_forecast():
    body = request.get_json(silent=True) or {}
    latitude = body.get("lat", config.DEFAULT_LATITUDE)
    longitude = body.get("lon", config.DEFAULT_LONGITUDE)
    days = body.get("days", 3)

    try:
        df = fetch_weather_display(latitude, longitude, days=days)
    except Exception as e:
        return jsonify({"error": f"Weather fetch failed: {e}"}), 502

    daily_summary = summarize_weather_by_day(df)
    hourly = [
        {
            "timestamp": str(row["timestamp"]),
            "outdoor_temp_C": round(float(row["outdoor_temp_C"]), 1),
            "humidity_pct": round(float(row["humidity_pct"]), 0),
            "solar_irradiance_Wm2": round(float(row["solar_irradiance_Wm2"]), 0),
            "cloudcover_pct": round(float(row["cloudcover_pct"]), 0),
            "precipitation_mm": round(float(row["precipitation_mm"]), 2),
        }
        for _, row in df.iterrows()
    ]

    return jsonify({"latitude": latitude, "longitude": longitude, "daily_summary": daily_summary, "hourly": hourly})


@app.route("/api/advisor-tips", methods=["POST"])
def advisor_tips():
    body = request.get_json(silent=True) or {}
    latitude = body.get("lat", config.DEFAULT_LATITUDE)
    longitude = body.get("lon", config.DEFAULT_LONGITUDE)
    pv_capacity_kW = body.get("pv_capacity", 4.0)
    background_load_kW = body.get("background_load_kW", config.DEFAULT_BACKGROUND_LOAD_KW)
    user_question = body.get("question")
    appliances = get_session_appliances()

    try:
        days_out = compute_multiday(latitude, longitude, pv_capacity_kW, background_load_kW, appliances, forecast_days=3)
        weather_df = fetch_weather_display(latitude, longitude, days=3)
    except Exception as e:
        return jsonify({"error": f"Could not build advisor context: {e}"}), 502

    weather_daily = summarize_weather_by_day(weather_df)
    result, source = advisor.get_ai_advice(days_out, appliances, weather_daily, background_load_kW, user_question)

    return jsonify({"result": result, "source": source, "days": days_out, "weather_daily": weather_daily})


@app.route("/api/voice-command", methods=["POST"])
def voice_command():
    body = request.get_json(silent=True) or {}
    transcription = body.get("transcription")
    if not transcription:
        return jsonify({"error": "No transcription provided."}), 400
    
    result = advisor.process_voice_command(transcription)
    
    if result.get("action") == "ADD_APPLIANCE" and "appliance" in result:
        appliances = get_session_appliances()
        appliances.append(result["appliance"])
        session["appliances"] = appliances
        
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)