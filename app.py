from flask import Flask, jsonify, request, render_template, session, send_from_directory, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import pandas as pd
from datetime import datetime
import json
import os
from werkzeug.middleware.proxy_fix import ProxyFix

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
from models import db, User, UserBadge, DailyLog

app = Flask(__name__)
# Tell Flask it is behind a proxy (like Render) so it gets the real IP and doesn't drop sessions
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=0)
app.secret_key = "solar_secret_key"
DATA_DIR = "data"
# Render uses ephemeral disks on the free tier, which means SQLite databases get wiped when the server restarts.
# By using DATABASE_URL, we can connect a free PostgreSQL database (like Supabase or Render Postgres) later!
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///solar_advisor.db")
# Fix for some postgres URIs that start with postgres:// instead of postgresql://
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Secure cookies for Render HTTPS
if os.environ.get("RENDER"):
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

@app.before_request
def require_login():
    allowed_routes = ['login', 'register', 'serve_sw', 'static']
    if request.endpoint not in allowed_routes and not current_user.is_authenticated:
        return redirect(url_for('login'))

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
                      appliances, forecast_days=3, system_loss_pct=0, battery=None, electricity_rate=225.0):
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
        summary = summarize_day(df_day, recs, appliances, electricity_rate=electricity_rate)

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


# ---------------- Auth Pages ----------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return redirect(url_for('register'))
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(name=name, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('home'))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash("Login Unsuccessful. Please check email and password", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", active="profile")

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
    default_settings = {
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
        "electricity_rate": 225.0
    }
    
    if current_user.is_authenticated:
        try:
            user_settings = json.loads(current_user.live_settings)
            user_settings["electricity_rate"] = current_user.electricity_rate
            return jsonify({**default_settings, **user_settings})
        except:
            return jsonify(default_settings)
            
    settings = session.get("live_settings", default_settings)
    return jsonify(settings)


@app.route("/api/live-settings", methods=["POST"])
def save_live_settings():
    body = request.get_json(silent=True) or {}
    
    if current_user.is_authenticated:
        if "electricity_rate" in body:
            current_user.electricity_rate = float(body.pop("electricity_rate"))
        current_user.live_settings = json.dumps(body)
        db.session.commit()
    else:
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
    electricity_rate = body.get("electricity_rate", 225.0)
    battery = body.get("battery")
    appliances = get_session_appliances()

    try:
        days_out = compute_multiday(
            latitude, longitude, pv_capacity_kW, background_load_kW, appliances,
            forecast_days=1, system_loss_pct=system_loss_pct, battery=battery, electricity_rate=electricity_rate,
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

    if current_user.is_authenticated:
        # Gamification: Grant badges
        existing_badges = {b.badge_name for b in current_user.badges}
        
        def grant_badge(name):
            if name not in existing_badges:
                new_badge = UserBadge(user_id=current_user.id, badge_name=name)
                db.session.add(new_badge)
                existing_badges.add(name)

        grant_badge("First Plan")
        
        if today["summary"].get("solar_quality") == "good":
            grant_badge("Perfect Solar Day")
            
        if today["battery_enabled"]:
            grant_badge("Off-Grid Master")
            
        db.session.commit()

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
    electricity_rate = body.get("electricity_rate", 225.0)
    battery = body.get("battery")
    forecast_days = body.get("days", 3)
    appliances = get_session_appliances()

    try:
        days_out = compute_multiday(
            latitude, longitude, pv_capacity_kW, background_load_kW, appliances,
            forecast_days=forecast_days, system_loss_pct=system_loss_pct, battery=battery, electricity_rate=electricity_rate,
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

    if current_user.is_authenticated:
        # Gamification: Grant badges
        existing_badges = {b.badge_name for b in current_user.badges}
        
        def grant_badge(name):
            if name not in existing_badges:
                new_badge = UserBadge(user_id=current_user.id, badge_name=name)
                db.session.add(new_badge)
                existing_badges.add(name)

        grant_badge("First Plan")
        
        if today["summary"].get("solar_quality") == "good":
            grant_badge("Perfect Solar Day")
            
        if today["battery_enabled"]:
            grant_badge("Off-Grid Master")
            
        db.session.commit()

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


@app.route("/api/copilot", methods=["POST"])
def copilot():
    body = request.get_json(silent=True) or {}
    message = body.get("message")
    context_str = body.get("context", "")
    
    if not message:
        return jsonify({"error": "No message provided."}), 400
        
    chat_history = session.get("copilot_history", [])
    chat_history.append({"role": "user", "content": message})
    
    result = advisor.process_copilot_chat(chat_history, context_str)
    
    if "message" in result:
        chat_history.append({"role": "assistant", "content": result["message"]})
        # Keep last 20 messages in session
        session["copilot_history"] = chat_history[-20:]
    
    if result.get("action") == "ADD_APPLIANCE" and "appliance" in result:
        appliances = get_session_appliances()
        appliances.append(result["appliance"])
        session["appliances"] = appliances
        
    return jsonify(result)

# ---------------- Push Notifications ----------------

@app.route("/api/vapid-public-key", methods=["GET"])
def vapid_public_key():
    return jsonify({"publicKey": config.VAPID_PUBLIC_KEY})


@app.route("/api/subscribe", methods=["POST"])
@login_required
def subscribe():
    subscription = request.get_json()
    if not subscription:
        return jsonify({"error": "No subscription data"}), 400
    
    current_user.push_subscription = json.dumps(subscription)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/test-push", methods=["POST"])
@login_required
def test_push():
    if not current_user.push_subscription:
        return jsonify({"error": "User not subscribed"}), 400
        
    try:
        from pywebpush import webpush, WebPushException
        subscription = json.loads(current_user.push_subscription)
        webpush(
            subscription_info=subscription,
            data=json.dumps({"title": "Solar Advisor", "body": "This is a test notification!"}),
            vapid_private_key=config.VAPID_PRIVATE_KEY,
            vapid_claims=config.VAPID_CLAIMS
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)