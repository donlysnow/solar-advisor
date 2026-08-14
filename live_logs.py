import json
import os
import config


def _load_all_logs():
    if not os.path.exists(config.LOG_PATH):
        return {}
    with open(config.LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_all_logs(logs):
    with open(config.LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


def log_day(date_str, latitude, longitude, pv_capacity_kW, summary, recommendations, day_data):
    """
    Saves (or overwrites) one day's live forecast result, keyed by date.
    Re-running Live Mode on the same day replaces that day's entry with
    the latest result rather than duplicating it.
    """
    logs = _load_all_logs()
    logs[date_str] = {
        "date": date_str,
        "latitude": latitude,
        "longitude": longitude,
        "pv_capacity_kW": pv_capacity_kW,
        "summary": summary,
        "recommendations": recommendations,
        "day_data": day_data,
    }
    _save_all_logs(logs)


def get_logged_dates():
    logs = _load_all_logs()
    return sorted(logs.keys())


def get_logged_day(date_str):
    logs = _load_all_logs()
    return logs.get(date_str)