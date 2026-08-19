import os
import os as _os 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "groq/compound"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Fill in the path to your downloaded Kaggle CSV
DATA_PATH = "C:/Users/sambo/Downloads/HEMS_dataset.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "data", "live_logs.json")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Columns
TIMESTAMP_COL = "timestamp"
HOUSE_COL = "house_id"

LOAD_TARGET_15 = "target_load_kW_15"
LOAD_TARGET_1D = "target_load_kW_1D"
PV_TARGET_15 = "target_pv_kW_15"
PV_TARGET_1D = "target_pv_kW_1D"
NONFLEX_TARGET_15 = "target_nonflex_kW_15"
NONFLEX_TARGET_1D = "target_nonflex_kW_1D"

FLEXIBLE_APPLIANCES = {
    "washing_machine": {
        "flag_col": "washing_machine_on",
        "duration_slots": 4,
        "typical_power_kW": 0.8,
    },
    "dishwasher": {
        "flag_col": "dishwasher_on",
        "duration_slots": 6,
        "typical_power_kW": 1.2,
    },
    "ev_charging": {
        "flag_col": "ev_charging_on",
        "duration_slots": 12,
        "typical_power_kW": 3.5,
    },
}

RANDOM_STATE = 42
TEST_FRACTION = 0.2

# Default location for live weather forecasting. Update per deployment.
DEFAULT_LATITUDE = 6.5244
DEFAULT_LONGITUDE = 3.3792

# Used only for the estimated-savings display. Rough assumption, not a
# real tariff lookup -- update to match a real local rate if known.
DEFAULT_GRID_PRICE_PER_KWH = 225  # NGN/kWh, adjust as needed

# Default assumed background load for Live Mode/Advisor -- a fixed user
# assumption, not a model prediction. Roughly "fridge + standby electronics."
DEFAULT_BACKGROUND_LOAD_KW = 0.3

# System losses: inverter heat, wiring resistance, panel derating under
# real conditions all reduce usable output below the nameplate rating.
# 0 means "don't apply any extra reduction" (the trained model's PV output
# already reflects real-world generation ratios from its training data).
DEFAULT_SYSTEM_LOSS_PCT = 0

# Battery defaults, used to pre-fill the UI. Battery is OFF by default --
# these only apply if the user explicitly enables it.
DEFAULT_BATTERY_CAPACITY_KWH = 5.0
DEFAULT_BATTERY_CHARGE_RATE_KW = 2.0
DEFAULT_BATTERY_DISCHARGE_RATE_KW = 2.0
DEFAULT_BATTERY_INITIAL_SOC_PCT = 50

# ---------------------------------------------------------
# Push Notification VAPID Keys
# ---------------------------------------------------------
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "BDgQSK93TW6AVS1z1JWfgD2KcacWfx1HRkaHnzs3TaEFrcCTMpGz0HXfVAcOYOgrVQFTZua0WGXG8ySxI3QRpmc")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "s7Ji9etMhlDrQscKKVQIqSPbvCaZSQTvDdQSIzl-15U")
VAPID_CLAIMS = {
    "sub": "mailto:admin@solar-advisor.com"
}