# Smart Solar Energy Advisor

Predicts a household's non-flexible load (base load + HVAC) and solar
generation, then recommends the best time window to run flexible appliances
(washing machine, dishwasher, EV charging) based on forecasted solar surplus.

## Two modes, one model

- **Historical mode**: pick a house/date from the bundled Kaggle dataset.
- **Live mode**: enter your location and panel capacity, get a
  recommendation for right now using real weather from Open-Meteo.

Both modes use the exact same trained models and the exact same feature
engineering code (`data_pipeline.py`), because every feature is
forecast-safe: weather, calendar, and panel capacity only. No live sensor
reading (battery state, current appliance status, grid import/export) is
ever required as an input. This is what makes live mode possible without a
smart meter -- a new user with zero history can get a recommendation
immediately.

## Project structure

```
solar_app/
  config.py           Data path, model paths, appliance settings
  data_pipeline.py     Forecast-safe feature engineering (calendar +
                        weather + PV physics). Explicit allow-list of
                        features, shared by training and live inference.
  train_models.py      Trains XGBoost non-flexible-load and PV models
  recommender.py        Prediction + appliance scheduling logic
  live_forecast.py      Fetches weather (forecast or historical archive)
                        and builds a forecast-safe feature row
  app.py                 Flask backend and API routes
  templates/
    dashboard.html       Frontend dashboard
  models/                 Saved trained models (created after training)
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your data path in `config.py` (only needed for historical mode):
   ```python
   DATA_PATH = "/path/to/your/home_energy_management.csv"
   ```

3. Train the models:
   ```
   python train_models.py
   ```
   This trains four models (non-flexible load 15min/1day, PV 15min/1day)
   using only forecast-safe features, and saves them to `models/`.

4. Run the app:
   ```
   python app.py
   ```
   Open `http://localhost:5000`.

## How the recommendation logic works

1. The non-flexible-load model predicts `base_load_kW + hvac_power_kW`
   (household demand minus the appliances we're scheduling).
2. The PV model predicts solar generation from weather + panel capacity.
3. Surplus = predicted PV minus predicted non-flexible load.
4. For each flexible appliance, a sliding window search finds the
   contiguous block with the highest average surplus, sized to that
   appliance's typical run duration.
5. If the average surplus still doesn't cover the appliance's typical
   power draw, the app flags "grid assistance likely needed" instead of
   claiming full solar coverage.

## Why the load model accuracy is now lower than the first version

Earlier iterations of this project let the load model see `hvac_power_kW`
and other live sensor readings as direct inputs, which pushed R2 to
0.999 -- but that number was hollow, since a live app has no way to know
"is HVAC running right now" for a brand new date. Restricting the model to
weather + calendar features that are always knowable in advance drops
accuracy, but makes the number honest and the app actually deployable for
a date that isn't already sitting in a CSV.

## Deployment

The app runs on Flask's dev server locally, but needs a production WSGI
server (`gunicorn`) for real traffic before deploying to a host like
Render. Model files and (optionally) the CSV ship with the repo since
they're small; live mode doesn't need the CSV at all.

## Extending this project

- Add a cost-optimization layer using `tou_price_per_kWh` if your tariff
  schedule is fixed and known in advance (it can be added back as a
  forecast-safe feature if it's deterministic by hour/day).
- Add `fetch_historical_weather()` (already in `live_forecast.py`) to
  backtest the app against any real past date anywhere in the world,
  without needing the original Kaggle CSV.
- Swap the greedy sliding-window scheduler for a linear program (PuLP) if
  you need to schedule multiple appliances without them competing for the
  same surplus window.
