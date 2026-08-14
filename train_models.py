import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from data_pipeline import (
    load_raw_data, engineer_features, time_based_split, get_feature_columns,
    augment_pv_capacity,
)


def train_single_model(train_df, test_df, feature_cols, target_col, model_name):
    train_rows = train_df.dropna(subset=[target_col])
    test_rows = test_df.dropna(subset=[target_col])

    X_train, y_train = train_rows[feature_cols], train_rows[target_col]
    X_test, y_test = test_rows[feature_cols], test_rows[target_col]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mae",
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{model_name}")
    print(f"  MAE:  {mae:.4f} kW")
    print(f"  RMSE: {rmse:.4f} kW")
    print(f"  R2:   {r2:.4f}")

    model_path = os.path.join(config.MODEL_DIR, f"{model_name}.joblib")
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    print(f"  Saved to {model_path}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


def train_all_models(data_path=None):
    print("Loading data...")
    df = load_raw_data(data_path)

    print("Augmenting with smaller (~0.3kW) and larger (~27kW) PV panel sizes...")
    df = augment_pv_capacity(df)
    print(f"Shape after augmentation: {df.shape}")

    print("Engineering forecast-safe features...")
    df = engineer_features(df)

    print("Splitting train/test by house (time based)...")
    train_df, test_df = time_based_split(df)

    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} forecast-safe features: {feature_cols}")

    targets = {
        "nonflex_model_15min": config.NONFLEX_TARGET_15,
        "nonflex_model_1day": config.NONFLEX_TARGET_1D,
        "pv_model_15min": config.PV_TARGET_15,
        "pv_model_1day": config.PV_TARGET_1D,
    }

    results = {}
    for model_name, target_col in targets.items():
        _, metrics = train_single_model(train_df, test_df, feature_cols, target_col, model_name)
        results[model_name] = metrics

    print("\nAll models trained.")
    print("\nNote: PV models were trained on data augmented with larger panel")
    print("capacities (up to ~27kW) so predictions stay reliable for real")
    print("residential/small-commercial systems, not just the original")
    print("dataset's ~7.8kW ceiling.")

    return results


if __name__ == "__main__":
    train_all_models()