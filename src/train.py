import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocess import load_and_preprocess_data

def train_model(data_dir, model_path, scaler_path):
    X, y, scaler = load_and_preprocess_data(data_dir)

    # Sanity check 1: is 'mark' close to (bid + ask) / 2?
    import pandas as pd
    import os, json
    all_records = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(data_dir, file_name)) as f:
                data = json.load(f)
                if isinstance(data, list) and isinstance(data[0], list):
                    data = data[0]
                all_records.extend(data)
    df_raw = pd.DataFrame(all_records)
    df_raw['mark'] = pd.to_numeric(df_raw['mark'], errors='coerce')
    df_raw['bid'] = pd.to_numeric(df_raw['bid'], errors='coerce')
    df_raw['ask'] = pd.to_numeric(df_raw['ask'], errors='coerce')
    df_raw = df_raw.dropna(subset=['mark', 'bid', 'ask'])

    df_raw['midpoint'] = (df_raw['bid'] + df_raw['ask']) / 2
    correlation = df_raw['mark'].corr(df_raw['midpoint'])

    print(f"Correlation between mark and (bid + ask)/2: {correlation:.4f}")
    if correlation > 0.95:
        print("High correlation — model may just be learning bid/ask average!")

    # Sanity check 2: Random label shuffle
    y_shuffled = np.random.permutation(y)
    dummy_model = RandomForestRegressor(n_estimators=100, random_state=42)
    dummy_model.fit(X, y_shuffled)
    y_dummy_pred = dummy_model.predict(X)
    r2_dummy = r2_score(y_shuffled, y_dummy_pred)
    print(f"🎲 R² with shuffled labels (should be near 0): {r2_dummy:.4f}")
    if r2_dummy > 0.2:
        print("Model may be overfitting or data leakage is occurring!")

    # Train real model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("Evaluation Metrics:")
    print(f"   - Mean Squared Error (MSE): {mse:.4f}")
    print(f"   - R² Score: {r2:.4f}")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and scaler saved.")
