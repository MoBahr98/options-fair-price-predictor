import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_dir):
    all_records = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(data_dir, file_name)) as f:
                data = json.load(f)

                # Unwrap nested list if needed
                if isinstance(data, list) and isinstance(data[0], list):
                    data = data[0]

                all_records.extend(data)

    df = pd.DataFrame(all_records)

    # Convert numeric columns from string to float
    numeric_cols = [
        'last', 'mark', 'bid', 'ask', 'strike', 'bid_size',
        'ask_size', 'volume', 'open_interest', 'implied_volatility',
        'delta', 'gamma', 'theta', 'vega', 'rho'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    # Choose some features and label (e.g. predict whether 'mark' increases)
    features = ['strike', 'bid', 'ask', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']
    label = 'mark'

    df = df[df[label] > 0.01]  # Filter out rows where label is too small
    df = df[df[label] < 1000]  # Filter out rows where label is too large

    X = df[features]
    y = df[label]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler
