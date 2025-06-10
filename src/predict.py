import json
import joblib
import pandas as pd

def predict_from_json(json_path, model_path, scaler_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # If data is a nested list, unwrap it
    if isinstance(data, list) and isinstance(data[0], list):
        data = data[0]

    df = pd.DataFrame(data)

    numeric_cols = [
        'strike', 'bid', 'ask', 'implied_volatility',
        'delta', 'gamma', 'theta', 'vega', 'rho'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    # Filter out unrealistic or zero values in key features
    df = df[
        (df['bid'] > 0.01) &
        (df['ask'] > 0.01) &
        (df['implied_volatility'] > 0.01)
]


    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    X = scaler.transform(df[numeric_cols])
    predictions = model.predict(X)

    return predictions.tolist()
