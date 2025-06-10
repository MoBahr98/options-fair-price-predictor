# command line interface for predicting option prices using a pre-trained model

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/stock_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Define the features your model expects
FEATURES = [
    'strike', 'bid', 'ask', 'implied_volatility',
    'delta', 'gamma', 'theta', 'vega', 'rho'
]

def get_user_input():
    print("\nEnter option contract details:")
    data = {}
    for feature in FEATURES:
        while True:
            try:
                val = float(input(f"{feature}: "))
                data[feature] = val
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    return data

def predict_option_price(data):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.DataFrame([data])  # Convert input dict to DataFrame
    X_scaled = scaler.transform(df[FEATURES])
    prediction = model.predict(X_scaled)[0]

    return prediction

if __name__ == "__main__":
    user_data = get_user_input()
    price = predict_option_price(user_data)
    print(f"\nPredicted fair market price (mark): ${price:.2f}")
