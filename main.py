from src.train import train_model
from src.predict import predict_from_json

# Paths
DATA_DIR = "data"
MODEL_PATH = "models/stock_model.pkl"
SCALER_PATH = "models/scaler.pkl"
INPUT_PATH = "data/spy_options_data_24.json"

if __name__ == "__main__":
    print("Training model...")
    train_model(DATA_DIR, MODEL_PATH, SCALER_PATH)

    print("Predicting new data...")
    predictions = predict_from_json(INPUT_PATH, MODEL_PATH, SCALER_PATH)
    # Formatted summary
    print("\nSample predictions:")
    for i, pred in enumerate(predictions[:10]):  # Show first 10 predictions only
        print(f"{i+1}: {pred:.2f}")

