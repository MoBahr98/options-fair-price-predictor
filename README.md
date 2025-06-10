# S&P500 Options (SPY) Market Price Predictor

## Developer

- **Name**: Mohamed Bahr  
- **GitHub Repository**: [Repository](https://github.com/MoBahr98/options-fair-price-predictor.git)  

---

This project is a machine learning model designed to predict the fair market price ("mark") of S&P 500 (SPY) options contracts. It is built using Python, scikit-learn, Pandas, and NumPy, with a CLI interface for usability.

---

## Use Case

Financial analysts, institutions, or retail investors can use this tool to:

- Estimate a fair price for options contracts based on market conditions.

- Compare predictions to actual bid/ask values to identify mispriced opportunities.

## Installation

1. **Clone the repository**:
   ```bash
    git clone https://github.com/your-username/stock-trend-predictor.git
    cd stock-trend-predictor
    pip install -r requirements.txt
   ```

2. **Train and Predict**:
   ```bash
   python3 main.py
   ```

3. **CLI**:
   Run the command line interface:
   ```bash
   python3 ui/cli.py
   ```

---

## Dataset

This project uses the Kaggle dataset: [S&P 500 Options (SPY) Implied Volatility 2019–2024](https://www.kaggle.com/datasets/shankerabhigyan/s-and-p500-options-spy-implied-volatility-2019-24)

- Download the files and create data directory to place them in.

---

## Limitations

- Model has very high correlation with bid/ask average, which may reduce predictive novelty.

---

## Future plans

- Flask API for serving the model via REST
- React-based Web UI 

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it as permitted under this license.
