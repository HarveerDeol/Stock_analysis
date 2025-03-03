# S&P 500 Stock Price Prediction App 📈

A sophisticated web application that analyzes S&P 500 stocks and predicts whether the next day's closing price will move up or down, built with Streamlit and machine learning. this project piggy backs off the works of the following paper:

[https://arxiv.org/abs/1605.00003](url)




## Features

- **S&P 500 Stock Selection**: Choose from all stocks in the S&P 500 index
- **Interactive Data Visualization**: View 5-year closing price history charts
- **Technical Indicator Analysis**: Calculate and visualize key technical indicators:
  - Relative Strength Index (RSI)
  - Stochastic Oscillator
  - Williams %R
  - MACD (Moving Average Convergence Divergence)
  - Price Rate of Change
  - On Balance Volume
- **Machine Learning Prediction**: Random Forest classifier predicts if tomorrow's price will go UP ⬆️ or DOWN ⬇️
- **Performance Metrics**: Displays model accuracy for each stock
- **Data Export**: Download selected stock data as CSV

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock_analysis.git
cd stock_analysis



# Install required packages
pip install -r requirements.txt

# Open the web app
streamlit run stock.py
```

## Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
yfinance
scikit-learn
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Access the web application in your browser 

3. Use the sidebar to:
   - Select stocks from the S&P 500 list
   - Adjust the number of companies to display

4. Click "Show Plots" to generate:
   - 5-year price history charts
   - Technical indicator visualizations
   - Next-day price movement predictions

## How It Works

The application implements a machine learning pipeline that:

1. **Data Collection**: Fetches historical stock data using yfinance
2. **Feature Engineering**: Calculates technical indicators that traders commonly use:
   - RSI identifies overbought/oversold conditions
   - Stochastic Oscillator measures momentum
   - Williams %R shows market extremes
   - MACD identifies trend changes
   - Price Rate of Change measures price momentum
   - On Balance Volume connects price changes with volume

3. **Prediction Model**: Uses a Random Forest classifier to learn patterns from technical indicators and predict price movements
4. **Visualization**: Presents data and predictions through interactive charts and metrics

## Project Structure

```
sp500-stock-prediction/
├── stock.py          # Main Streamlit application
├── requirements.txt  # Required Python packages
├── README.md         # Project documentation
└── .gitignore        # Git ignore file
```

## Future Improvements

- Add more advanced technical indicators
- Implement additional machine learning models for comparison
- Add backtesting functionality to evaluate trading strategies
- Incorporate sentiment analysis from news and social media
- Create email alerts for predicted price movements

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Disclaimer

This application is for educational and informational purposes only. It is not intended to provide investment advice. Always conduct your own research or consult with a financial advisor before making investment decisions.
