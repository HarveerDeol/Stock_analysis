import os
import sys

import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, classification_report

st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
@st.cache_resource
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df

df = load_data()

# Sort stocks alphabetically
sorted_stocks = sorted(df['Symbol'].unique())

# Sidebar - Stock selection
selected_stocks = st.sidebar.multiselect('Select Stocks', sorted_stocks, sorted_stocks[:5])  # Default selects first 5

# Filter DataFrame by selected stocks
df_selected_stocks = df[df['Symbol'].isin(selected_stocks)]

st.header('Display Selected Companies')
st.write(f'Data Dimension: {df_selected_stocks.shape[0]} rows and {df_selected_stocks.shape[1]} columns.')
st.dataframe(df_selected_stocks)

# Function to allow CSV download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_stocks), unsafe_allow_html=True)

# Fetch stock data using yfinance
data = yf.download(
    tickers=list(df_selected_stocks.Symbol),
    period="5y",
    interval="1d",
    group_by='ticker',
    auto_adjust=True,
    prepost=True,
    threads=True,
    proxy=None
)











# Function to plot stock prices


def price_plot(symbol):
    df_stock = pd.DataFrame(data[symbol].Close)
    df_stock['Date'] = df_stock.index

    fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size
    ax.fill_between(df_stock.Date, df_stock.Close, color='skyblue', alpha=0.3)
    ax.plot(df_stock.Date, df_stock.Close, color='skyblue', alpha=0.8)

    # Format x-axis for better readability
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust date ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format as YYYY-MM
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels

    ax.set_title(f"{symbol} Closing Prices (5-Year)", fontweight='bold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Closing Price', fontweight='bold')

    return st.pyplot(fig)



def candle_plot(stock):
    df_stock = data[stock]


    full_price_history = []
    
    # Iterate through each row of the DataFrame for the stock symbol
    for index, row in df_stock.iterrows():
        candle = {
            'timestamp': index,  # Timestamp of the candle (date)
            'symbol': stock,    # Stock symbol
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }
        full_price_history.append(candle)


    price_data = pd.DataFrame(full_price_history)
    price_data['change_in_price'] = price_data['close'].diff()
    price_data = price_data[['symbol','timestamp','close','high','low','open','volume','change_in_price']]

    return price_data

def smooth_data(price_data):
    # define the number of days out you want to predict
    days_out = 30

    # Group by symbol, then apply the rolling function and grab the Min and Max.
    price_data_smoothed = price_data.groupby(['symbol'])[['close','low','high','open','volume']].transform(lambda x: x.ewm(span = days_out).mean())

    # Join the smoothed columns with the symbol and datetime column from the old data frame.
    smoothed_df = pd.concat([price_data[['symbol','timestamp','change_in_price']], price_data_smoothed], axis=1, sort=False)

    # create a new column that will house the flag, and for each group calculate the diff compared to 30 days ago. Then use Numpy to define the sign.
    smoothed_df['Signal_Flag'] = smoothed_df.groupby('symbol')['close'].transform(lambda x : np.sign(x.diff(days_out)))

    return smoothed_df


def calc_RSI(price_data):
    n = 14

    # First make a copy of the data frame twice
    up_df, down_df = price_data[['symbol','change_in_price']].copy(), price_data[['symbol','change_in_price']].copy()

    # For up days, if the change is less than 0 set to 0.
    up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0

    # For down days, if the change is greater than 0 set to 0.
    down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0

    # We need change in price to be absolute.
    down_df['change_in_price'] = down_df['change_in_price'].abs()

    # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
    ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())

    # Calculate the Relative Strength
    relative_strength = ewma_up / ewma_down

    # Calculate the Relative Strength Index
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    # Add the info to the data frame.
    price_data['down_days'] = down_df['change_in_price']
    price_data['up_days'] = up_df['change_in_price']
    price_data['RSI'] = relative_strength_index

    return price_data


def calc_stochastic_oscillator(price_data):
    n = 14

    # Make a copy of the high and low column.
    low_14, high_14 = price_data[['symbol','low']].copy(), price_data[['symbol','high']].copy()

    # Group by symbol, then apply the rolling function and grab the Min and Max.
    low_14 = low_14.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

    # Calculate the Stochastic Oscillator.
    k_percent = 100 * ((price_data['close'] - low_14) / (high_14 - low_14))

    # Add the info to the data frame.
    price_data['low_14'] = low_14
    price_data['high_14'] = high_14
    price_data['k_percent'] = k_percent

    return price_data


def calc_williams_R(price_data):
    n = 14

    # Make a copy of the high and low column.
    low_14, high_14 = price_data[['symbol','low']].copy(), price_data[['symbol','high']].copy()

    # Group by symbol, then apply the rolling function and grab the Min and Max.
    low_14 = low_14.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

    # Calculate William %R indicator.
    r_percent = ((high_14 - price_data['close']) / (high_14 - low_14)) * - 100

    # Add the info to the data frame.
    price_data['r_percent'] = r_percent

    return price_data


def calc_MACD(price_data):
    # Calculate the MACD
    ema_26 = price_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = price_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26

    # Calculate the EMA
    ema_9_macd = macd.ewm(span = 9).mean()

    # Store the data in the data frame.
    price_data['MACD'] = macd
    price_data['MACD_EMA'] = ema_9_macd

    return price_data


def calc_price_rate_of_change(price_data):
    n = 9

    # Calculate the Rate of Change in the Price, and store it in the Data Frame.
    price_data['Price_Rate_Of_Change'] = price_data.groupby('symbol')['close'].transform(lambda x: x.pct_change(periods = n))

    return price_data


def calc_on_balance_volume(price_data):
    # Calculate price change
    price_data['change_in_price'] = price_data['close'].diff()

    # Initialize OBV column
    obv_values = [0]  # Start with OBV = 0

    # Calculate OBV iteratively
    for i in range(1, len(price_data)):
        if price_data.loc[i, 'change_in_price'] > 0:
            obv_values.append(obv_values[-1] + price_data.loc[i, 'volume'])
        elif price_data.loc[i, 'change_in_price'] < 0:
            obv_values.append(obv_values[-1] - price_data.loc[i, 'volume'])
        else:
            obv_values.append(obv_values[-1])

    # Assign OBV values to the DataFrame
    price_data['On Balance Volume'] = obv_values

    return price_data


def price_change(price_data):
    # Compute price change only for 'close' column
    price_data = price_data.dropna()

    price_data['Prediction'] = np.sign(price_data['close'].diff())

    # Convert 0s to 1s for binary classification
    price_data.loc[price_data['Prediction'] == 0.0, 'Prediction'] = 1.0

    price_data = price_data.dropna()

    price_data.to_csv('final_metrics.csv')

    return price_data


# building the model 
def split_data(price_data):
    price_data = candle_plot(price_data)
    price_data = smooth_data(price_data)
    price_data = calc_RSI(price_data)
    price_data = calc_stochastic_oscillator(price_data)
    price_data = calc_williams_R(price_data)
    price_data = calc_MACD(price_data)
    price_data = calc_price_rate_of_change(price_data)
    price_data = calc_on_balance_volume(price_data)
    price_data = price_change(price_data)

    X_Cols = price_data[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD','On Balance Volume']]
    Y_Cols = price_data['Prediction']

    # Split X and y into X_
    X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)

    # Create a Random Forest Classifier
    rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

    # Fit the data to the model
    rand_frst_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = rand_frst_clf.predict(X_test)
    # Compute accuracy
    accuracy = accuracy_score(y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0

    # Return the accuracy as a string
    return f'Correct Prediction (%): {accuracy:.2f}'














# Sidebar slider for number of stocks to plot
num_company = st.sidebar.slider('Number of Companies to Plot', 1, len(selected_stocks), 3)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    # Initialize the list to store candle data


    # Iterate over the selected stocks
    for stock in selected_stocks[:num_company]:
        price_plot(stock)
        result = split_data(stock)  # Call the function and store the result
        st.markdown(result)
