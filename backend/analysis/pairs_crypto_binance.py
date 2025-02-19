import datetime
import requests
import pandas as pd
import time
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt



# Fetch BTC data (1-minute candles) once and store in memory
def fetch_btc_data(symbol, interval='1m', period=7, end_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)):
    start_time_ms = end_time - (period * 24 * 60 * 60 * 1000)  # Period in milliseconds

    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    while start_time_ms < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time,
            "limit": 1000
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}, {response.text}")
            break
        
        data = response.json()
        if not data:
            break

        all_data.extend(data)
        start_time_ms = data[-1][0] + 60000  # Move forward 1 minute

    timestamps = [datetime.datetime.fromtimestamp(candle[0] / 1000, tz=datetime.timezone.utc) for candle in all_data]
    closing_prices = [float(candle[4]) for candle in all_data]
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': closing_prices
    })
    df.set_index('timestamp', inplace=True)

    return df

# Function to check for cointegration
def check_cointegration(df1, df2):
    if len(df1) == len(df2):
        score, p_value, _ = coint(df1['close'], df2['close'])
        return p_value
    return 1  # Return non-stationary if length mismatch

# Backtest function
def backtest(df1, df2, pair1, pair2):
    spread = df1['close'] - df2['close']
    mean = spread.mean()
    std = spread.std()
    upper = mean + std
    lower = mean - std

    positions = []
    for i in range(1, len(spread)):
        if spread.iloc[i-1] > upper:  # Sell first, buy second
            positions.append(-1)
        elif spread.iloc[i-1] < lower:  # Buy first, sell second
            positions.append(1)
        else:
            positions.append(0)

    df1['position'] = [0] + positions
    df2['position'] = -df1['position']  # Opposite positions
    df1['returns'] = df1['position'].shift(1) * df1['close'].pct_change()
    df2['returns'] = df2['position'].shift(1) * df2['close'].pct_change()
    df1['strategy'] = df1['returns'] + df2['returns']

    df1['strategy'].cumsum().plot()
    plt.title(f'Backtest for {pair1} and {pair2}')
    plt.show()

# List of pairs
pairs = ["AVAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "LTCUSDT", "UNIUSDT", "AAVEUSDT", 'XRPUSDT', 
         'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'XLMUSDT', 'TONUSDT', 'SUIUSDT', 'DOTUSDT']

# Fetch all data at once and store in memory
start_total_time = time.time()
data_cache = {}

for pair in pairs:
    start_time = time.time()
    data_cache[pair] = fetch_btc_data(pair)
    elapsed_time = time.time() - start_time
    print(f"Fetching {pair} data took {elapsed_time:.2f} seconds")

# Check for cointegration
stationary_pairs = []
for i in range(len(pairs)):
    for j in range(i+1, len(pairs)):
        pair1, pair2 = pairs[i], pairs[j]
        df1, df2 = data_cache[pair1], data_cache[pair2]

        p_value = check_cointegration(df1, df2)
        if p_value < 0.05:
            stationary_pairs.append((pair1, pair2, p_value))

# Print results
print("Stationary Pairs:")
for pair in stationary_pairs:
    print(f"{pair[0]} and {pair[1]} - p-value: {pair[2]}")

# Backtest stationary pairs
for pair1, pair2, _ in stationary_pairs:
    backtest(data_cache[pair1], data_cache[pair2], pair1, pair2)

total_elapsed_time = time.time() - start_total_time
print(f"Total execution time: {total_elapsed_time:.2f} seconds")
