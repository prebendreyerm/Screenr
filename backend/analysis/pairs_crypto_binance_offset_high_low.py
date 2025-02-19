import datetime
import requests
import pandas as pd
import time
from statsmodels.tsa.stattools import coint

# Fetch BTC data (1-minute candles)
def fetch_btc_data(symbol, interval='1m', period=7, end_time=None):
    if end_time is None:
        end_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
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
    open_prices = [float(candle[1]) for candle in all_data]  # Open price
    high_prices = [float(candle[2]) for candle in all_data]
    low_prices = [float(candle[3]) for candle in all_data]
    close_prices = [float(candle[4]) for candle in all_data]  # Close price
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    df.set_index('timestamp', inplace=True)

    return df

# Function to check for cointegration
def check_cointegration(df1, df2):
    if len(df1) == len(df2):
        score, p_value, _ = coint(df1['close'], df2['close'])
        return p_value
    return 1  # Return non-stationary if length mismatch

# Backtest function with adjusted entry/exit thresholds and capital tracking
def backtest(df1, df2, capital_per_pair, fee=0.002):
    spread = df1['close'] - df2['close']
    mean = spread.mean()
    std = spread.std()
    entry_upper = mean + 2 * std
    entry_lower = mean - 2 * std
    exit_upper = mean + 0.5 * std
    exit_lower = mean - 0.5 * std

    capital = capital_per_pair
    position1 = 0
    position2 = 0
    entry_price1 = 0
    entry_price2 = 0

    for i in range(1, len(spread)):
        # Exit condition: spread reverts to within exit range
        # Exit condition: spread reverts to within exit range
        if position1 != 0 and position2 != 0 and exit_lower < spread.iloc[i-1] < exit_upper:
            if position1 == 1:  # Long position exit
                exit_price1 = df1['low'].iloc[i]   # Sell at low
                exit_price2 = df2['high'].iloc[i]  # Buy at high
            else:  # Short position exit
                exit_price1 = df1['high'].iloc[i]  # Buy at high
                exit_price2 = df2['low'].iloc[i]   # Sell at low

            # Calculate returns based on entry and exit prices
            ret1 = position1 * ((exit_price1 - entry_price1) / entry_price1)
            ret2 = position2 * ((exit_price2 - entry_price2) / entry_price2)
            trade_fee = fee * (abs(position1) + abs(position2))  # Fee on both entry & exit

            capital *= (1 + ret1 + ret2 - trade_fee)

            # Reset position after exit
            position1 = 0
            position2 = 0
            entry_price1 = 0
            entry_price2 = 0

        # Entry condition
        if spread.iloc[i-1] > entry_upper and position1 == 0 and position2 == 0:  
            # Sell first, buy second
            position1 = -1
            position2 = 1
            entry_price1 = df1['high'].iloc[i]  # Enter short at high
            entry_price2 = df2['low'].iloc[i]   # Enter long at low
        elif spread.iloc[i-1] < entry_lower and position1 == 0 and position2 == 0:  
            # Buy first, sell second
            position1 = 1
            position2 = -1
            entry_price1 = df1['low'].iloc[i]  # Enter long at low
            entry_price2 = df2['high'].iloc[i]  # Enter short at high

        print(f"Week {i} - Spread: {spread.iloc[i-1]}, Position1: {position1}, Position2: {position2}, Capital: {capital:.2f}")

    return capital


# List of pairs
pairs = ["AVAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "LTCUSDT", "UNIUSDT", "AAVEUSDT", 'XRPUSDT', 
         'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'XLMUSDT', 'TONUSDT', 'SUIUSDT', 'DOTUSDT']

now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
weeks = 75
initial_capital = 1000
capital = initial_capital
capital_history = []

for week in range(weeks):
    end_time_cointegration = now - ((weeks - week) * 7 * 24 * 60 * 60 * 1000)
    end_time_backtest = end_time_cointegration + (7 * 24 * 60 * 60 * 1000)
    
    # Fetch data for cointegration check
    cointegration_data = {pair: fetch_btc_data(pair, period=7, end_time=end_time_cointegration) for pair in pairs}
    
    # Check for cointegration
    stationary_pairs = []
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            pair1, pair2 = pairs[i], pairs[j]
            df1, df2 = cointegration_data[pair1], cointegration_data[pair2]

            p_value = check_cointegration(df1, df2)
            if p_value < 0.05:
                stationary_pairs.append((pair1, pair2, p_value))
    
    # Fetch data for backtesting
    backtest_data = {pair: fetch_btc_data(pair, period=7, end_time=end_time_backtest) for pair in pairs}
    
    # Distribute capital among pairs
    num_pairs = len(stationary_pairs)
    if num_pairs > 0:
        capital_per_pair = capital / num_pairs
    else:
        capital_per_pair = 0  # No trades this week
    
    total_final_capital = 0
    for pair1, pair2, _ in stationary_pairs:
        final_cap = backtest(backtest_data[pair1], backtest_data[pair2], capital_per_pair)
        total_final_capital += final_cap
    
    # Update capital for next week
    capital = total_final_capital if num_pairs > 0 else capital
    capital_history.append((week + 1, initial_capital, capital))
    
    print(f"Week {week + 1}: Start Capital = {initial_capital:.2f}, End Capital = {capital:.2f}")
    initial_capital = capital  # Carry forward capital

# Print summary
print("\nFinal Summary")
for entry in capital_history:
    print(f"Week {entry[0]}: Start = {entry[1]:.2f}, End = {entry[2]:.2f}")