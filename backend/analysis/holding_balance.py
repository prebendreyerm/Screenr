import requests
import datetime
import numpy as np
import pandas as pd
import time

# Initial capital for backtesting
INITIAL_CAPITAL = 10000  # USD
POSITION_SIZE = 0.01  # BTC per trade
THRESHOLD = 0.002  # Mean reversion threshold (0.2% change)
LOOKBACK = 100  # Moving average period

# Fetch BTC data (1-minute candles)
def fetch_btc_data():
    end_time = int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)
    start_time = end_time - (7 * 24 * 60 * 60 * 1000)  # 7 days of data

    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    while start_time < end_time:
        params = {
            "symbol": "AVAUSDT",
            "interval": "1m",
            "startTime": start_time,
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
        start_time = data[-1][0] + 60000  # Move forward 1 minute
        time.sleep(0.5)

    timestamps = [datetime.datetime.fromtimestamp(candle[0] / 1000, tz=datetime.UTC) for candle in all_data]
    closing_prices = [float(candle[4]) for candle in all_data]
    
    return timestamps, closing_prices


# Market Making Parameters
SPREAD = 0.0005  # 0.05% spread
TRADE_SIZE = 0.01  # BTC per trade
CAPITAL = 1000  # USDT starting capital
INVENTORY = 0  # BTC holdings

# Convert into DataFrame for backtesting
timestamps, closing_prices = fetch_btc_data()
df = pd.DataFrame({"timestamp": timestamps, "price": closing_prices})

def market_making_backtest(df):
    """Simulate market-making strategy on historical BTC price data."""
    global CAPITAL, INVENTORY

    for i in range(len(df)):
        mid_price = df.loc[i, "price"]

        # Calculate buy and sell prices
        buy_price = mid_price * (1 - SPREAD)
        sell_price = mid_price * (1 + SPREAD)

        # Execute buy trade
        if CAPITAL >= buy_price * TRADE_SIZE:
            INVENTORY += TRADE_SIZE
            CAPITAL -= buy_price * TRADE_SIZE
            print(f"[{df.loc[i, 'timestamp']}] BUY {TRADE_SIZE} BTC at {buy_price:.2f}, New Balance: {CAPITAL:.2f} USDT, Inventory: {INVENTORY:.4f} BTC")

        # Execute sell trade
        if INVENTORY >= TRADE_SIZE:
            INVENTORY -= TRADE_SIZE
            CAPITAL += sell_price * TRADE_SIZE
            print(f"[{df.loc[i, 'timestamp']}] SELL {TRADE_SIZE} BTC at {sell_price:.2f}, New Balance: {CAPITAL:.2f} USDT, Inventory: {INVENTORY:.4f} BTC")

# Run backtest
market_making_backtest(df)
