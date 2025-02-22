import datetime
import requests
import pandas as pd
import time
from statsmodels.tsa.stattools import coint

class CryptoPairTrading:
    def __init__(self, pairs, initial_capital=1000, interval='1m', period=7, fee=0.002, mode="backtest"):
        self.pairs = pairs
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.interval = interval
        self.period = period
        self.fee = fee
        self.mode = mode  # "backtest" or "live"
        self.capital_history = []
    
    def fetch_btc_data(self, symbol, end_time=None):
        if end_time is None:
            end_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        start_time_ms = end_time - (self.period * 24 * 60 * 60 * 1000)

        url = "https://api.binance.com/api/v3/klines"
        all_data = []
        
        while start_time_ms < end_time:
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": start_time_ms,
                "endTime": end_time,
                "limit": 1000
            }

            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Error fetching data for {symbol}: {response.status_code}, {response.text}")
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

    def check_cointegration(self, df1, df2):
        if len(df1) == len(df2):
            score, p_value, _ = coint(df1['close'], df2['close'])
            return p_value
        return 1

    def backtest(self, df1, df2, capital_per_pair):
        spread = df1['close'] - df2['close']
        mean = spread.mean()
        std = spread.std()
        upper = mean + std
        lower = mean - std

        position1, position2 = 0, 0
        capital = capital_per_pair

        print(f"Starting backtest with initial capital: {capital_per_pair:.2f}")

        for i in range(1, len(spread)):
            if spread.iloc[i-1] > upper:
                position1, position2 = -1, 1  # Sell df1, Buy df2
                print(f"{df1.index[i]} | SHORT {df1.name}, LONG {df2.name} | Spread: {spread.iloc[i-1]:.2f}")
            elif spread.iloc[i-1] < lower:
                position1, position2 = 1, -1  # Buy df1, Sell df2
                print(f"{df1.index[i]} | LONG {df1.name}, SHORT {df2.name} | Spread: {spread.iloc[i-1]:.2f}")
            else:
                position1, position2 = 0, 0  # No position

            # Calculate returns
            ret1 = position1 * df1['close'].pct_change().iloc[i]
            ret2 = position2 * df2['close'].pct_change().iloc[i]
            trade_fee = self.fee * abs(position1 + position2)
            capital *= (1 + ret1 + ret2 - trade_fee)

            # Print capital after each trade
            print(f"Capital after trade: {capital:.2f}")

        return capital

    def backtest(self, df1, df2, capital_per_pair, pair1, pair2):
        spread = df1['close'] - df2['close']
        mean = spread.mean()
        std = spread.std()
        upper = mean + std
        lower = mean - std

        position1, position2 = 0, 0
        capital = capital_per_pair

        print(f"Starting backtest with initial capital: {capital_per_pair:.2f}")

        for i in range(1, len(spread)):
            if spread.iloc[i-1] > upper:
                position1, position2 = -1, 1  # Sell df1, Buy df2
                print(f"{df1.index[i]} | SHORT {pair1}, LONG {pair2} | Spread: {spread.iloc[i-1]:.2f}")
            elif spread.iloc[i-1] < lower:
                position1, position2 = 1, -1  # Buy df1, Sell df2
                print(f"{df1.index[i]} | LONG {pair1}, SHORT {pair2} | Spread: {spread.iloc[i-1]:.2f}")
            else:
                position1, position2 = 0, 0  # No position

            # Calculate returns
            ret1 = position1 * df1['close'].pct_change().iloc[i]
            ret2 = position2 * df2['close'].pct_change().iloc[i]
            trade_fee = self.fee * abs(position1 + position2)
            capital *= (1 + ret1 + ret2 - trade_fee)

            # Print capital after each trade
            print(f"Capital after trade: {capital:.2f}")

        return capital


    def run_live(self, refresh_interval=60):
        print("Starting live trading mode...")
        while True:
            try:
                now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
                
                # Set the rolling window: calculate start_time as 7 days prior to now
                start_time = now - (self.period * 24 * 60 * 60 * 1000)  # 7 days worth of data
                # Fetch data for all pairs within the rolling window
                cointegration_data = {pair: self.fetch_btc_data(pair, end_time=now) for pair in self.pairs}
                stationary_pairs = []

                # Loop through all pairs to check for cointegration
                for i in range(len(self.pairs)):
                    for j in range(i+1, len(self.pairs)):
                        df1, df2 = cointegration_data[self.pairs[i]], cointegration_data[self.pairs[j]]
                        p_value = self.check_cointegration(df1, df2)
                        if p_value < 0.05:  # Add stationary pairs based on p-value threshold
                            stationary_pairs.append((self.pairs[i], self.pairs[j], p_value))

                # If stationary pairs found, backtest or implement trading logic
                if stationary_pairs:
                    print("Stationary pairs detected:", stationary_pairs)

                    # Implement your logic for placing live trades here
                    for pair1, pair2, _ in stationary_pairs:
                        df1, df2 = cointegration_data[pair1], cointegration_data[pair2]
                        self.backtest(df1, df2, self.capital, pair1, pair2)  # Pass pair names to backtest method
                        
                else:
                    print("No trading opportunities found.")

                # Sleep for the refresh_interval before re-checking
                time.sleep(refresh_interval)

            except Exception as e:
                print("Error in live trading:", e)
                time.sleep(5)





# Example Usage:
pairs = ["AVAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "LTCUSDT", "UNIUSDT", "AAVEUSDT", 'XRPUSDT', 
         'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'XLMUSDT', 'TONUSDT', 'SUIUSDT', 'DOTUSDT']
strategy = CryptoPairTrading(pairs, mode="live")
# strategy.run_backtest(weeks=52)  # Uncomment to backtest
strategy.run_live()  # Runs live trading loop
