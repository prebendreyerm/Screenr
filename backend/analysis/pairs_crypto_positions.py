import datetime
import requests
import pandas as pd
import time
from statsmodels.tsa.stattools import coint

class PairTrading:
    def __init__(self, pairs, initial_capital=1000, fee=0.002, weeks=52):
        self.pairs = pairs
        self.initial_capital = initial_capital
        self.fee = fee
        self.weeks = weeks
        self.capital = initial_capital
        self.capital_history = []
        self.now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        self.rolling_data = {}  # Store rolling price data
        self.positions = {}  # Store positions for each pair
        self.active_pairs = []  # List of active stationary pairs
        self.capital_per_pair = {}  # Capital allocated to each pair

    def fetch_btc_data(self, symbol, interval='1m', period=7, end_time=None):
        """Fetches historical data for the given period."""
        if end_time is None:
            end_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        start_time_ms = end_time - (period * 24 * 60 * 60 * 1000)

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
                print(f"Error fetching data for {symbol}: {response.status_code}, {response.text}")
                break

            data = response.json()
            if not data:
                break

            all_data.extend(data)
            start_time_ms = data[-1][0] + 60000  # Move forward 1 minute

        timestamps = [datetime.datetime.fromtimestamp(candle[0] / 1000, tz=datetime.timezone.utc) for candle in all_data]
        closing_prices = [float(candle[4]) for candle in all_data]

        df = pd.DataFrame({'timestamp': timestamps, 'close': closing_prices})
        df.set_index('timestamp', inplace=True)

        return df

    def initialize_rolling_data(self):
        """Fetches the last 7 days of data for all pairs and stores them in rolling_data."""
        print("\nInitializing rolling window with historical data...")
        for pair in self.pairs:
            self.rolling_data[pair] = self.fetch_btc_data(pair, period=7)
            self.positions[pair] = {"long": 0, "short": 0}  # Initialize positions

    def update_rolling_data(self, symbol, new_price):
        """Updates the rolling window by appending new price and removing the oldest."""
        now = datetime.datetime.now(datetime.timezone.utc)
        if symbol in self.rolling_data:
            df = self.rolling_data[symbol]
            df.loc[now] = new_price  # Append new data point

            # Keep only the last 7 days of data
            cutoff_time = now - datetime.timedelta(days=7)
            self.rolling_data[symbol] = df[df.index >= cutoff_time].copy()

    def check_cointegration(self, df1, df2):
        """Runs a cointegration test on two time series."""
        if len(df1) == len(df2):
            _, p_value, _ = coint(df1['close'], df2['close'])
            return p_value
        return 1  # Non-stationary if length mismatch

    def run_cointegration_test(self):
        """Checks cointegration on updated rolling window for all pairs."""
        # print("\nRunning cointegration test on updated data...\n")
        
        stationary_pairs = []
        for i in range(len(self.pairs)):
            for j in range(i+1, len(self.pairs)):
                df1, df2 = self.rolling_data[self.pairs[i]], self.rolling_data[self.pairs[j]]
                p_value = self.check_cointegration(df1, df2)
                if p_value < 0.2:
                    stationary_pairs.append((self.pairs[i], self.pairs[j], p_value))

        # print(f"Found {len(stationary_pairs)} stationary pairs.")
        # for pair in stationary_pairs:
        #     print(f"{pair[0]} - {pair[1]}: p-value = {pair[2]:.6f}")

        return stationary_pairs

    def get_binance_last_price(self, symbol):
        """Fetches the latest price of a symbol from Binance."""
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        if response.status_code == 200:
            return float(response.json()['price'])
        else:
            print(f"Error fetching last price: {response.status_code}, {response.text}")
            return None

    def open_position(self, pair, position_type, amount):
        """Open a long or short position."""
        self.positions[pair][position_type] += amount
        print(f"Opening {position_type} position for {pair} with amount {amount}")

    def close_position(self, pair, position_type):
        """Close a long or short position."""
        self.positions[pair][position_type] = 0
        print(f"Closing {position_type} position for {pair}")

    def allocate_capital(self, active_pairs):
        """Allocates capital evenly among active pairs."""
        if not active_pairs:
            return {}
        capital_per_pair = self.capital / len(active_pairs)
        return {pair: capital_per_pair for pair in active_pairs}

    def fetch_price_continuously(self):
        """Fetches live prices, updates rolling data, and runs cointegration tests in real-time."""
        print("\nFetching live prices...\n")

        # Initialize rolling window with historical data
        self.initialize_rolling_data()

        # Print header row
        print("Timestamp".ljust(20) + " ".join(symbol.ljust(10) for symbol in self.pairs))
        print("=" * (20 + len(self.pairs) * 11))

        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            prices = []

            for symbol in self.pairs:
                price = self.get_binance_last_price(symbol)
                if price is not None:
                    self.update_rolling_data(symbol, price)  # Update rolling data
                prices.append(f"{price:.6f}" if price is not None else "N/A")

            # Print row with timestamp and latest prices
            print(timestamp.ljust(20) + " ".join(price.ljust(10) for price in prices))

            # Run cointegration test after updating rolling window
            stationary_pairs = self.run_cointegration_test()
            self.active_pairs = [pair[0] for pair in stationary_pairs]  # Store active pairs

            # Allocate capital to active pairs
            self.capital_per_pair = self.allocate_capital(self.active_pairs)

            # Open/close positions based on cointegration and spread conditions
            for pair in stationary_pairs:
                symbol1, symbol2 = pair[0], pair[1]
                df1, df2 = self.rolling_data[symbol1], self.rolling_data[symbol2]
                
                # Reindex to ensure same timestamps
                df1, df2 = df1.reindex(df1.index.union(df2.index)).interpolate(), df2.reindex(df1.index.union(df2.index)).interpolate()
                
                spread = df1['close'] - df2['close']
                mean = spread.mean()
                std = spread.std()
                upper = mean + 2*std
                lower = mean - 2*std
                print(f"Spread for {symbol1} - {symbol2}: {spread.iloc[-1]:.2f}, upper: {upper:.2f}, lower: {lower:.2f}")

                # Check if there is enough capital to open positions
                capital_available = self.capital_per_pair.get(symbol1, 0) + self.capital_per_pair.get(symbol2, 0)

                # Open positions based on spread thresholds and available capital
                if spread.iloc[-1] > upper and self.positions[symbol1]['long'] == 0 and self.positions[symbol2]['short'] == 0 and capital_available > 0:
                    amount = capital_available / 2  # Split capital evenly between the positions
                    self.open_position(symbol1, 'short', amount)
                    self.open_position(symbol2, 'long', amount)
                elif spread.iloc[-1] < lower and self.positions[symbol1]['short'] == 0 and self.positions[symbol2]['long'] == 0 and capital_available > 0:
                    amount = capital_available / 2  # Split capital evenly between the positions
                    self.open_position(symbol1, 'long', amount)
                    self.open_position(symbol2, 'short', amount)
                else:
                    # Close positions when spread is within mean range
                    if self.positions[symbol1]['long'] > 0:
                        self.close_position(symbol1, 'long')
                    if self.positions[symbol2]['short'] > 0:
                        self.close_position(symbol2, 'short')
                    if self.positions[symbol1]['short'] > 0:
                        self.close_position(symbol1, 'short')
                    if self.positions[symbol2]['long'] > 0:
                        self.close_position(symbol2, 'long')

            time.sleep(20)  # Wait before fetching again

# Example usage
if __name__ == "__main__":
    pairs = ['AVAUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT', 'UNIUSDT', 'AAVEUSDT', 'XRPUSDT', 
             'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'XLMUSDT', 'TONUSDT', 'SUIUSDT', 'DOTUSDT']

    backtester = PairTrading(pairs, initial_capital=1000, weeks=52)
    backtester.fetch_price_continuously()
