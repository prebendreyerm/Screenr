import datetime
import requests
import pandas as pd
import time
from statsmodels.tsa.stattools import coint
import statsmodels as sm

class PairTrading:
    def __init__(self, pairs, initial_capital=1000, fee=0.002, weeks=52):
        self.pairs = pairs
        self.initial_capital = initial_capital
        self.fee = fee
        self.weeks = weeks
        self.capital = initial_capital  # Total capital
        self.available_capital = initial_capital  # Capital available for trading
        self.capital_history = []
        self.now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        self.rolling_data = {}
        self.positions = {}
        self.active_pairs = []
        self.capital_per_pair = {}

    def fetch_btc_data(self, symbol, interval='1m', period=7, end_time=None):
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
                return pd.DataFrame()  # Return empty DataFrame on error

            data = response.json()
            if not data:
                break

            all_data.extend(data)
            start_time_ms = data[-1][0] + 60000

        timestamps = [datetime.datetime.fromtimestamp(candle[0] / 1000, tz=datetime.timezone.utc) for candle in all_data]
        closing_prices = [float(candle[4]) for candle in all_data]

        df = pd.DataFrame({'timestamp': timestamps, 'close': closing_prices})
        df.set_index('timestamp', inplace=True)

        return df

    def initialize_rolling_data(self):
        print("\nInitializing rolling window with historical data...")
        for pair in self.pairs:
            self.rolling_data[pair] = self.fetch_btc_data(pair, period=7)
            self.positions[pair] = {"long": {'amount': 0, 'entry_price': 0}, "short": {'amount': 0, 'entry_price': 0}}

    def update_rolling_data(self, symbol, new_price):
        now = datetime.datetime.now(datetime.timezone.utc)
        if symbol in self.rolling_data:
            df = self.rolling_data[symbol]
            df.loc[now] = new_price

            cutoff_time = now - datetime.timedelta(days=7)
            self.rolling_data[symbol] = df[df.index >= cutoff_time].copy()

    def check_cointegration(self, df1, df2):
        if len(df1) == len(df2):
            _, p_value, _ = coint(df1['close'], df2['close'])
            return p_value
        return 1

    def run_cointegration_test(self):
        stationary_pairs = []
        for i in range(len(self.pairs)):
            for j in range(i + 1, len(self.pairs)):
                df1, df2 = self.rolling_data[self.pairs[i]], self.rolling_data[self.pairs[j]]
                p_value = self.check_cointegration(df1, df2)
                if p_value < 0.2:
                    stationary_pairs.append((self.pairs[i], self.pairs[j], p_value))
        return stationary_pairs

    def get_binance_last_price(self, symbol):
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            return float(response.json()['price'])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching last price for {symbol}: {e}")
            return None

    def open_position(self, pair, position_type, amount):
        if amount > self.available_capital:
            print(f"Not enough available capital to open position for {pair}. Available: {self.available_capital}, Required: {amount}")
            return

        self.capital_per_pair[pair] = self.capital_per_pair.get(pair, 0) + amount
        self.available_capital -= amount

        current_price = self.get_binance_last_price(pair)
        self.positions[pair][position_type] = {'amount': amount, 'entry_price': current_price}

        print(f"Opening {position_type} position for {pair} with amount {amount} at price {current_price}")

    def close_position(self, pair, position_type):
        position = self.positions[pair][position_type]
        if position['amount'] == 0:
            print(f"No {position_type} position to close for {pair}")
            return

        current_price = self.get_binance_last_price(pair)
        entry_price = position['entry_price']
        profit_loss = (current_price - entry_price) * position['amount'] if position_type == 'long' else (entry_price - current_price) * position['amount']

        self.capital += profit_loss
        self.available_capital += position['amount']
        self.capital_per_pair[pair] -= position['amount']

        self.positions[pair][position_type] = {'amount': 0, 'entry_price': 0}

        print(f"Closing {position_type} position for {pair} with P/L: {profit_loss}")

    def allocate_capital(self, active_pairs):
        if not active_pairs or self.available_capital <= 0:
            return {}

        capital_per_pair = self.available_capital / len(active_pairs) if active_pairs else 0
        allocated_capital = {}

        for pair in active_pairs:
            allocated_capital[pair] = capital_per_pair
        return allocated_capital

    def calculate_total_capital(self):
        return self.capital

    def fetch_price_continuously(self):
        print("\nFetching live prices...\n")

        self.initialize_rolling_data()

        print("Timestamp".ljust(20) + " ".join(symbol.ljust(10) for symbol in self.pairs))
        print("=" * (20 + len(self.pairs) * 11))

        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            prices = []

            total_capital = self.calculate_total_capital()
            print(f"Total capital at {timestamp}: {total_capital:.2f}, Available Capital: {self.available_capital:.2f}")

            for symbol in self.pairs:
                price = self.get_binance_last_price(symbol)
                if price is not None:
                    self.update_rolling_data(symbol, price)
                prices.append(f"{price:.6f}" if price is not None else "N/A")

            print(timestamp.ljust(20) + " ".join(price.ljust(10) for price in prices))

            stationary_pairs = self.run_cointegration_test()
            self.active_pairs = [(pair[0], pair[1]) for pair in stationary_pairs]
            self.capital_per_pair = self.allocate_capital(self.active_pairs)

            for pair in stationary_pairs:
                symbol1, symbol2 = pair[0], pair[1]
                pair_tuple = (symbol1, symbol2)
                df1, df2 = self.rolling_data[symbol1], self.rolling_data[symbol2]

                # Ensure both DataFrames have the same indices before calculating spread
                common_index = df1.index.intersection(df2.index)
                df1 = df1.loc[common_index]
                df2 = df2.loc[common_index]

                spread = df1['close'] - df2['close']
                mean = spread.mean()
                std = spread.std()
                upper = mean + 2 * std
                lower = mean - 2 * std
                upper_exit = mean + 0.5 * std
                lower_exit = mean - 0.5 * std

                capital_available = self.capital_per_pair.get(pair_tuple, 0)

                print(f'spread: {spread.iloc[-1]}, upper: {upper}, lower: {lower}, upper_exit: {upper_exit}, lower_exit: {lower_exit}, capital: {capital_available}')

                # Check if positions are already open BEFORE attempting to open new ones
                if self.positions[symbol1]['long']['amount'] == 0 and self.positions[symbol2]['short']['amount'] == 0:
                    if spread.iloc[-1] > upper and capital_available > 0:
                        amount = capital_available / 2
                        self.open_position(symbol1, 'short', amount)
                        self.open_position(symbol2, 'long', amount)
                        self.capital_per_pair[pair_tuple] = 0  # Deduct used capital

                    elif spread.iloc[-1] < lower and capital_available > 0:
                        amount = capital_available / 2
                        self.open_position(symbol1, 'long', amount)
                        self.open_position(symbol2, 'short', amount)
                        self.capital_per_pair[pair_tuple] = 0  # Deduct used capital
                else:
                    if self.positions[symbol1]['long']['amount'] > 0 and lower_exit < spread.iloc[-1] < upper_exit:
                        print(f"Spread for {symbol1} - {symbol2}: {spread.iloc[-1]:.2f}, upper: {upper:.2f}, lower: {lower:.2f}")
                        self.close_position(symbol1, 'long')
                        self.close_position(symbol2, 'short') # Close the other position as well
                    if self.positions[symbol1]['short']['amount'] > 0 and lower_exit < spread.iloc[-1] < upper_exit:
                        print(f"Spread for {symbol1} - {symbol2}: {spread.iloc[-1]:.2f}, upper: {upper:.2f}, lower: {lower:.2f}")
                        self.close_position(symbol1, 'short')
                        self.close_position(symbol2, 'long') # Close the other position as well

            time.sleep(26)

if __name__ == "__main__":
    pairs = ['AVAUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT', 'UNIUSDT', 'AAVEUSDT', 'XRPUSDT', 
             'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'XLMUSDT', 'TONUSDT', 'SUIUSDT', 'DOTUSDT']

    backtester = PairTrading(pairs, initial_capital=1000, weeks=52)
    backtester.fetch_price_continuously()