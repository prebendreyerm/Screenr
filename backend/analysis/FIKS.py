import sys
import pandas as pd
import numpy as np
import os
import requests
import time
import datetime
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from matplotlib import pyplot as plt

class PairTrading:
    def __init__(self, pairs, initial_capital=1000, fee=0.001, weeks=52):
        '''
        Initializing function of the class
        '''
        self.pairs = pairs
        self.initial_capital = initial_capital
        self.fee = fee
        self.weeks = weeks
        self.capital = initial_capital  # Total capital
        self.total_capital = initial_capital
        self.now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        self.rolling_data = {}
        self.positions = {}
        self.active_pairs = {}
        self.capital_per_pair = {}
        self.last_prices = {}
        self.last_rolling_update = {}
        self.historical_data = {}
    
    def fetch_last_price(self, symbol):
        '''
        Function to fetch the most recent price of a given asset.
        '''
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        try:
            response = requests.get(url, timeout = 1)
            response.raise_for_status()
            data = response.json()

            # Update and return the latest price
            self.last_prices[symbol] = data['price']
            return data
        
        except requests.exceptions.RequestException as e:
            print(f'Error fetching last price data for {symbol}: {e}')
            

            # Return last known price if available
            if symbol in self.last_prices:
                print(f'Returning last known price for {symbol}: {self.last_prices[symbol]}')
                return {'symbol': symbol, 'price': self.last_prices[symbol]}
            return None
    
    def fetch_price_continuous(self, symbol):
        '''
        A continuous function - simply calling the fetch_last_price function continuously, *** MIGHT BE REMOVED ***
        '''
        while True:
            price = self.fetch_last_price(symbol)
            if price:
                print(f"{symbol}: {price['price']}")
            else:
                print(f"Failed to fetch prices for pair: {symbol}")
                
    def fetch_historical_prices(self, symbol):
        '''
        Function to fetch the previous seven days at the time of initialization.
        '''
        end_time = int(time.time() * 1000)
        start_time = end_time - 30 * 24 * 60 * 60 * 1000  # 30 days in milliseconds
        url = f"https://api.binance.com/api/v3/klines"
        limit = 1000  # Binance API limit for klines
        all_data = []

        while start_time < end_time:
            params = {
                'symbol': symbol,
                'interval': '1h',
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                # Convert the data into a pandas DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'quote_asset_volume', 'number_of_trades', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                all_data.append(df[['close']])
                
                # Update the start_time for the next batch
                start_time = int(df.index[-1].timestamp() * 1000) + 1  # Add 1 ms to avoid overlap
            
            except requests.exceptions.RequestException as e:
                print(f'Error fetching historical price data for {symbol}: {e}')
                break
        
        if all_data:
            return pd.concat(all_data)
        else:
            return None
        
    def fetch_historical_prices_backtest(self, symbol, end_time):
        '''
        Function to fetch historical prices at 1-minute intervals for the past year.
        '''
        # Simulate fetching data a year ago
        start_time = end_time - 365 * 24 * 60 * 60 * 1000  # 30 days in milliseconds
        url = f"https://api.binance.com/api/v3/klines"
        limit = 1000
        all_data = []

        while start_time < end_time:
            params = {
                'symbol': symbol,
                'interval': '1m',
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'quote_asset_volume', 'number_of_trades', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                all_data.append(df[['close']])
                
                start_time = int(df.index[-1].timestamp() * 1000) + 1
            
            except requests.exceptions.RequestException as e:
                print(f'Error fetching historical price data for {symbol}: {e}')
                break
        
        if all_data:
            return pd.concat(all_data)
        else:
            return None
        
    def rolling_window(self, symbol):
        '''
        Function to update the rolling window with the latest price.
        Ensures updates occur only if at least an hour has passed.
        '''
        current_time = time.time()

        # If first time, initialize the rolling data
        if symbol not in self.rolling_data or self.rolling_data[symbol] is None:
            print(f"[INIT] Fetching historical data for {symbol}")
            self.rolling_data[symbol] = self.fetch_historical_prices(symbol)
            self.last_rolling_update[symbol] = current_time
            return

        # Ensure updates occur only if at least an hour has passed
        if current_time - self.last_rolling_update[symbol] < 3600:
            return

        # Fetch the latest price
        next_price_data = self.fetch_last_price(symbol)
        if next_price_data is None:
            return

        next_price = float(next_price_data['price'])
        new_timestamp = pd.to_datetime(current_time, unit='s')  # Convert time.time() correctly

        # Append the new price to the DataFrame
        new_row = pd.DataFrame({'close': [next_price]}, index=[new_timestamp])
        self.rolling_data[symbol] = pd.concat([self.rolling_data[symbol], new_row])

        # Maintain a max window size (30 days * 24 hours)
        if len(self.rolling_data[symbol]) > 30 * 24:
            self.rolling_data[symbol] = self.rolling_data[symbol].iloc[1:]

        # Update last update timestamp
        self.last_rolling_update[symbol] = current_time

    def rolling_window_backtest(self, symbol, timestamp):
        '''
        Update rolling window with the latest price during backtest.
        Ensures updates occur sequentially and only if at least an hour has passed.
        '''
        if symbol not in self.rolling_data:
            # Initialize rolling data with 1-hour intervals (resampled)
            self.rolling_data[symbol] = (
                self.historical_data[symbol]
                .iloc[:43199]  # Take the first 30 days * 24 hours of data
                .resample('1h')  # Resample to 1-hour intervals
                .ffill()  # Forward-fill missing values
            ).copy()

            # Ensure the first value is not NaN after resampling
            if self.rolling_data[symbol].isnull().values.any():
                self.rolling_data[symbol].bfill(inplace=True)  # Back-fill if necessary@

            self.last_rolling_update[symbol] = self.rolling_data[symbol].index[-1]  # Store last timestamp
            return

        # Ensure at least an hour has passed before updating
        if (timestamp - self.last_rolling_update[symbol]).total_seconds() < 3600:
            return  # Skip update if not enough time has passed

        # Add new data point to rolling window
        if timestamp in self.historical_data[symbol].index:
            next_price = self.historical_data[symbol].loc[timestamp]['close']
            new_row = pd.DataFrame({'close': [next_price]}, index=[timestamp])
            self.rolling_data[symbol] = pd.concat([self.rolling_data[symbol], new_row])

            # Maintain a max window size (30 days * 24 hours)
            if len(self.rolling_data[symbol]) > 30 * 24:
                self.rolling_data[symbol] = self.rolling_data[symbol].iloc[1:]

            # Update last update timestamp
            self.last_rolling_update[symbol] = timestamp

    def check_cointegration(self, symbol1, symbol2):
        '''
        Function to check the cointegration between two assets, symbol1 and symbol2 should be a single column of prices.
        For instance, symbol1 = self.rolling_window(symbol1)['close']
        '''
        price1 = self.rolling_data[symbol1]['close']
        price2 = self.rolling_data[symbol2]['close']

        _, p_value, _ = coint(price1, price2)
        if p_value < 0.05:
            return True
        else:
            return False

    def check_active_pairs(self, symbol1, symbol2):
        '''
        Function to identify active pairs and store it within the active pairs dictionary with timestamps.
        If the pair is already active, it updates the timestamp.
        '''
        pair = (symbol1, symbol2)
        if self.check_cointegration(symbol1, symbol2):
            if pair in self.active_pairs:
                # print(f"Pair {symbol1}-{symbol2} is already active. Updating timestamp.")
                return
            else:
                print(f"Pair {symbol1}-{symbol2} added to active pairs.")

            # Update or add the pair with the current timestamp
            self.active_pairs[pair] = datetime.datetime.now()
        # else:
        #     print(f"Pair {symbol1}-{symbol2} is not cointegrated.")

    def remove_expired_pairs(self):
        '''
        Function to remove pairs that have been active for more than seven days.
        '''
        current_time = datetime.datetime.now()
        to_remove = []
        
        for pair, timestamp in self.active_pairs.items():
            if (current_time - timestamp).days >= 7:
                to_remove.append(pair)
        
        for pair in to_remove:
            del self.active_pairs[pair]
            print(f"Pair {pair} removed from active pairs due to expiration.")
    
    def open_position(self, symbol1, symbol2):
        '''
        Function to open a position in a given asset with the available capital based on the number of active pairs.
        '''
        pair = (symbol1, symbol2)
        if pair in self.positions:
            print(f"Position already open for pair {symbol1}-{symbol2}.")
            return
        
        num_active_pairs = len(self.active_pairs)
        if num_active_pairs == 0:
            print("No active pairs to allocate capital.")
            return
        
        capital_per_pair = self.capital / num_active_pairs
        long_capital = capital_per_pair / 2
        short_capital = capital_per_pair / 2

        # Fetch the latest prices for both symbols
        price1 = float(self.fetch_last_price(symbol1)['price'])
        price2 = float(self.fetch_last_price(symbol2)['price'])

        # Calculate the amount of each asset to buy/sell
        long_amount = long_capital / price1
        short_amount = short_capital / price2
        
        # Subtract the allocated capital and the fee from the total capital
        self.capital -= (capital_per_pair + (capital_per_pair * self.fee))
        
        # Store the position details
        self.positions[pair] = {
            'long': {
                'symbol': symbol1,
                'amount': long_amount,
                'entry_price': price1,
                'entry_capital': long_capital
            },
            'short': {
                'symbol': symbol2,
                'amount': short_amount,
                'entry_price': price2,
                'entry_capital': short_capital,
                'timestamp': datetime.datetime.now()
            }
        }
        
        print(f"Opened long position for {symbol1} with capital {long_capital} and short position for {symbol2} with capital {short_capital}.")
    
    def open_position_backtest(self, symbol1, symbol2, timestamp, long_symbol, short_symbol):
        pair = (symbol1, symbol2)
        inverted_pair = (symbol2, symbol1)

        # Check if the position is already open
        if pair in self.positions or inverted_pair in self.positions:
            print(f"Position already open for pair {symbol1}-{symbol2}.")
            return

        num_active_pairs = len(self.active_pairs)

        if num_active_pairs == 0:
            print("No active pairs to allocate capital.")
            return

        # Allocate capital per pair dynamically
        capital_per_pair = self.capital / max(1, num_active_pairs)
        long_capital = capital_per_pair / 2
        short_capital = capital_per_pair / 2

        # Get entry prices
        price_long = self.historical_data[long_symbol].loc[timestamp]['close']
        price_short = self.historical_data[short_symbol].loc[timestamp]['close']

        # Compute position sizes
        long_amount = long_capital / price_long
        short_amount = short_capital / price_short

        # Deduct capital including the opening fee
        trade_fee = capital_per_pair * self.fee
        self.capital -= capital_per_pair + trade_fee

        # Store position details
        self.positions[pair] = {
            'long': {
                'symbol': long_symbol,
                'amount': long_amount,
                'entry_price': price_long,
                'entry_capital': long_capital
            },
            'short': {
                'symbol': short_symbol,
                'amount': short_amount,
                'entry_price': price_short,
                'entry_capital': short_capital
            }
        }

        print(f'Opened LONG in {long_symbol}, SHORT in {short_symbol}')

    def close_position(self, symbol1, symbol2):
        '''
        Function to close a position in a given asset returning the profit/loss to the total capital.
        '''
        pair = (symbol1, symbol2)
        if pair not in self.positions:
            print(f"No open position for pair {symbol1}-{symbol2}.")
            return
        
        # Fetch the latest prices for both symbols
        price1 = float(self.fetch_last_price(symbol1)['price'])  # Current price for the long position
        price2 = float(self.fetch_last_price(symbol2)['price'])  # Current price for the short position
        
        # Retrieve the position details
        position = self.positions[pair]

        # Calculate the total amount received from closing the long position
        long_total_received = price1 * position['long']['amount']
        # Calculate the total amount received from closing the short position
        short_total_received = price2 * position['short']['amount']
        
        # Calculate the profit/loss for the long position
        long_profit_loss = long_total_received - position['long']['entry_capital']
        
        # Calculate the profit/loss for the short position
        short_profit_loss = position['short']['entry_capital'] - short_total_received
        
        # Calculate the total profit/loss
        total_profit_loss = long_profit_loss + short_profit_loss

        # Deduct the fee from the total received for each position
        long_total_after_fee = long_total_received - (long_total_received * self.fee)
        short_total_after_fee = short_total_received - (short_total_received * self.fee)
        
        # Update capital with the total received after fee deduction for both positions
        self.capital += long_total_after_fee + short_total_after_fee
        
        # Remove the position from the positions dictionary
        del self.positions[pair]
        
        print(f"Closed positions for pair {symbol1}-{symbol2}. Total P&L: {total_profit_loss}.")

    def close_position_backtest(self, symbol1, symbol2, latest_price_symbol1, latest_price_symbol2):
        pair = (symbol1, symbol2)
        inverted_pair = (symbol2, symbol1)

        # Ensure we only access an existing position
        if pair in self.positions:
            position = self.positions[pair]
            price_long = latest_price_symbol1
            price_short = latest_price_symbol2
        elif inverted_pair in self.positions:
            position = self.positions[inverted_pair]
            price_long = latest_price_symbol2
            price_short = latest_price_symbol1
        else:
            print(f"No position found for {symbol1}-{symbol2}")
            return

        # Compute the final value of the long and short positions
        long_total_received = price_long * position['long']['amount']
        short_total_received = price_short * position['short']['amount']

        # Calculate profit/loss
        long_profit_loss = long_total_received - position['long']['entry_capital']
        short_profit_loss = position['short']['entry_capital'] - short_total_received  # Short sells high, buys back low

        total_profit_loss = long_profit_loss + short_profit_loss

        # Deduct exit fee based on initial capital, not just P&L
        trade_fee = (position['long']['entry_capital'] + position['short']['entry_capital']) * self.fee
        self.capital += total_profit_loss - trade_fee

        # Restore initial capital that was used in the position
        self.capital += position['long']['entry_capital'] + position['short']['entry_capital']

        # Remove position from tracking
        del self.positions[pair]

        print(f"Closed positions for pair {symbol1}-{symbol2}. Total P&L: {total_profit_loss:.2f}.")

    def check_capital(self):
        total_position_value = 0
        
        for position in self.positions.values():
            symbol1 = position['long']['symbol']
            symbol2 = position['short']['symbol']
            
            # Fetch the latest prices for both symbols
            price1 = float(self.fetch_last_price(symbol1)['price'])
            price2 = float(self.fetch_last_price(symbol2)['price'])
            
            # Calculate the current value of the long position
            long_value = price1 * position['long']['amount']
            # Calculate the current value of the short position
            short_value = price2 * position['short']['amount']
            
            # Add the values to the total position value
            total_position_value += long_value + short_value
        
        self.total_capital = self.capital + total_position_value
        # print(f'Total capital including positions: {self.total_capital}')
        # print(f'Available capital: {self.capital}')
        # print(f'Total value of open positions: {total_position_value}')

    def calculate_beta(self, symbol1, symbol2):
        """
        Estimate beta coefficient using OLS regression for two symbols.
        """
        # Convert to NumPy arrays and ensure the values are floats
        y = self.rolling_data[symbol1]['close'].to_numpy().astype(float)
        X = self.rolling_data[symbol2]['close'].to_numpy().astype(float)
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Perform OLS regression
        model = sm.OLS(y, X).fit()
        
        # Get the beta (slope) coefficient
        beta = model.params[1]
        
        return beta

    def calculate_spread(self, symbol1, symbol2):
        """
        Calculate the return-based spread between two symbols.
        Assumes rolling_data has synchronized timestamps.
        """
        if symbol1 not in self.rolling_data or symbol2 not in self.rolling_data:
            return None

        # Convert 'close' prices from string to float before applying pct_change
        prices1 = self.rolling_data[symbol1]['close'].astype(float)
        prices2 = self.rolling_data[symbol2]['close'].astype(float)

        # Compute returns
        returns1 = prices1.pct_change()
        returns2 = prices2.pct_change()

        # Drop the first NaN value
        returns1 = returns1.dropna()
        returns2 = returns2.dropna()

        # Convert to NumPy array (optional, but useful for further calculations)
        returns1 = returns1.to_numpy()
        returns2 = returns2.to_numpy()

        Y = returns1
        X = returns2

        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()

        beta = model.params[1]
        spread = Y - beta * X[:, 1]  # X[:, 1] extracts the return values, ignoring the intercept

        return spread

    def backtest(self):
        print("Running backtest")
        cryptocurrencies = ['AVAUSDT', 'BTCUSDT']
        data_dir = 'historical_data'  # Directory where CSV files will be saved
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Load historical data for each symbol
        end_time = int(time.time() * 1000)
        for symbol in cryptocurrencies:
            file_path = os.path.join(data_dir, f"{symbol}_historical_data.csv")
            
            if os.path.exists(file_path):
                print(f"Loading historical data for {symbol} from file")
                self.historical_data[symbol] = pd.read_csv(file_path, index_col=0, parse_dates=True)
            else:
                print(f"Fetching and saving historical data for {symbol}")
                self.historical_data[symbol] = self.fetch_historical_prices_backtest(symbol, end_time)
                self.historical_data[symbol].index = pd.to_datetime(self.historical_data[symbol].index)
                self.historical_data[symbol].to_csv(file_path) 

        if self.historical_data[symbol].isnull().values.any():
            self.historical_data[symbol].ffill(inplace=True)

        timestamps = self.historical_data['BTCUSDT'].index

        for i, timestamp in enumerate(timestamps[43198:], start=43199):
            sys.stdout.write(f'\rProcessing {i}/{len(timestamps)}: {timestamp}')
            sys.stdout.flush()
            for symbol in cryptocurrencies:
                self.rolling_window_backtest(symbol, timestamp)

            for i in range(len(cryptocurrencies)):
                for j in range(i + 1, len(cryptocurrencies)):
                    symbol1 = cryptocurrencies[i]
                    symbol2 = cryptocurrencies[j]
                    self.check_active_pairs(symbol1, symbol2)
            self.remove_expired_pairs()
            
            for symbol1, symbol2 in self.active_pairs:
                rolling_spread = self.calculate_spread(symbol1, symbol2)
                latest_price1 = self.historical_data[symbol1].loc[timestamp, "close"]
                latest_price2 = self.historical_data[symbol2].loc[timestamp, "close"]

                prev_timestamp = self.historical_data[symbol1].index[self.historical_data[symbol1].index.get_loc(timestamp) - 1]
                prev_price1 = self.historical_data[symbol1].loc[prev_timestamp, "close"]
                prev_price2 = self.historical_data[symbol2].loc[prev_timestamp, "close"]

                latest_price1 = float(latest_price1)
                prev_price1 = float(prev_price1)

                return1 = (latest_price1 / prev_price1) - 1
                return2 = (latest_price2 / prev_price2) - 1

                spread = return1 - return2

                mean = rolling_spread.mean()
                std = rolling_spread.std()
                upper_open = mean + 2 * std
                lower_open = mean - 2 * std
                upper_close = mean + 0.5 * std
                lower_close = mean - 0.5 * std
                
                if spread > upper_open:
                    print('Spread greater than upper open')
                    self.open_position_backtest(symbol1, symbol2, timestamp, symbol1, symbol2)
                if spread < lower_open:
                    print('Spread less than lower open')
                    self.open_position_backtest(symbol1, symbol2, timestamp, symbol2, symbol1)
                if lower_close < spread < upper_close and (symbol1, symbol2) in self.positions:
                    self.close_position_backtest(symbol1, symbol2, latest_price1, latest_price2)
                if lower_close < spread < upper_close and (symbol2, symbol1) in self.positions:
                    self.close_position_backtest(symbol2, symbol1, latest_price2, latest_price1)
                


    def main(self):
        '''
        Main function to contain the logic binding functions together, fetching all data, running tests, and executing changes in positions.
        '''
        print("Running main function")
        cryptocurrencies = ['AVAUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT', 'UNIUSDT', 'AAVEUSDT', 'XRPUSDT', 
             'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'XLMUSDT', 'TONUSDT', 'SUIUSDT', 'DOTUSDT']
        
        while True:
            # Update rolling windows
            for symbol in cryptocurrencies:
                self.rolling_window(symbol)

            # Check for cointegration and manage pairs
            for i in range(len(cryptocurrencies)):
                for j in range(i + 1, len(cryptocurrencies)):
                    symbol1 = cryptocurrencies[i]
                    symbol2 = cryptocurrencies[j]
                    self.check_active_pairs(symbol1, symbol2)
            self.remove_expired_pairs()

            # Check spread conditions and open positions
            for symbol1, symbol2 in self.active_pairs:
                spread = self.calculate_spread(symbol1, symbol2)
                
                if spread is None:
                    continue  # Skip if data is missing
                
                mean = spread.mean()
                std = spread.std()
                upper_open = mean + 2 * std
                lower_open = mean - 2 * std
                upper_close = mean + 0.5 * std
                lower_close = mean - 0.5 * std
                pair = (symbol1, symbol2)

                if spread[-1] > upper_open:  # Short first, long second
                    if pair not in self.positions:
                        self.open_position(symbol2, symbol1)  # Corrected order
                        print(f"Opened SHORT on {symbol1} and LONG on {symbol2} (Spread: {spread[-1]:.6f})")
                    
                elif spread[-1] < lower_open:  # Long first, short second
                    if pair not in self.positions:
                        self.open_position(symbol1, symbol2)  # Corrected order
                        print(f"Opened LONG on {symbol1} and SHORT on {symbol2} (Spread: {spread[-1]:.6f})")
                if lower_close < spread[-1] < upper_close and pair in self.positions:
                    self.close_position(symbol1, symbol2)
                    print(f'Closed position on {symbol1} and {symbol2}')
            self.check_capital()

if __name__ == '__main__':
    pair_trading = PairTrading(pairs=[])
    # pair_trading.main()
    pair_trading.backtest()