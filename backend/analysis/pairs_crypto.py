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
    def __init__(self, pairs, initial_capital=1000, fee=0.002, weeks=52):
        '''
        Initializing function of the class
        '''
        self.pairs = pairs
        self.initial_capital = initial_capital
        self.fee = fee
        self.weeks = weeks
        self.capital = initial_capital  # Total capital
        self.now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        self.rolling_data = {}
        self.positions = {}
        self.active_pairs = {}
        self.capital_per_pair = {}
    
    def fetch_last_price(self, symbol):
        '''
        Function to fetch the most recent price of a given asset.
        '''
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            response = response.json()
            return response
        except requests.exceptions.RequestException as e:
            print(f'Error fetching last price data for {symbol}: {e}')
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
        start_time = end_time - 7 * 24 * 60 * 60 * 1000  # 7 days in milliseconds
        url = f"https://api.binance.com/api/v3/klines"
        limit = 1000  # Binance API limit for klines
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
        
    def rolling_window(self, symbol):
        '''
        Function to update the rolling window with the latest price.
        '''
        if symbol not in self.rolling_data:
            self.rolling_data[symbol] = self.fetch_historical_prices(symbol)
        
        next_price_data = self.fetch_last_price(symbol)
        if next_price_data:
            next_price = float(next_price_data['price'])
            current_time = pd.to_datetime(time.time(), unit='s')
            
            # Append the new price to the DataFrame
            new_row = pd.DataFrame({'close': [next_price]}, index=[current_time])
            self.rolling_data[symbol] = pd.concat([self.rolling_data[symbol], new_row])
            
            # Remove the oldest row to maintain the size
            if len(self.rolling_data[symbol]) > 7 * 24 * 60:  # 7 days * 24 hours * 60 minutes
                self.rolling_data[symbol] = self.rolling_data[symbol].iloc[1:]
            
            # Print the last few rows for debugging
            # print(symbol, self.rolling_data[symbol].tail())
                      
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
                print(f"Pair {symbol1}-{symbol2} is already active. Updating timestamp.")
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
        
        # Subtract the allocated capital from the total capital
        self.capital -= capital_per_pair
        
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
                'entry_capital': short_capital
            }
        }
        
        print(f"Opened long position for {symbol1} with capital {long_capital} and short position for {symbol2} with capital {short_capital}.")
    
    def close_position(self, symbol1, symbol2):
        '''
        Function to close a position in a given asset returning the profit/loss to the total capital.
        '''
        pair = (symbol1, symbol2)
        if pair not in self.positions:
            print(f"No open position for pair {symbol1}-{symbol2}.")
            return
        
        # Fetch the latest prices for both symbols
        price1 = float(self.fetch_last_price(symbol1)['price'])
        price2 = float(self.fetch_last_price(symbol2)['price'])
        
        # Retrieve the position details
        position = self.positions[pair]

        # Calculate the profit/loss for the long position
        long_profit_loss = (price1 - position['long']['entry_price']) * position['long']['amount']
        
        # Calculate the profit/loss for the short position
        short_profit_loss = (position['short']['entry_price'] - price2) * position['short']['amount']
        
        # Calculate the total profit/loss
        total_profit_loss = long_profit_loss + short_profit_loss
        
        # Add the total profit/loss back to the capital
        self.capital += position['long']['entry_capital'] + position['short']['entry_capital'] + total_profit_loss
        
        # Remove the position from the positions dictionary
        del self.positions[pair]
        
        print(f"Closed positions for pair {symbol1}-{symbol2}. Total P&L: {total_profit_loss}.")
    
    def check_capital(self):
        print(f'Total capital:{self.capital}')

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
        Calculate the spread between two symbols using the hedge ratio β.
        Assumes rolling_data has synchronized timestamps.
        """
        if symbol1 not in self.rolling_data or symbol2 not in self.rolling_data:
            return None
        
        # Get price series
        prices1 = self.rolling_data[symbol1]['close'].to_numpy().astype(float)
        prices2 = self.rolling_data[symbol2]['close'].to_numpy().astype(float)

        # Compute hedge ratio (β)
        beta = self.calculate_beta(symbol1, symbol2)

        # Calculate spread using hedge ratio
        spread = prices1 - beta * prices2
        return spread



    def main(self):
        '''
        Main function to contain the logic binding functions together, fetching all data, running tests, and executing changes in positions.
        '''
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
                    self.open_position(symbol1, symbol2)
                    print(f"Opened SHORT on {symbol1} and LONG on {symbol2} (Spread: {spread[-1]:.6f})")
                    
                elif spread[-1] < lower_open:  # Long first, short second
                    self.open_position(symbol1, symbol2)
                    print(f"Opened LONG on {symbol1} and SHORT on {symbol2} (Spread: {spread[-1]:.6f})")
                if lower_close < spread[-1] < upper_close and pair in self.positions:
                    self.close_position(symbol1, symbol2)
                    print(f'Closed position on {symbol1} and {symbol2}')
            self.check_capital()

if __name__ == '__main__':
    pair_trading = PairTrading(pairs=[])
    pair_trading.main()