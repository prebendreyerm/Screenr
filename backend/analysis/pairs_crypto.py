import os, sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from statsmodels.tsa.stattools import coint
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import time


load_dotenv()
api_key = os.getenv('API_KEY')


# # Historical daily
# def get_crypto_data(crypto):
#     url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{crypto}?apikey={api_key}'
#     response = requests.get(url)
#     data = response.json()
#     historical_data = data['historical']
#     df = pd.DataFrame(historical_data)
#     return df

# Intraday
def get_crypto_data(crypto):
    start = '2024-12-01'
    end = '2025-01-31'
    url = f'https://financialmodelingprep.com/api/v3/historical-chart/4hour/{crypto}?from={start}&to={end}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()  # The response is already a list, not a dictionary
    # Convert directly to DataFrame
    df = pd.DataFrame(data)
    return df

def match_dates(df1, df2):
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    df1.set_index('date', inplace=True)
    df2.set_index('date', inplace=True)
    df1, df2 = df1.align(df2, join='inner', axis=0)
    return df1, df2

def scoring(crypto1, crypto2):
    data1 = get_crypto_data(crypto1)
    data2 = get_crypto_data(crypto2)
    data1, data2 = match_dates(data1, data2)
    score, pvalue, _ = coint(data1['close'], data2['close'])
    if pvalue < 0.05:
        results.append((crypto1, crypto2, pvalue, score))
    return results

def check_for_stationarity(X, cutoff=0.01):
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print(f'p-value = {pvalue}. The series is likely stationary.')
        return True, pvalue
    else:
        print(f'p-value = {pvalue}. The series is likely non-stationary.')
        return False, pvalue

start_date = '2024-12-01'
end_date = '2025-01-31'



def cointegration_test(crypto1, crypto2):
    try:
        X1 = get_crypto_data(crypto1)
        X2 = get_crypto_data(crypto2)

        # Restrict to the given date range
        X1 = X1[(X1['date'] >= start_date) & (X1['date'] <= end_date)]
        X2 = X2[(X2['date'] >= start_date) & (X2['date'] <= end_date)]

        # Match dates
        X1, X2 = match_dates(X1, X2)

        # Convert to time series format
        X1 = X1['close'][::-1].values
        X2 = X2['close'][::-1].values

        # Run OLS regression
        X1 = sm.add_constant(X1)
        results = sm.OLS(X2, X1).fit()
        beta = results.params[1]
        z = X2 - beta * X1[:, 1]  # Residuals

        # Check for stationarity
        boolean, pvalue = check_for_stationarity(z)
        return boolean, pvalue, z

    except Exception as e:
        print(f"Error in cointegration test for {crypto1} and {crypto2}: {e}")
        return False, None



pairs = pd.read_csv('crypto_pairs_21_22.csv')
pairs = pairs.sort_values(by='pvalue')







def apply_slippage(price, direction, slippage=0.003):
    """Apply slippage to the trade prices"""
    if direction == "buy":
        return price * (1 + slippage)  # Price goes up for buy
    elif direction == "sell":
        return price * (1 - slippage)  # Price goes down for sell
    return price

def apply_fee(price, fee_rate=0.002):
    """Apply transaction fee to the price"""
    return price * (1 - fee_rate)

def backtest(coin1, coin2):
    coin1_prices = get_crypto_data(coin1)
    coin2_prices = get_crypto_data(coin2)
    coin1_prices = coin1_prices[(coin1_prices['date'] >= start_date) & (coin1_prices['date'] <= end_date)]
    coin2_prices = coin2_prices[(coin2_prices['date'] >= start_date) & (coin2_prices['date'] <= end_date)]
    
    boolean, pvalue, spread = cointegration_test(coin1, coin2)
    normalized_spread = (spread - np.mean(spread)) / np.std(spread)

    p1 = coin1_prices['close'][::-1].values
    p2 = coin2_prices['close'][::-1].values

    entry_threshold = 2
    exit_threshold = 0.5
    capital = 1000
    position = 0  # 1 = Short Coin1, Long Coin2, -1 = Long Coin1, Short Coin2, 0 = No position
    long_asset = short_asset = 0  # Holding amounts

    # Store trade markers for plotting
    entry_points = []
    exit_points = []
    entry_values = []
    exit_values = []

    if boolean == True:
        # Start of backtesting loop
        for i in range(len(normalized_spread) - 1):
            if position == 0 and normalized_spread[i] > entry_threshold:
                # Spread is too high → Long Coin1, Short Coin2
                position = 1
                entry_coin1_price = apply_slippage(p1[i], "buy")  # Apply slippage for Coin1
                entry_coin2_price = apply_slippage(p2[i], "sell")  # Apply slippage for Coin2

                # Apply transaction fees to entry prices
                entry_coin1_price = apply_fee(entry_coin1_price)  
                entry_coin2_price = apply_fee(entry_coin2_price)

                # Use half of the available capital for each side
                half_capital = capital / 2
                long_asset = half_capital / entry_coin1_price  # Buy Coin1
                short_asset = half_capital / entry_coin2_price  # Borrow & sell Coin2
                capital = half_capital

                print(f"Long Coin1 at {entry_coin1_price}, Short Coin2 at {entry_coin2_price}")
                entry_points.append(i)
                entry_values.append(normalized_spread[i])

            elif position == 0 and normalized_spread[i] < -entry_threshold:
                # Spread is too low → Short Coin1, Long Coin2
                position = -1
                entry_coin1_price = apply_slippage(p1[i], "sell")  # Apply slippage for Coin1
                entry_coin2_price = apply_slippage(p2[i], "buy")  # Apply slippage for Coin2

                # Apply transaction fees to entry prices
                entry_coin1_price = apply_fee(entry_coin1_price)  
                entry_coin2_price = apply_fee(entry_coin2_price)

                # Use half of the available capital for each side
                half_capital = capital / 2
                short_asset = half_capital / entry_coin1_price  # Borrow & sell Coin1
                long_asset = half_capital / entry_coin2_price   # Buy Coin2
                capital = half_capital

                print(f"Short Coin1 at {entry_coin1_price}, Long Coin2 at {entry_coin2_price}")
                entry_points.append(i)
                entry_values.append(normalized_spread[i])

            elif position == 1 and -exit_threshold <= normalized_spread[i] <= exit_threshold:
                # Closing a (1) trade: Long Coin1, Short Coin2
                exit_coin1_price = apply_slippage(p1[i], "sell")  # Apply slippage for Coin1 exit
                exit_coin2_price = apply_slippage(p2[i], "buy")   # Apply slippage for Coin2 exit

                # Apply transaction fees to exit prices
                exit_coin1_price = apply_fee(exit_coin1_price)  
                exit_coin2_price = apply_fee(exit_coin2_price)

                # Calculate P&L with slippage and fees considered
                short_profit_loss = short_asset * (entry_coin2_price - exit_coin2_price)  # Short profit/loss
                long_value = long_asset * (exit_coin1_price)  # Long value

                # Update capital with profit/loss
                capital += short_profit_loss + long_value  # Update capital with profit/loss

                # Reset positions
                short_asset = 0
                long_asset = 0
                position = 0

                print(f"Exited long Coin1 at {exit_coin1_price}, short Coin2 at {exit_coin2_price}, Capital: {capital:.2f}")
                exit_points.append(i)
                exit_values.append(normalized_spread[i])

            elif position == -1 and -exit_threshold <= normalized_spread[i] <= exit_threshold:
                # Closing a (-1) trade: Short Coin1, Long Coin2
                exit_coin1_price = apply_slippage(p1[i], "buy")  # Apply slippage for Coin1 exit
                exit_coin2_price = apply_slippage(p2[i], "sell")   # Apply slippage for Coin2 exit

                # Apply transaction fees to exit prices
                exit_coin1_price = apply_fee(exit_coin1_price)  
                exit_coin2_price = apply_fee(exit_coin2_price)

                # Calculate P&L with slippage and fees considered
                short_profit_loss = short_asset * (entry_coin1_price - exit_coin1_price)  # Short profit/loss
                long_value = long_asset * (exit_coin2_price)  # Long value

                # Update capital with profit/loss
                capital += short_profit_loss + long_value  # Update capital with profit/loss

                # Reset positions
                short_asset = 0
                long_asset = 0
                position = 0

                print(f"Exited short Coin1 at {exit_coin1_price}, long Coin2 at {exit_coin2_price}, Capital: {capital:.2f}")
                exit_points.append(i)
                exit_values.append(normalized_spread[i])

            # Ensure capital never goes negative
            if capital < 0:
                print("Capital is less than zero. Nullifying positions and stopping trade.")
                position = 0
                short_asset = 0
                long_asset = 0
                capital = 0  # Reset capital to zero to avoid negative values
                break # Exit the loop entirely to prevent further trades with zero capital
    else:
        pass



    # # Plotting
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # # --- Top plot: Normalized Spread with Trade Markers ---
    # axes[0].plot(normalized_spread, label="Normalized Spread", color='blue')
    # axes[0].axhline(y=entry_threshold, color='green', linestyle='--', label="Entry Threshold (+2)")
    # axes[0].axhline(y=-entry_threshold, color='red', linestyle='--', label="Entry Threshold (-2)")
    # axes[0].axhline(y=exit_threshold, color='purple', linestyle='--', label="Exit Threshold (+1)")
    # axes[0].axhline(y=-exit_threshold, color='orange', linestyle='--', label="Exit Threshold (-1)")

    # # Trade markers
    # axes[0].scatter(entry_points, entry_values, color='green', marker='^', label="Entry", zorder=3)
    # axes[0].scatter(exit_points, exit_values, color='red', marker='v', label="Exit", zorder=3)

    # axes[0].legend()
    # axes[0].set_title("Pairs Trading Strategy: Normalized Spread with Entry/Exit Points")
    # axes[0].set_ylabel("Normalized Spread")

    # # --- Bottom plot: Coin Prices with Dual Y-Axes ---
    # ax1 = axes[1]  # Primary y-axis (Coin1)
    # ax2 = ax1.twinx()  # Secondary y-axis (Coin2)

    # ax1.plot(p1, label="Coin1 Price", color='blue')
    # ax2.plot(p2, label="Coin2 Price", color='orange')

    # # Labels for each y-axis
    # ax1.set_ylabel("Coin1 Price", color='blue')
    # ax2.set_ylabel("Coin2 Price", color='orange')

    # ax1.tick_params(axis='y', colors='blue')
    # ax2.tick_params(axis='y', colors='orange')

    # # Title & legend
    # ax1.set_title("Coin Prices Over Time")
    # ax1.set_xlabel("Time")

    # Ensures proper layout
    # plt.tight_layout()
    # plt.show()
    return




if __name__ == "__main__":
    for index, row in pairs.iterrows():
        print(row["crypto1"], row["crypto2"])
        backtest(row['crypto1'], row['crypto2'])
