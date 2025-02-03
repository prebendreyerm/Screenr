import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

# Set the seed for reproducibility
np.random.seed(42)

# Generate two random walks (stationary)
n_steps = 1000

# Random walk with mean zero and unit variance (stationary)
series_1 = np.cumsum(np.abs(np.random.normal(0, 1, n_steps)))
series_2 = 0.5 * series_1 + np.abs(np.random.normal(0, 0.5, n_steps))

def check_for_stationarity(X, cutoff=0.01):
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print(f'p-value = {pvalue}. The series is likely stationary.')
        return True, pvalue
    else:
        print(f'p-value = {pvalue}. The series is likely non-stationary.')
        return False, pvalue

def cointegration_test(series1, series2):
    try:
        X1 = series1
        X2 = series2

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

boolean, pvalue, spread = cointegration_test(series_1, series_2)

normalized_spread = (spread - np.mean(spread))/np.std(spread)

def backtest(p1, p2):
    # With spread over +2, X1 is overpriced relative to X2 and thus X1 should be short, while X2 should be long

    # With spread under -2, X1 is underprice relative to X2 and thus X1 long, X2 short
    entry_threshold = 2
    exit_threshold = 0.5
    capital = 10
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
                entry_coin1_price = p1[i]  # Store entry price for Coin1
                entry_coin2_price = p2[i]  # Store entry price for Coin2

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
                entry_coin1_price = p1[i]  # Store entry price for Coin1
                entry_coin2_price = p2[i]  # Store entry price for Coin2

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
                short_profit_loss = short_asset * (entry_coin2_price - p2[i])  # Short profit/loss
                long_value = long_asset * p1[i]  # Long value

                # Check if profit/loss is more negative than capital, and nullify position if necessary
                if short_profit_loss + long_value < 0 and capital + short_profit_loss + long_value < 0:
                    print(f"Position defaults: Insufficient capital to close position at step {i}. Capital remains unchanged.")
                    # Nullify the position and reset capital, short and long assets to 0
                    capital = capital  # Capital remains unchanged
                    position = 0
                    short_asset = 0
                    long_asset = 0
                else:
                    capital += short_profit_loss + long_value  # Update capital with profit/loss

                # Reset positions
                short_asset = 0
                long_asset = 0
                position = 0

                print(f"Exited long Coin1 at {p1[i]}, short Coin2 at {p2[i]}, Capital: {capital:.2f}")
                exit_points.append(i)
                exit_values.append(normalized_spread[i])

            elif position == -1 and -exit_threshold <= normalized_spread[i] <= exit_threshold:
                # Closing a (-1) trade: Short Coin1, Long Coin2
                short_profit_loss = short_asset * (entry_coin1_price - p1[i])  # Short profit/loss
                long_value = long_asset * p2[i]  # Long value

                # Check if profit/loss is more negative than capital, and nullify position if necessary
                if short_profit_loss + long_value < 0 and capital + short_profit_loss + long_value < 0:
                    print(f"Position defaults: Insufficient capital to close position at step {i}. Capital remains unchanged.")
                    # Nullify the position and reset capital, short and long assets to 0
                    capital = capital  # Capital remains unchanged
                    position = 0
                    short_asset = 0
                    long_asset = 0
                else:
                    capital += short_profit_loss + long_value  # Update capital with profit/loss

                # Reset positions
                short_asset = 0
                long_asset = 0
                position = 0

                print(f"Exited short Coin1 at {p1[i]}, long Coin2 at {p2[i]}, Capital: {capital:.2f}")
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

#Removed unnecessary print statements and corrected capital update at entry.

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # --- Top plot: Normalized Spread with Trade Markers ---
    axes[0].plot(normalized_spread, label="Normalized Spread", color='blue')
    axes[0].axhline(y=entry_threshold, color='green', linestyle='--', label="Entry Threshold (+2)")
    axes[0].axhline(y=-entry_threshold, color='red', linestyle='--', label="Entry Threshold (-2)")
    axes[0].axhline(y=exit_threshold, color='purple', linestyle='--', label="Exit Threshold (+1)")
    axes[0].axhline(y=-exit_threshold, color='orange', linestyle='--', label="Exit Threshold (-1)")

    # Trade markers
    axes[0].scatter(entry_points, entry_values, color='green', marker='^', label="Entry", zorder=3)
    axes[0].scatter(exit_points, exit_values, color='red', marker='v', label="Exit", zorder=3)

    axes[0].legend()
    axes[0].set_title("Pairs Trading Strategy: Normalized Spread with Entry/Exit Points")
    axes[0].set_ylabel("Normalized Spread")

    # --- Bottom plot: Coin Prices with Dual Y-Axes ---
    ax1 = axes[1]  # Primary y-axis (Coin1)
    ax2 = ax1.twinx()  # Secondary y-axis (Coin2)

    ax1.plot(p1, label="Coin1 Price", color='blue')
    ax2.plot(p2, label="Coin2 Price", color='orange')

    # Labels for each y-axis
    ax1.set_ylabel("Coin1 Price", color='blue')
    ax2.set_ylabel("Coin2 Price", color='orange')

    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='orange')

    # Title & legend
    ax1.set_title("Coin Prices Over Time")
    ax1.set_xlabel("Time")

    # Ensures proper layout
    plt.tight_layout()
    plt.show()
    return

if __name__ == "__main__":
    backtest(series_1,series_2)