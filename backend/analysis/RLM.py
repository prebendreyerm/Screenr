import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Function to remove outliers based on IQR, ensuring only numerical columns are included
def remove_outliers(df, columns):
    numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Step 1: Load data from the SQLite database
conn = sqlite3.connect(r'backend\data\financial_data.db')
df_Ratios = pd.read_sql_query("SELECT * FROM RatiosQuarter", conn)
df_prices = pd.read_sql_query("SELECT * FROM HistoricalPricesQuarter", conn)
conn.close()  # Close the connection when done

# Merge the Ratios and Prices DataFrames
final_df = pd.merge(df_Ratios, df_prices[['symbol', 'date', 'stockPrice']], on=['symbol', 'date'], how='inner')

# Step 3: Calculate the change in stock price for each symbol
final_df['price_change'] = final_df.groupby('symbol')['stockPrice'].diff()

# Step 4: Create the binary target variable (1 for price increase, 0 for decrease)
final_df['price_increase'] = (final_df['price_change'] > 0).astype(int)

# Drop rows with NaN values that result from the price change calculation
final_df = final_df.dropna(subset=['price_increase'])

# Step 2: Clean the data (drop NaNs and filter out irrelevant columns)
on_columns = ['symbol', 'date', 'period', 'calendarYear']  # Specify the columns to drop
final_df = final_df.drop(columns=on_columns)  # Drop symbol, date, period, and calendarYear for features

# Apply the outlier removal function on the filtered DataFrame
final_df = remove_outliers(final_df, final_df.columns)
# Print final_df to inspect the data

# Step 5: Define the custom Gym environment for stock trading
class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.net_worth_history = []  # To track net worth over time
        self.previous_net_worth = initial_balance  # Initialize previous net worth
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.data.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.previous_net_worth = self.balance  # Reset previous net worth
        self.net_worth_history = []  # Reset the net worth history
        return self._get_observation()

    def _get_observation(self):
        # Observation now only includes the features (no need to drop symbol/date anymore)
        obs = self.data.iloc[self.current_step].values
        return obs

    def step(self, action):
        # Get the price change and stock price
        price = self.data.iloc[self.current_step]['stockPrice']
        price_change = self.data.iloc[self.current_step]['price_change']
        reward = 0

        # Record the previous net worth before the action
        previous_net_worth = self.net_worth

        if action == 1:  # Buy
            self.shares_held += 1
            self.balance -= price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += price
                reward = price_change  # Reward is the price change when selling

        # Update the net worth
        self.net_worth = self.balance + (self.shares_held * price)
        self.net_worth_history.append(self.net_worth)  # Track the net worth

        # Reward based on the change in net worth
        reward = self.net_worth - previous_net_worth  # Reward is the change in net worth

        self.current_step += 1

        # End the episode if we're at the end of the data
        done = self.current_step >= len(self.data) - 1
        obs = self._get_observation()
        return obs, reward, done, {}



# Step 6: Initialize the environment and RL model
env = StockTradingEnv(final_df)

# Initialize PPO (Proximal Policy Optimization) model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model for a specified number of timesteps
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("stock_trading_model")

# Step 7: Test the trained model (for demonstration purposes)
obs = env.reset()
for _ in range(100):  # Test for 100 timesteps
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

# Plot the net worth over time
plt.plot(env.net_worth_history)
plt.title('Net Worth Over Time')
plt.xlabel('Timesteps')
plt.ylabel('Net Worth')
plt.show()
