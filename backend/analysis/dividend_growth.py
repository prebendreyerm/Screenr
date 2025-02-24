import numpy as np
"""
This script selects the best stocks according to EV/EBITDA and ensures consistent dividend growth.

Steps:
1. Connect to the SQLite database and load data from various tables.
2. Merge and preprocess the data to create a unified DataFrame.
3. Filter the data for a user-selected year and find stocks with consistent dividend growth over the last 10 years.
4. Compute the average dividend growth for each stock.
5. Rank the stocks based on their average dividend growth.
6. Calculate the projected dividend in 8 years relative to the current stock price.

Note:
- The 'dividendYield' is a fraction representing the current dividend yield.
- The 'avgDividendGrowth' is in dollars.
"""
import pandas as pd
import sqlite3
from pathlib import Path
import streamlit as st
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# Strategy: Select best stocks according to EV/EBITDA and ensure consistent dividend growth.

# Connect to the SQLite database
db_path = Path('backend/data/financial_data.db')
conn = sqlite3.connect(db_path)

# Load the data
df = pd.read_sql_query('SELECT * FROM FinancialGrowthAnnual', conn)
df2 = pd.read_sql_query('SELECT * FROM KeyMetricsAnnual', conn)
df3 = pd.read_sql_query('SELECT * FROM HistoricalPricesAnnual', conn)
df4 = pd.read_sql_query('SELECT * FROM Assets', conn)
df5 = pd.read_sql_query('SELECT * FROM KeyMetricsTTM', conn)

# Merge and preprocess data
merged_df = pd.merge(
    df2[['symbol', 'date', 'calendarYear', 'enterpriseValue', 'evToFreeCashFlow', 'payoutRatio', 'dividendYield']], 
    df3[['symbol', 'date', 'stockPrice']], 
    on=['symbol', 'date'], how='inner'
)

merged_df = pd.merge(
    merged_df, 
    df4[['symbol', 'sector', 'name']], 
    on=['symbol'], how='inner'
)

merged_df = pd.merge(
    merged_df, 
    df[['symbol', 'calendarYear', 'dividendsperShareGrowth', 'revenueGrowth']], 
    on=['symbol', 'calendarYear'], how='inner'
)

merged_df = pd.merge(
    merged_df, 
    df5[['symbol', 'enterpriseValueOverEBITDATTM']], 
    on=['symbol'], how='inner'
)

# Preprocessing: Remove unwanted rows and duplicates
merged_df = merged_df.dropna()
# Preprocessing: Remove unwanted rows and duplicates
merged_df = merged_df.dropna()
merged_df = merged_df.drop_duplicates(subset=['symbol', 'date', 'calendarYear'])


# Convert calendarYear to integer for filtering
merged_df['calendarYear'] = merged_df['calendarYear'].astype(int)

# User selects a year, e.g., 2022
selected_year = 2023


# Filter for the selected year and find stocks with consistent dividend growth
filtered_df = merged_df[merged_df['calendarYear'] <= selected_year] 


# Function to filter and compute average dividend growth
def filter_and_compute_avg_growth(group):
    group = group.sort_values(by='calendarYear')
    recent_years = group[group['calendarYear'] <= selected_year].tail(10)  

    if len(recent_years) == 10 and all(recent_years['dividendsperShareGrowth'] > 0) and all(recent_years['revenueGrowth'] > 0):
        avg_growth = recent_years['dividendsperShareGrowth'].mean()
        recent_years = recent_years.copy()  # Avoid modifying a slice of the original DataFrame
        recent_years['avgDividendGrowth'] = avg_growth  # Add new column
        return recent_years
    return None

# Apply function to each group
filtered_stocks = filtered_df.groupby('symbol').apply(filter_and_compute_avg_growth)

# Remove groups that returned None
filtered_stocks = filtered_stocks.dropna(subset=['avgDividendGrowth'])

# Keep only data for the selected year
filtered_df = filtered_stocks[filtered_stocks['calendarYear'] == selected_year]
ranked_stocks = filtered_df.sort_values(by='avgDividendGrowth', ascending=False)
ranked_stocks['dividendNow'] = ranked_stocks['dividendYield'] * ranked_stocks['stockPrice']
ranked_stocks['dividendIn8Years'] = ranked_stocks['dividendNow'] + 8 * ranked_stocks['avgDividendGrowth']
ranked_stocks['dividendYieldIn8Years'] = ranked_stocks['dividendIn8Years'] / ranked_stocks['stockPrice']
ranked_stocks.sort_values(by='dividendYieldIn8Years', ascending=False, inplace=True)
# Reorder columns to display the 'name' column all the way to the left
columns = ['name'] + [col for col in ranked_stocks.columns if col != 'name']
ranked_stocks = ranked_stocks[columns]

ranked_stocks = ranked_stocks[
    # (ranked_stocks['enterpriseValue'] > 0) &
    # (ranked_stocks['evToFreeCashFlow'] > 0) &
    # (ranked_stocks['enterpriseValueOverEBITDATTM'] > 0) &
    (ranked_stocks['payoutRatio'] > 0) &
    (ranked_stocks['payoutRatio'] < 0.9) &
    (ranked_stocks['dividendYieldIn8Years'] > 0.05)
]

# Function to fetch historical prices for the last year
def fetch_historical_prices(symbol):
    api_key = os.getenv('API_KEY')  # Replace with your FinancialModelingPrep API key
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date.strftime("%Y-%m-%d")}&to={end_date.strftime("%Y-%m-%d")}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    if 'historical' in data:
        return pd.DataFrame(data['historical'])
    return pd.DataFrame()

def calculate_std_position(row):
    symbol = row['symbol']
    prices_df = fetch_historical_prices(symbol)

    if prices_df.empty or len(prices_df) < 30:  # Ensure sufficient data
        print(f"Skipping {symbol} due to insufficient data")
        return None

    # Ensure data is sorted chronologically
    prices_df = prices_df.sort_values(by='date', ascending=True)

    # Compute daily log returns
    prices_df['log_return'] = np.log(prices_df['close'] / prices_df['close'].shift(1))
    prices_df = prices_df.dropna(subset=['log_return'])  # Drop NaN returns

    if len(prices_df) < 30:
        print(f"Skipping {symbol} due to insufficient return data")
        return None

    mean_return = prices_df['log_return'].mean()
    std_dev = prices_df['log_return'].std()

    if std_dev == 0 or np.isnan(std_dev):
        return None  # Prevent division by zero

    latest_return = prices_df['log_return'].iloc[-1]  # Most recent log return
    std_position = (latest_return - mean_return) / std_dev  # Standardized return position

    return std_position  # Return the computed value

# Apply to DataFrame
ranked_stocks['stdPosition'] = ranked_stocks.apply(calculate_std_position, axis=1)
ranked_stocks = ranked_stocks.dropna(subset=['stdPosition'])  # Remove invalid rows

print(ranked_stocks.head(50))


# # Apply the function to each row in the ranked_stocks DataFrame
# ranked_stocks['stdPosition'] = ranked_stocks.apply(calculate_std_position, axis=1)

# # Drop rows where stdPosition could not be calculated
# ranked_stocks = ranked_stocks.dropna(subset=['stdPosition'])

