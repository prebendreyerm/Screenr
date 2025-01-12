import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

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
    df[['symbol', 'calendarYear', 'dividendsperShareGrowth']], 
    on=['symbol', 'calendarYear'], how='inner'
)

merged_df = pd.merge(
    merged_df, 
    df5[['symbol', 'enterpriseValueOverEBITDATTM']], 
    on=['symbol'], how='inner'
)

# Preprocessing: Remove unwanted rows
merged_df = merged_df.dropna()
merged_df = merged_df[
    (merged_df['enterpriseValue'] > 0) &
    (merged_df['evToFreeCashFlow'] > 0) &
    (merged_df['enterpriseValueOverEBITDATTM'] > 0) &
    (merged_df['payoutRatio'] > 0) &
    (merged_df['payoutRatio'] < 0.2) &
    (merged_df['dividendYield'] > 0)
]

# Convert calendarYear to integer for filtering
merged_df['calendarYear'] = merged_df['calendarYear'].astype(int)

# User selects a year, e.g., 2022
selected_year = 2022

# Filter for the selected year and find stocks with consistent dividend growth
filtered_df = merged_df[merged_df['calendarYear'] <= selected_year]

# Group by symbol and check for consistent dividend growth
def has_consistent_dividend_growth(group):
    group = group.sort_values(by='calendarYear')
    recent_years = group[group['calendarYear'] < selected_year].tail(5)
    return len(recent_years) == 5 and all(recent_years['dividendsperShareGrowth'] > 0)

consistent_dividend_stocks = filtered_df.groupby('symbol').filter(has_consistent_dividend_growth)

selected_year = 2022

consistent_dividend_stocks = consistent_dividend_stocks[consistent_dividend_stocks['calendarYear'] == selected_year]

# Filter for the lowest EV/EBITDA per sector
lowest_per_sector = consistent_dividend_stocks.loc[consistent_dividend_stocks.groupby('sector')['enterpriseValueOverEBITDATTM'].idxmin()]

# Select columns to display
columns_to_display = [
    'symbol', 'name', 'calendarYear', 'sector', 
    'enterpriseValue', 'evToFreeCashFlow', 'enterpriseValueOverEBITDATTM', 
    'stockPrice', 'dividendYield'
]

lowest_per_sector = lowest_per_sector[columns_to_display]

# Print the final DataFrame
print(lowest_per_sector)