import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

# Strategy: Select best stocks according to EV/EBITDA and ensure consistent dividend growth.

# Connect to the SQLite database
db_path = Path('backend/data/financial_data.db')
conn = sqlite3.connect(db_path)

# Load the data
financial_growth = pd.read_sql_query('SELECT * FROM FinancialGrowthAnnual', conn)
key_metrics = pd.read_sql_query('SELECT * FROM KeyMetricsAnnual', conn)
prices = pd.read_sql_query('SELECT * FROM HistoricalPricesAnnual', conn)
assets = pd.read_sql_query('SELECT * FROM Assets', conn)
key_metrics_ttm = pd.read_sql_query('SELECT * FROM KeyMetricsTTM', conn)

# Merge and preprocess data
merged_df = pd.merge(
    key_metrics[['symbol', 'date', 'calendarYear', 'enterpriseValue', 'evToFreeCashFlow', 'operatingCashFlowPerShare', 'freeCashFlowYield',  'enterpriseValueOverEBITDA', 'evToOperatingCashFlow']], 
    prices[['symbol', 'date', 'stockPrice']], 
    on=['symbol', 'date'], how='inner'
)
assets = assets.drop_duplicates(subset=['symbol'])
merged_df = pd.merge(
    merged_df, 
    assets[['symbol', 'sector', 'name']], 
    on=['symbol'], how='inner'
)

merged_df = pd.merge(
    merged_df, 
    financial_growth[['symbol', 'calendarYear', 'operatingCashFlowGrowth']], 
    on=['symbol', 'calendarYear'], how='inner'
)

# merged_df = pd.merge(
#     merged_df, 
#     key_metrics_ttm[['symbol', 'enterpriseValueOverEBITDATTM']], 
#     on=['symbol'], how='inner'
# )

merged_df['calendarYear'] = merged_df['calendarYear'].astype(int)


# Preprocessing: Remove unwanted rows
merged_df = merged_df.dropna()
# merged_df = merged_df[
#     (merged_df['enterpriseValue'] > 0) &
#     (merged_df['evToFreeCashFlow'] > 0) &
#     (merged_df['enterpriseValueOverEBITDA'] > 0)
#     ]

# User selects a year, e.g., 2022
selected_year = 2023

def filter_positive_growth(group):
    group = group.sort_values(by='calendarYear')
    recent_years = group[group['calendarYear'] <= selected_year].tail(6)  # Include the selected year and the 5 prior years
    return len(recent_years) == 6 and all(recent_years['operatingCashFlowGrowth'] > 0)

merged_df = merged_df.groupby('symbol').filter(filter_positive_growth)

# Now filter for the selected year only
merged_df = merged_df[merged_df['calendarYear'] == selected_year]
merged_df = merged_df.sort_values('operatingCashFlowGrowth', ascending=False)

# Print the filtered dataframe
print(merged_df)







# # Filter for the lowest EV/EBITDA per sector
# lowest_per_sector = consistent_growth_stocks.loc[consistent_growth_stocks.groupby('sector')['enterpriseValueOverEBITDA'].idxmin()]

# # Select columns to display
# columns_to_display = [
#     'symbol', 'name', 'calendarYear', 'sector', 
#     'enterpriseValue', 'evToFreeCashFlow', 'enterpriseValueOverEBITDA', 
#     'stockPrice', 'operatingCashFlowGrowth'
# ]

# lowest_per_sector = lowest_per_sector[columns_to_display]

# # Print the final DataFrame
# print(lowest_per_sector)