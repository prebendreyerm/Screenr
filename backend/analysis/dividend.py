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
# merged_df = merged_df[
#     (merged_df['enterpriseValue'] > 0) &
#     (merged_df['evToFreeCashFlow'] > 0) &
#     (merged_df['enterpriseValueOverEBITDATTM'] > 0) &
#     (merged_df['payoutRatio'] > 0) &
#     (merged_df['payoutRatio'] < 0.2) &
#     (merged_df['dividendYield'] > 0)
# ]

# Convert calendarYear to integer for filtering
merged_df['calendarYear'] = merged_df['calendarYear'].astype(int)

# User selects a year, e.g., 2022
selected_year = 2023

# Filter for the selected year and find stocks with consistent dividend growth
filtered_df = merged_df[merged_df['calendarYear'] <= selected_year]

# Group by symbol and check for consistent dividend growth
def filter_positive_growth(group):
    group = group.sort_values(by='calendarYear')
    recent_years = group[group['calendarYear'] <= selected_year].tail(6)  # Include the selected year and the 5 prior years
    return len(recent_years) == 6 and all(recent_years['dividendsperShareGrowth'] > 0)

consistent_dividend_stocks = filtered_df.groupby('symbol').filter(filter_positive_growth)

consistent_dividend_stocks = consistent_dividend_stocks[consistent_dividend_stocks['calendarYear'] == selected_year]

# Filter for the lowest EV/EBITDA per sector
lowest_per_sector = consistent_dividend_stocks.loc[consistent_dividend_stocks.groupby('sector')['dividendsperShareGrowth'].idxmax()]

# Select columns to display
columns_to_display = [
    'symbol', 'name', 'calendarYear', 'sector', 
    'enterpriseValue', 'evToFreeCashFlow', 'dividendsperShareGrowth', 
    'stockPrice', 'dividendYield'
]

lowest_per_sector = lowest_per_sector[columns_to_display]

# Rank stocks based on enterprise value and dividend growth
lowest_per_sector['ev_rank'] = lowest_per_sector['enterpriseValue'].rank(method='min', ascending=True)
lowest_per_sector['div_growth_rank'] = lowest_per_sector['dividendsperShareGrowth'].rank(method='min', ascending=False)

# Compute final score (you can adjust weights if needed)
lowest_per_sector['final_score'] = lowest_per_sector['ev_rank'] + lowest_per_sector['div_growth_rank']

# Sort stocks by final score (lower score is better)
ranked_stocks = lowest_per_sector.sort_values(by='final_score')

# Display the ranked stocks
print(ranked_stocks)
