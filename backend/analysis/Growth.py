import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
import streamlit as st

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
    financial_growth[['symbol', 'calendarYear', 'operatingCashFlowGrowth', 'revenueGrowth']], 
    on=['symbol', 'calendarYear'], how='inner'
)


merged_df['calendarYear'] = merged_df['calendarYear'].astype(int)


# Preprocessing: Remove unwanted rows
merged_df = merged_df.dropna()

selected_year = 2023

def filter_positive_growth(group):
    group = group.sort_values(by='calendarYear')
    recent_years = group[group['calendarYear'] <= selected_year].tail(6)  # Include the selected year and the 5 prior years
    return (
        len(recent_years) == 6 and 
        all(recent_years['operatingCashFlowGrowth'] > 0)
    )

merged_df = merged_df.groupby('symbol').filter(filter_positive_growth)

# Now filter for the selected year only
merged_df = merged_df[merged_df['calendarYear'] == selected_year]
merged_df = merged_df.sort_values('operatingCashFlowGrowth', ascending=False)

# Rank based on enterpriseValueOverEBITDA and revenueGrowth
merged_df['evToEbitdaRank'] = merged_df['enterpriseValueOverEBITDA'].rank()
merged_df['revenueGrowthRank'] = merged_df['revenueGrowth'].rank(ascending=False)

# Combine rankings into a single scoring column
merged_df['Score'] = merged_df['evToEbitdaRank'] + merged_df['revenueGrowthRank']

# Group by sector and get the top ten stocks for each sector
sectors = merged_df['sector'].unique()
columns_to_display = [
    'symbol', 'name', 'calendarYear', 'sector',
    'enterpriseValueOverEBITDA',
    'revenueGrowth', 'Score'
]
st.set_page_config(layout="wide")
st.title("Screenr")

# Create tabs for each sector
tabs = st.tabs([f"{sector}" for sector in sectors])


for tab, sector in zip(tabs, sectors):
    with tab:
        sector_df = merged_df[merged_df['sector'] == sector]
        top_ten_stocks = sector_df.nsmallest(20, 'Score')[columns_to_display]
        st.write(f"Top ten stocks in sector: {sector}")
        st.dataframe(top_ten_stocks, use_container_width=True, height=750)

# To run the app, use the command: streamlit run /path/to/this/file.py