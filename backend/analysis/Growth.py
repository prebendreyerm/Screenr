import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
import streamlit as st

# Constants
DB_PATH = Path('backend/data/financial_data.db')
SELECTED_YEAR = 2023

# Connect to the SQLite database
conn = sqlite3.connect(DB_PATH)

# Load the data
financial_growth = pd.read_sql_query('SELECT * FROM FinancialGrowthAnnual', conn)
key_metrics = pd.read_sql_query('SELECT * FROM KeyMetricsAnnual', conn)
prices = pd.read_sql_query('SELECT * FROM HistoricalPricesAnnual', conn)
assets = pd.read_sql_query('SELECT * FROM Assets', conn)
key_metrics_ttm = pd.read_sql_query('SELECT * FROM KeyMetricsTTM', conn)

# Merge and preprocess data
merged_df = pd.merge(
    key_metrics[['symbol', 'date', 'calendarYear', 'enterpriseValue', 'evToFreeCashFlow', 'operatingCashFlowPerShare', 'freeCashFlowYield', 'enterpriseValueOverEBITDA', 'evToOperatingCashFlow']],
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
    financial_growth[['symbol', 'calendarYear', 'operatingCashFlowGrowth', 'revenueGrowth', 'freeCashFlowGrowth']],
    on=['symbol', 'calendarYear'], how='inner'
)

merged_df = pd.merge(
    merged_df,
    key_metrics_ttm[['symbol', 'enterpriseValueOverEBITDATTM']],
    on=['symbol'], how='inner'
)
merged_df = merged_df.drop_duplicates(subset=['symbol', 'calendarYear'])


merged_df['calendarYear'] = merged_df['calendarYear'].astype(int)

# Ensure freeCashFlowGrowth and freeCashFlowYield are numeric
merged_df['freeCashFlowGrowth'] = pd.to_numeric(merged_df['freeCashFlowGrowth'], errors='coerce')
merged_df['freeCashFlowYield'] = pd.to_numeric(merged_df['freeCashFlowYield'], errors='coerce')

# Preprocessing: Remove unwanted rows
merged_df = merged_df.dropna()

# Filter stocks with positive free cash flow growth and yield in the previous five years
def filter_positive_growth(group):
    group = group.sort_values(by='calendarYear')
    recent_years = group[group['calendarYear'] <= SELECTED_YEAR].tail(6)  # Include the selected year and the 5 prior years
    # print(f"Processing {group['symbol'].iloc[0]}: {recent_years[['calendarYear', 'freeCashFlowGrowth', 'freeCashFlowYield']]}")
    return (
        len(recent_years) == 6 and 
        all(recent_years['freeCashFlowGrowth'] > 0) and
        all(recent_years['freeCashFlowYield'] > 0)
    )

merged_df = merged_df.groupby('symbol').filter(filter_positive_growth)

# Display columns for debugging
columns_to_display = ['freeCashFlowGrowth', 'freeCashFlowYield', 'symbol', 'calendarYear']

# print(merged_df[merged_df['symbol'] == 'MNDY'][columns_to_display])

# Filter for the selected year only
merged_df = merged_df[merged_df['calendarYear'] == SELECTED_YEAR]
merged_df = merged_df.sort_values('freeCashFlowGrowth', ascending=False)

# Rank based on enterpriseValueOverEBITDA and freeCashFlowGrowth
merged_df['evToEbitdaRank'] = merged_df['enterpriseValueOverEBITDATTM'].rank()
merged_df['freeCashFlowGrowthRank'] = merged_df['freeCashFlowGrowth'].rank(ascending=False)

# Combine rankings into a single scoring column
merged_df['Score'] = merged_df['evToEbitdaRank'] + merged_df['freeCashFlowGrowthRank']

# Group by sector and get the top ten stocks for each sector
sectors = merged_df['sector'].unique()
columns_to_display = [
    'symbol', 'name', 'calendarYear', 'sector',
    'enterpriseValueOverEBITDATTM',
    'freeCashFlowGrowth', 'Score'
]

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title("Screenr")

# Create tabs for each sector
tabs = st.tabs([f"{sector}" for sector in sectors])

for tab, sector in zip(tabs, sectors):
    with tab:
        sector_df = merged_df[merged_df['sector'] == sector]
        top_ten_stocks = sector_df.nsmallest(10, 'Score')[columns_to_display]
        st.write(f"Top ten stocks in sector: {sector}")
        st.dataframe(top_ten_stocks, use_container_width=True, height=750)

# To run the app, use the command: streamlit run /path/to/your/script.py
