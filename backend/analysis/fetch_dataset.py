import sqlite3
from pathlib import Path
import pandas as pd


# Connect to the SQLite database
db_path = Path('backend/data/financial_data.db')
conn = sqlite3.connect(db_path)

ratios = pd.read_sql_query('SELECT * FROM RatiosQuarter', conn)
price = pd.read_sql_query('SELECT * FROM HistoricalPricesQuarter', conn)

df = pd.merge(ratios, price, on=['symbol', 'date'], how='inner')

df.to_csv('data.csv')