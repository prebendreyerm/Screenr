import pandas as pd
import sqlite3
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')

# Connect to the SQLite database
db_path = Path('backend/data/financial_data.db')
conn = sqlite3.connect(db_path)
df = pd.read_sql_query('SELECT * FROM Assets', conn)

# List to store the top 20 most undervalued stocks
undervalued_stocks = []

for symbol in df['symbol']:
    url = f'https://financialmodelingprep.com/stable/custom-discounted-cash-flow?symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            result = data[0]
            equity_value_per_share = result.get("equityValuePerShare")
            stock_price = result.get("price")

            if equity_value_per_share is not None and stock_price is not None and stock_price > 0:
                undervaluation = ((equity_value_per_share - stock_price) / stock_price) * 100  # Percentage undervaluation
                stock_entry = {
                    "symbol": symbol,
                    "equity_value_per_share": equity_value_per_share,
                    "stock_price": stock_price,
                    "undervaluation (%)": undervaluation,
                    "date": result.get("year")
                }

                if len(undervalued_stocks) < 20:
                    # If the list has fewer than 20 stocks, just add the new one
                    undervalued_stocks.append(stock_entry)
                    undervalued_stocks.sort(key=lambda x: x["undervaluation (%)"], reverse=True)
                    print("\nUpdated Top 20 Most Undervalued Stocks (%):")
                    for stock in undervalued_stocks:
                        print(stock)
                else:
                    # List is full, check if new stock is more undervalued than the least undervalued one
                    if undervaluation > undervalued_stocks[-1]["undervaluation (%)"]:
                        undervalued_stocks[-1] = stock_entry  # Replace the least undervalued stock
                        undervalued_stocks.sort(key=lambda x: x["undervaluation (%)"], reverse=True)
                        print("\nUpdated Top 20 Most Undervalued Stocks (%):")
                        for stock in undervalued_stocks:
                            print(stock)

    else:
        print(f"Error fetching data for symbol {symbol}")
