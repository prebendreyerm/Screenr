import os
import pandas as pd
import requests
import sqlite3
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# Load API key
load_dotenv()
api_key = os.getenv('API_KEY')

# List of CSV files and their corresponding extensions
csv_files_with_extensions = {
    'Copenhagen.csv': '.CO',
    'Frankfurt.csv': '.F',
    'Helsinki.csv': '.HE',
    'London.csv': '.L',
    'Oslo.csv': '.OL',
    'Stockholm.csv': '.ST',
    'Toronto.csv': '.TO',
    'US.csv': ''  # US stocks don't have an extension
}

columns_to_read = ['Navn', 'Ticker']
renaming_columns = ['Name', 'Nordnet Symbol']

# Database connection
def get_db_connection():
    db_path = Path('backend/data/financial_data.db')
    conn = sqlite3.connect(db_path)
    return conn

# Create Assets table
def create_assets_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL UNIQUE,
        nordnet_symbol TEXT NOT NULL,
        name TEXT NOT NULL,
        industry TEXT,
        sector TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Fetch sector information
def get_sector(symbol):
    url = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and 'sector' in data[0]:
            return data[0]['sector'], symbol  # Return sector and the used symbol
    except (requests.exceptions.RequestException, IndexError, KeyError):
        pass

    # Fallback to alternate symbol format
    fallback_symbol = symbol.replace('.', '-', symbol.count('.') - 1)
    url = f'https://financialmodelingprep.com/api/v3/profile/{fallback_symbol}?apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and 'sector' in data[0]:
            return data[0]['sector'], fallback_symbol
    except (requests.exceptions.RequestException, IndexError, KeyError):
        pass

    return 'Unknown', symbol

# Sync CSV data with the database
def sync_csv_to_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    for csv_file, extension in tqdm(csv_files_with_extensions.items(), desc="Processing files"):
        print(f"\nProcessing file: {csv_file}")
        # Read and preprocess CSV
        df = pd.read_csv(Path(f'backend/data/Stock_lists/Nordnet/{csv_file}'), delimiter=';', usecols=columns_to_read)
        df.columns = renaming_columns
        df['Symbol'] = df['Nordnet Symbol'].str.replace(' ', '.').str.strip() + extension

        # Fetch existing symbols from the database
        existing_symbols = pd.read_sql_query("SELECT symbol FROM Assets", conn)['symbol'].tolist()

        # Separate symbols into those to add, remove, and retain
        csv_symbols = df['Symbol'].tolist()
        symbols_to_add = [symbol for symbol in csv_symbols if symbol not in existing_symbols]
        symbols_to_remove = [symbol for symbol in existing_symbols if symbol not in csv_symbols]

        # Add new entries
        for _, row in df[df['Symbol'].isin(symbols_to_add)].iterrows():
            sector, used_symbol = get_sector(row['Symbol'])
            if sector != 'Unknown':  # Only add if sector is successfully fetched
                cursor.execute('''
                INSERT INTO Assets (symbol, nordnet_symbol, name, sector)
                VALUES (?, ?, ?, ?)
                ''', (used_symbol, row['Nordnet Symbol'], row['Name'], sector))
        
        # Remove entries no longer in the CSV
        for symbol in symbols_to_remove:
            cursor.execute("DELETE FROM Assets WHERE symbol = ?", (symbol,))

        # Commit changes after processing each file
        conn.commit()

    conn.close()

# Main logic
if __name__ == "__main__":
    create_assets_table()
    sync_csv_to_database()
