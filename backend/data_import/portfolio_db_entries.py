import sqlite3
import pandas as pd

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(r'backend\data\financial_data.db')
cursor = conn.cursor()

# Create Assets table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE,
    cashAmount REAL,
    totalValue REAL,
    lastUpdate TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS Holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    shares REAL,
    costBasis REAL,
    startDate TEXT,
    endDate TEXT,
    lastUpdate TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS StockPrices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    holding_id INTEGER,
    date TEXT,
    closePrice REAL,
    FOREIGN KEY (holding_id) REFERENCES Holdings(id)
)
''') 
cursor.execute('''
CREATE TABLE IF NOT EXISTS Transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    shares REAL,
    price REAL,
    action TEXT, -- 'buy', 'sell', 'deposit', 'withdraw'
    date TEXT,
    FOREIGN KEY (ticker) REFERENCES Holdings(ticker)
)
''')