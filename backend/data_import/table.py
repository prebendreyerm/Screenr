import sqlite3
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

api_key = os.getenv('API_KEY')
# pd.set_option('display.max_rows', None)

def get_db_connection():
    db_path = Path('backend/data/financial_data.db')
    conn = sqlite3.connect(db_path)
    return conn

def list_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query to get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Check each table for a 'price' column
    for table in tables:
        table_name = table[0]
        # Query to get column names for the current table
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Check if any column is named 'price'
        if any(column[1] == 'sector' for column in columns):
            print(f"Table '{table_name}' has a 'sector' column.")
        else:
            print(f"Table '{table_name}' does not have a 'sector' column.")

    conn.close()

def get_table(table):
    conn = get_db_connection()
    cursor = conn.cursor()
    df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
    return df


def delete_tables(table_name):
    conn = get_db_connection()

    # Drop the table
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    print(f"Table {table_name} deleted (if it existed).")

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print("Finished deleting the table.")


def delete_strategy(strategy_name):
    # Path to the database
    db_path = "backend/data/financial_data.db"

    # Connect to the database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Delete the strategy with the specified name
    cursor.execute("DELETE FROM ScoringStrategies WHERE strategy_name = ?", (strategy_name,))
    print(f"Strategy '{strategy_name}' deleted (if it existed).")

    # Commit changes and close connection
    connection.commit()
    connection.close()

    print("Finished deleting the strategy.")

a = get_table('historicalPricesAnnual')
print(a.columns)


