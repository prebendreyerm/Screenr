import os
import sqlite3
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

load_dotenv()

api_key = os.getenv('API_KEY')

# Function to fetch and insert data into the specified table
def fetch_and_insert_data(api_url, table_name):
    '''function for fetching and inserting the data into the different tables'''
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data:
                # Convert JSON data to DataFrame
                df = pd.DataFrame(data)

                # Insert data into the corresponding table
                conn = sqlite3.connect(Path('backend/data/financial_data.db'))
                try:
                    df.to_sql(table_name, conn, if_exists='append', index=False)
                    conn.commit()
                except OverflowError as e:
                    print(f'OverflowError encountered while inserting data for {api_url}: {e}')
                finally:
                    conn.close()
            else:
                print(f'No data found for {api_url}')
        else:
            print(f'Failed to fetch data for {api_url}, Status code: {response.status_code}')
    except Exception as e:
        print(f'An error occurred while processing {api_url}: {e}')


def get_all_tickers():
    '''function for getting all the tickers to loop through and populate tables'''
    try:
        conn = sqlite3.connect(Path('backend/data/financial_data.db'))
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT symbol FROM Assets")
        tickers = cursor.fetchall()

        conn.close()

        # Return a list of ticker symbols
        return [ticker[0] for ticker in tickers]
    except Exception as e:
        print(f'An error occurred while fetching tickers: {e}')
        return []
    

def fetch_and_update_data(api_url, table_name, symbol=None):
    '''Function to fetch and insert new data into the given table, including the symbol if provided'''
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data:
                # Convert JSON data to DataFrame
                df = pd.DataFrame(data)

                # If symbol is provided, add it to the DataFrame
                if symbol:
                    df['symbol'] = symbol

                # Connect to the SQLite database
                conn = sqlite3.connect(Path('backend/data/financial_data.db'))

                # Insert the new data into the table
                df.to_sql(table_name, conn, if_exists='append', index=False)
                
                conn.commit()
                conn.close()
            else:
                print(f'No data found for {api_url}')
        else:
            print(f'Failed to fetch data for {api_url}, Status code: {response.status_code}')
    except Exception as e:
        print(f'An error occurred while processing {api_url}: {e}')


def clear_table(table_name):
    '''function to clear the contents of an existing table before updating it with new values'''
    conn = sqlite3.connect(Path('backend/data/financial_data.db'))
    cursor = conn.cursor()

    # Clear the contents of the table
    cursor.execute(f'DELETE FROM {table_name}')
    conn.commit()
    conn.close()

def delete_table(table_name):
    '''Function to delete an existing table from the database.'''
    conn = sqlite3.connect(Path('backend/data/financial_data.db'))
    cursor = conn.cursor()

    # Delete the table
    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
    conn.commit()
    conn.close()



if __name__ == '__main__':
    print('yes')
# # Main script to loop through each ticker and fetch data
# tickers = get_all_tickers()
# for ticker in tqdm(tickers):
#     api_url = f'https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={api_key}'
#     fetch_and_insert_data(api_url, 'KeyRatiosTTM')

