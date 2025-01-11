import os
import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm  # Import tqdm for progress bars

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

def get_sector(symbol):
    """
    Fetch the sector for a given symbol. Try the original symbol, then fallback to a '-' variation.
    """
    # Try original symbol first
    url = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()
        if data and isinstance(data, list) and 'sector' in data[0]:
            return data[0]['sector'], symbol  # Return sector and the used symbol
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
    except (IndexError, KeyError):
        print(f"Data not found for {symbol}")
    
    # Fallback to the '-' variation
    fallback_symbol = symbol.replace('.', '-', symbol.count('.') - 1)  # Replace last '.' with '-'
    url = f'https://financialmodelingprep.com/api/v3/profile/{fallback_symbol}?apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and 'sector' in data[0]:
            return data[0]['sector'], fallback_symbol  # Return sector and the fallback symbol
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {fallback_symbol}: {e}")
    except (IndexError, KeyError):
        print(f"Data not found for {fallback_symbol}")
    
    return 'Unknown', symbol  # Default if both variations fail

# Initialize an empty list to store the processed DataFrames
processed_dataframes = []

# Process each CSV file with tqdm for progress tracking
for csv_file, extension in tqdm(csv_files_with_extensions.items(), desc="Processing files"):
    # Read the CSV file, selecting only the columns you need
    print(f"\nProcessing file: {csv_file}")
    df = pd.read_csv(Path(f'backend/data/Stock_lists/Nordnet/{csv_file}'), delimiter=';', usecols=columns_to_read)
    
    # Rename the columns after reading the file
    df.columns = renaming_columns
    
    # Create the 'Symbol' column, appending the correct extension if it exists
    df['Symbol'] = df['Nordnet Symbol'].str.replace(' ', '.').str.strip() + extension
    
    # Initialize an empty list to store sectors
    sectors = []
    
    # Iterate through each row and fetch the sector with tqdm for row-level progress tracking
    for index, row in tqdm(df.iterrows(), desc=f"Fetching sectors for {csv_file}", total=len(df), leave=False):
        symbol = row['Symbol']
        sector, used_symbol = get_sector(symbol)
        sectors.append(sector)  # Append the fetched sector to the list

        # Update the 'Symbol' column if the fallback variation was used
        if used_symbol != symbol:
            df.at[index, 'Symbol'] = used_symbol  # Update the symbol in the DataFrame
    
    # Add the 'Sector' column to the DataFrame
    df['Sector'] = sectors
    
    # Filter out rows where the sector is "Unknown"
    df = df[df['Sector'] != 'Unknown']
    
    # Append the processed DataFrame to the list
    processed_dataframes.append(df)

# Concatenate all processed DataFrames into a single DataFrame
final_df = pd.concat(processed_dataframes, ignore_index=True)

# Print the resulting DataFrame
print("\nFinal DataFrame after processing all CSV files:")
print(final_df)

# Optionally, save the final DataFrame to a CSV file
final_df.to_csv(Path('backend/data/Stock_lists/Nordnet/Processed_Stocks.csv'), index=False)
