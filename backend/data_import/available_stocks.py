import os
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import time
import sqlite3

load_dotenv()

api_key = os.getenv('API_KEY')

table_name = 'Assets'
url = f'https://financialmodelingprep.com/api/v3/available-traded/list?apikey={api_key}'
r = requests.get(url, timeout=10)
available_traded = r.json()

available_exchanges = ['NYSE', 'HEL', 'OSL', 'STO', 'LSE', 'CPH', 'XETRA', 'NASDAQ', 'TSX', 'AMEX']

df = pd.DataFrame(available_traded)
df = df[(df['type'] == 'stock') & (df['exchangeShortName'].isin(available_exchanges))]

def get_industry_sector(ticker):
    '''Getting the industry and sector for the different assets '''
    url = f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}'
    r = requests.get(url)
    r = r.json()[0]
    industry = r['industry']
    sector = r['sector']
    return industry, sector

# Initialize empty lists to store industries and sectors
industries = []
sectors = []

# Iterate over each symbol in the DataFrame with tqdm to track progress
for symbol in tqdm(df['symbol'], desc="Fetching industry and sector data"):
    industry, sector = get_industry_sector(symbol)
    industries.append(industry)
    sectors.append(sector)
    
# Assign the industries and sectors to the DataFrame
df['industry'] = industries
df['sector'] = sectors

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df = df.dropna(subset=['name'])
df = df.dropna(subset=['symbol'])
df = df.dropna(subset=['sector'])
df = df.dropna(subset=['industry'])



conn = sqlite3.connect(r'backend\data\financial_data.db')
df.to_sql(table_name, conn, if_exists='append', index=False)

# Print the updated DataFrame
print(df)
