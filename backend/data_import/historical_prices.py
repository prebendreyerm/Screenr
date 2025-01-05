import os
import sqlite3
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')



url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}'