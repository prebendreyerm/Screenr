import os, sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from statsmodels.tsa.stattools import coint
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


load_dotenv()
api_key = os.getenv('API_KEY')



def get_crypto_data(crypto):
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{crypto}?apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    historical_data = data['historical']
    df = pd.DataFrame(historical_data)
    return df


def match_dates(df1, df2):
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    df1.set_index('date', inplace=True)
    df2.set_index('date', inplace=True)
    df1, df2 = df1.align(df2, join='inner', axis=0)
    return df1, df2

cryptocurrencies = [
    "BTCUSD", "ETHUSD", "XRPUSD", "SOLUSD", "BNBUSD", "DOGEUSD", "ADAUSD", "TRXUSD", 
    "LINKUSD", "AVAXUSD", "XLMUSD", "SUIUSD", "TONUSD", "HBARUSD", "SHIBUSD", "DOTUSD", 
    "LTCUSD", "LEOUSD", "HYPEUSD", "BCHUSD", "BGBUSD", "UNIUSD", "PEPEUSD", "NEARUSD", 
    "OMUSD", "AAVEUSD", "TRUMPUSD", "ONDOUSD", "APTUSD", "ICPUSD", "XMRUSD", "MNTUSD", 
    "ETCUSD", "VETUSD", "TAOUSD", "POLUSD", "CROUSD", "KASUSD", "OKBUSD", "ALGOUSD", 
    "FILUSD", "RENDERUSD", "ARBUSD", "FETUSD", "ATOMUSD", "ENAUSD", "TIAUSD", "GTUSD", 
    "LDOUSD", "RAYUSD", "INJUSD", "STXUSD", "IMXUSD", "THETAUSD", "OPUSD", "BONKUSD", 
    "DEXEUSD", "MOVEUSD", "JUPUSD", "GRTUSD", "WLDUSD", "KCSUSD", "SUSD", "XDCUSD", 
    "JASMYUSD", "SEIUSD", "FLRUSD", "QNTUSD", "FLOKIUSD", "SANDUSD", "VIRTUALUSD", 
    "ENSUSD", "EOSUSD", "GALAUSD", "WIFUSD", "XTZUSD", "SPXUSD", "KAIAUSD", "IOTAUSD", 
    "ARUSD", "PYTHUSD", "BTTUSD", "MKRUSD", "FLOWUSD", "XCNUSD", "NEOUSD", "CRVUSD", 
    "JTOUSD", "BSVUSD", "RONUSD", "PENGUUSD", "NEXOUSD", "MELANIAUSD", "FARTCOINUSD", 
    "AEROUSD"
]

results = []

def scoring(crypto1, crypto2):
    data1 = get_crypto_data(crypto1)
    data2 = get_crypto_data(crypto2)
    data1, data2 = match_dates(data1, data2)
    score, pvalue, _ = coint(data1['close'], data2['close'])
    if pvalue < 0.05:
        results.append((crypto1, crypto2, pvalue, score))
    return results

def check_for_stationarity(X, cutoff=0.01):
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print(f'p-value = {pvalue}. The series is likely stationary.')
        return True, pvalue
    else:
        print(f'p-value = {pvalue}. The series is likely non-stationary.')
        return False, pvalue

start_date = '2020-01-01'
end_date = '2024-01-01'



def cointegration_test(crypto1, crypto2):
    try:
        X1 = get_crypto_data(crypto1)
        X2 = get_crypto_data(crypto2)

        # Restrict to the given date range
        X1 = X1[(X1['date'] >= start_date) & (X1['date'] <= end_date)]
        X2 = X2[(X2['date'] >= start_date) & (X2['date'] <= end_date)]

        # Match dates
        X1, X2 = match_dates(X1, X2)

        # Convert to time series format
        X1 = X1['close'][::-1].values
        X2 = X2['close'][::-1].values

        # Ensure there's enough data
        if len(X1) < 10 or len(X2) < 10:
            raise ValueError("Insufficient data for cointegration test")

        # Run OLS regression
        X1 = sm.add_constant(X1)
        results = sm.OLS(X2, X1).fit()
        beta = results.params[1]
        z = X2 - beta * X1[:, 1]  # Residuals

        # Check for stationarity
        boolean, pvalue = check_for_stationarity(z)
        return boolean, pvalue, z

    except Exception as e:
        print(f"Error in cointegration test for {crypto1} and {crypto2}: {e}")
        return False, None



for i in tqdm(range(len(cryptocurrencies))):
    for j in tqdm(range(i + 1, len(cryptocurrencies))):
        crypto1 = cryptocurrencies[i]
        crypto2 = cryptocurrencies[j]
        boolean, pvalue = cointegration_test(crypto1, crypto2)
        if boolean == True:
            print(f'{crypto1} and {crypto2} are cointegrated with p-value: {pvalue}')
            results.append((crypto1, crypto2, pvalue))
        else:
            print(f'{crypto1} and {crypto2} are not cointegrated with p-value: {pvalue}')
results = pd.DataFrame(results, columns=['crypto1', 'crypto2', 'pvalue'])
results.to_csv('crypto_pairs.csv', index=False)