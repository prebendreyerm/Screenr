import os
import requests
import fmp
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import Process

load_dotenv()

api_key = os.getenv('API_KEY')


def update_keyMetricsAnnual():
    tickers = fmp.get_all_tickers()
    table = 'KeyMetricsAnnual'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=annual&apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)


def update_keyMetricsQuarter():
    tickers = fmp.get_all_tickers()
    table = 'KeyMetricsQuarter'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=quarter&apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)

def update_ratiosAnnual():
    tickers = fmp.get_all_tickers()
    table = 'RatiosAnnual'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=annual&apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)


def update_ratiosQuarter():
    tickers = fmp.get_all_tickers()
    table = 'RatiosQuarter'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=quarter&apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)

def update_keyMetricsTTM():
    tickers = fmp.get_all_tickers()
    table = 'KeyMetricsTTM'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/key-metrics-TTM/{ticker}?apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)

def update_ratiosTTM():
    tickers = fmp.get_all_tickers()
    table = 'RatiosTTM'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/ratios-TTM/{ticker}?apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)


def update_FinancialGrowthAnnual():
    tickers = fmp.get_all_tickers()
    table = 'FinancialGrowthAnnual'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?period=annual&apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)


def update_FinancialGrowthQuarter():
    tickers = fmp.get_all_tickers()
    table = 'FinancialGrowthQuarter'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?period=Quarter&apikey={api_key}'
        fmp.fetch_and_update_data(url, table, ticker)

def update_Prices():
    # Get all the tickers
    tickers = fmp.get_all_tickers()
    
    # Split the tickers into smaller batches if needed (FMP has a limit of 100 symbols per request)
    batch_size = 100
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    
    table = 'Prices'
    
    for batch in tqdm(ticker_batches):
        # Join the tickers into a single comma-separated string for the URL
        ticker_string = ','.join(batch)
        url = f'https://financialmodelingprep.com/api/v3/quote/{ticker_string}?apikey={api_key}'
        
        # Fetch and insert data for the batch
        fmp.fetch_and_insert_data(url, table)
def update_HistoricalPricesAnnual():
    tickers = fmp.get_all_tickers()
    table = 'HistoricalPricesAnnual'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}/?period=annual&apikey={api_key}'
        fmp.fetch_and_update_data(url,table,ticker)


def update_HistoricalPricesQuarter():
    tickers = fmp.get_all_tickers()
    table = 'HistoricalPricesQuarter'
    # fmp.clear_table(table)
    for ticker in tqdm(tickers):
        url = f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}/?period=quarter&apikey={api_key}'
        fmp.fetch_and_update_data(url,table,ticker)




def process_one():
    # update_ratiosQuarter()
    # update_ratiosAnnual()
    # update_keyMetricsQuarter()
    # update_FinancialGrowthQuarter()
    update_ratiosTTM()
    # update_Prices()


    

def process_two():
    # update_HistoricalPricesAnnual()
    # update_HistoricalPricesQuarter()
    # update_FinancialGrowthAnnual()
    update_keyMetricsTTM()
    # update_keyMetricsAnnual()


if __name__ == '__main__':
    # Create processes
    p1 = Process(target=process_one)
    p2 = Process(target=process_two)

    # Start processes
    p1.start()
    p2.start()

    # Wait for both processes to complete
    p1.join()
    p2.join()


    print("All updates completed.")

