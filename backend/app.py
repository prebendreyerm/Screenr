import os
import requests
from flask import Flask, jsonify, request
import sqlite3
import pandas as pd
from analysis import scoring
import numpy as np
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv



app = Flask(__name__)
CORS(app)

load_dotenv()

API_KEY = os.getenv('API_KEY')

def get_db_connection():
    conn = sqlite3.connect(r'backend\data\financial_data.db')
    conn.row_factory = sqlite3.Row
    return conn

# Function to fetch historical prices
def fetch_historical_prices(ticker):
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'historical' in data:
            return data['historical']
    return []

# Function to fetch historical currency conversion rates
def fetch_historical_currency_rates(currency_pair, start_date, end_date):
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{currency_pair}?from={start_date}&to={end_date}&apikey={API_KEY}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get('historical', [])  # Adjust as necessary based on the actual structure
    return []



@app.route('/api/currency/conversion-rate', methods=['GET'])
def get_conversion_rate():
    try:
        # Get the currency pair from query parameters (e.g., NOKUSD or SEKUSD)
        currency_pair = request.args.get('pair')
        if not currency_pair:
            return jsonify({'error': 'Currency pair is required'}), 400

        # Fetch historical currency rates for the pair
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        rates = fetch_historical_currency_rates(currency_pair, start_date, end_date)

        return jsonify({'rates': rates})
    
    except Exception as e:
        print(f"Error fetching conversion rates:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks/prices', methods=['POST'])
def get_stock_prices():
    try:
        # Get data from the request body
        data = request.json
        ticker = data.get('ticker')
        start_date = data.get('startDate')
        end_date = data.get('endDate')

        historical_prices = fetch_historical_prices(ticker)

        if start_date and end_date:
            filtered_prices = [
                price for price in historical_prices
                if datetime.fromisoformat(price['date']) >= datetime.strptime(start_date, '%Y-%m-%d') and
                datetime.fromisoformat(price['date']) <= datetime.strptime(end_date, '%Y-%m-%d')
            ]
        else:
            filtered_prices = historical_prices

        # Check if the ticker ends with '.OL' or '.ST' for currency conversion
        if ticker.endswith('.OL'):
            currency_rates = fetch_historical_currency_rates('NOKUSD', start_date, end_date)
        elif ticker.endswith('.ST'):
            currency_rates = fetch_historical_currency_rates('SEKUSD', start_date, end_date)
        else:
            currency_rates = []

        # Create a dictionary for currency rates for easy lookup
        currency_dict = {rate['date']: rate['close'] for rate in currency_rates}  # Assuming 'close' is the conversion rate

        # Adjust prices based on currency conversion
        for price in filtered_prices:
            if ticker.endswith('.OL') or ticker.endswith('.ST'):
                conversion_rate = currency_dict.get(price['date'])
                if conversion_rate:
                    price['close'] *= conversion_rate  # Adjust the price to USD

        return jsonify(filtered_prices)

    except Exception as e:
        print(f"Error fetching stock prices:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks/scores', methods=['GET'])
def get_scored_stocks():
    try:
        # Get query parameters for columns, strategy, pagination, sorting
        selected_columns = request.args.getlist('columns')
        strategy = request.args.get('strategy', 'baseline')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        sort_by = request.args.get('sort_by', 'score')
        sort_order = request.args.get('sort_order', 'desc')
        name_search = request.args.get('name_search', '').lower()
        symbol_search = request.args.get('symbol_search', '').lower()
        sector_search = request.args.get('sector_search', '').lower()

        # Connect to the database
        conn = get_db_connection()
        ratiosTTM = pd.read_sql_query('SELECT * FROM RatiosTTM', conn)
        assets = pd.read_sql_query('SELECT * FROM Assets', conn)
        conn.close()

        # Merge dataframes
        df = pd.merge(assets, ratiosTTM, on='symbol', how='left')
        df.rename(columns={'dividendYielTTM':'dividendYieldTTM'}, inplace=True)
        df.fillna(0, inplace=True)

        # Load the appropriate baseline strategy
        baseline_file = f'C:\\Users\\Preben\\OneDrive\\Dokumenter\\GitHub\\screenr_pub\\backend\\data\\Baselines\\{strategy}.csv'
        scoring_baseline = pd.read_csv(baseline_file)
        scoring_baseline['scoring_columns'] = scoring_baseline['scoring_columns'] + 'TTM'

        # Calculate scores
        df_scored = scoring.calculate_score(df, np.array(scoring_baseline['scoring_columns']), np.array(scoring_baseline['booleans']))

        # Applying search
        if name_search:
            df_scored = df_scored[df_scored['name'].str.lower().str.contains(name_search)]
        if symbol_search:
            df_scored = df_scored[df_scored['symbol'].str.lower().str.contains(symbol_search)]
        if sector_search:
            df_scored = df_scored[df_scored['sector'].str.lower().str.contains(sector_search)]

        # Apply sorting
        sort_by = sort_by if sort_by in df_scored.columns else 'score'
        sort_order = True if sort_order == 'asc' else False
        df_scored = df_scored.sort_values(by=sort_by, ascending=sort_order)

        # Apply pagination
        start = (page - 1) * limit
        end = start + limit
        df_scored_paginated = df_scored.iloc[start:end]

        # Convert DataFrame to a list of dictionaries
        scored_stocks_list = df_scored_paginated.to_dict(orient='records')

        return jsonify({
            'data': scored_stocks_list,
            'page': page,
            'limit': limit,
            'total': len(df_scored)
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    try:
        conn = get_db_connection()
        stocks = conn.execute('SELECT * FROM RatiosTTM').fetchall()
        conn.close()

        stocks_list = [dict(row) for row in stocks]
        return jsonify(stocks_list)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/portfolio/stock', methods=['POST'])
def update_stock():
    try:
        # Get data from the request
        data = request.json
        ticker = data.get('ticker')
        shares = float(data.get('shares'))
        price = float(data.get('price'))
        action = data.get('action')  # 'buy' or 'sell'
        
        # Default the date to today if not provided
        transaction_date = data.get('startDate', datetime.now().strftime('%Y-%m-%d'))

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Retrieve the current holding for the ticker, if it exists
        cursor.execute('SELECT shares, costBasis FROM Holdings WHERE ticker = ?', (ticker,))
        holding = cursor.fetchone()

        # Handling 'buy' action
        if action == 'buy':
            if holding:
                # Update the existing holding: new shares and new cost basis
                current_shares, current_cost_basis = holding
                total_shares = current_shares + shares
                total_cost = (current_cost_basis * current_shares) + (price * shares)
                new_cost_basis = total_cost / total_shares

                cursor.execute(
                    'UPDATE Holdings SET shares = ?, costBasis = ?, lastUpdate = ? WHERE ticker = ?',
                    (total_shares, new_cost_basis, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker)
                )
            else:
                # Insert a new holding
                cursor.execute(
                    'INSERT INTO Holdings (ticker, shares, costBasis, startDate, endDate, lastUpdate) VALUES (?, ?, ?, ?, ?, ?)',
                    (ticker, shares, price, transaction_date, transaction_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )

        # Handling 'sell' action
        elif action == 'sell':
            if holding:
                current_shares, current_cost_basis = holding
                if current_shares >= shares:
                    # If selling all shares, delete the holding
                    if current_shares == shares:
                        cursor.execute('DELETE FROM Holdings WHERE ticker = ?', (ticker,))
                    else:
                        # Otherwise, reduce the shares count
                        new_shares = current_shares - shares
                        cursor.execute(
                            'UPDATE Holdings SET shares = ?, lastUpdate = ? WHERE ticker = ?',
                            (new_shares, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker)
                        )
                else:
                    return jsonify({'error': 'Not enough shares to sell'}), 400
            else:
                return jsonify({'error': 'No holdings found for this ticker'}), 400

        # Log the transaction to the Transactions table
        cursor.execute(
            'INSERT INTO Transactions (ticker, shares, price, date, action) VALUES (?, ?, ?, ?, ?)',
            (ticker, shares, price, transaction_date, action)
        )

        conn.commit()
        conn.close()

        return jsonify({'message': f'Stock {action} successful'})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500




    
@app.route('/api/portfolio/cash', methods=['POST'])
def update_cash():
    try:
        data = request.json
        amount = data.get('amount')
        action = data.get('action')

        if action not in ['deposit', 'withdraw']:
            return jsonify({'error': 'Invalid action'}), 400

        if not isinstance(amount, (int, float)) or amount < 0:
            return jsonify({'error': 'Invalid amount'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        if action == 'deposit':
            cursor.execute('UPDATE Portfolio SET cashAmount = cashAmount + ? WHERE id = 1', (amount,))
        elif action == 'withdraw':
            current_cash = cursor.execute('SELECT cashAmount FROM Portfolio WHERE id = 1').fetchone()['cashAmount']
            if current_cash < amount:
                return jsonify({'error': 'Insufficient funds'}), 400
            cursor.execute('UPDATE Portfolio SET cashAmount = cashAmount - ? WHERE id = 1', (amount,))
        
        conn.commit()

        # Fetch the updated cash amount to return it
        updated_cash = cursor.execute('SELECT cashAmount FROM Portfolio WHERE id = 1').fetchone()['cashAmount']
        return jsonify({'cash': updated_cash, 'message': f'Cash {action} successful'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/portfolio/holdings', methods=['GET'])
def get_holdings():
    try:
        conn = get_db_connection()
        
        # Fetch holdings
        holdings = conn.execute('SELECT * FROM Holdings').fetchall()
        holdings_list = [dict(row) for row in holdings]

        # Fetch cash amount directly from the cashAmount column
        cash = conn.execute('SELECT cashAmount FROM Portfolio WHERE id = 1').fetchone()
        
        # Check if cash exists before accessing it
        cash_amount = cash['cashAmount'] if cash else 0  # Default to 0 if cash is None

        conn.close()
        
        return jsonify({'stocks': holdings_list, 'cash': cash_amount})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/portfolio/transactions', methods=['GET'])
def get_transactions():
    try:
        # Connect to the database
        conn = get_db_connection()
        
        # Fetch all transactions from the Transactions table
        transactions = conn.execute('SELECT * FROM Transactions').fetchall()
        
        # Convert the result into a list of dictionaries
        transactions_list = [
            {
                'id': row[0],  # Use index if row factory is not set
                'ticker': row[1],
                'shares': row[2],
                'price': row[3],
                'date': row[4],
                'action': row[5]
            }
            for row in transactions
        ]

        conn.close()
        
        # Return the transactions list as JSON
        return jsonify({'transactions': transactions_list})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
