from flask import Flask, render_template
import threading
import numpy as np
from pairs_crypto_returns_spread import PairTrading

app = Flask(__name__)

# Assuming PairTrading class is already defined as in your previous code
pair_trading = PairTrading(pairs=[])

@app.route('/')
def index():
    active_pairs = [(pair, timestamp) for pair, timestamp in pair_trading.active_pairs.items()]
    positions = [(pair, details) for pair, details in pair_trading.positions.items()]
    
    # Fetch the latest prices for all symbols in positions
    last_prices = {}
    for pair, details in positions:
        last_prices[details['long']['symbol']] = pair_trading.fetch_last_price(details['long']['symbol'])['price']
        last_prices[details['short']['symbol']] = pair_trading.fetch_last_price(details['short']['symbol'])['price']
    
    # Calculate spreads and their limits
    spread_limits = {}
    for pair in pair_trading.active_pairs:
        spread = pair_trading.calculate_spread(pair[0], pair[1])
        if spread is not None:
            mean = np.mean(spread)
            std = np.std(spread)
            upper_open_limit = mean + 2 * std
            lower_open_limit = mean - 2 * std
            upper_close_limit = mean + 0.5 * std
            lower_close_limit = mean - 0.5 * std
            spread_limits[pair] = {
                'upper_close_limit': upper_close_limit,
                'spread': spread[-1],
                'lower_close_limit': lower_close_limit,
                'upper_open_limit': upper_open_limit,
                'lower_open_limit': lower_open_limit
            }
    
    capital = pair_trading.capital
    total_capital = pair_trading.total_capital
    return render_template('index.html', active_pairs=active_pairs, positions=positions, spread_limits=spread_limits, last_prices=last_prices, capital=capital, total_capital=total_capital)

def run_trading():
    pair_trading.main()

if __name__ == '__main__':
    # Start the pair trading logic in a separate thread
    threading.Thread(target=run_trading, daemon=True).start()
    
    # Start the Flask web server
    app.run(debug=False)
