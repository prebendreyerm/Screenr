from flask import Flask, render_template, jsonify
import threading
import time
from pairs_crypto import PairTrading

app = Flask(__name__)

# Create a global instance of your PairTrading class
pair_trading = PairTrading(pairs=[])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    # Return the required data as JSON
    return jsonify({
        'active_pairs': list(pair_trading.active_pairs.keys()),
        'positions': pair_trading.positions,
        'capital': pair_trading.capital
    })

def run_bot():
    # Run the main function of the bot in a separate thread
    pair_trading.main()

if __name__ == '__main__':
    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    
    # Run the Flask app
    app.run(debug=True)
