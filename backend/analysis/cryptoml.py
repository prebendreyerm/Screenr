import pandas as pd
import requests
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Load API key
load_dotenv()
api_key = os.getenv('API_KEY')

def load_crypto_data(crypto_symbol, start_date, end_date):
    """Fetch historical data for a given cryptocurrency over a date range."""
    url = f'https://financialmodelingprep.com/api/v3/historical-chart/5min/{crypto_symbol}?from={start_date}&to={end_date}&apikey={api_key}'
    r = requests.get(url)
    r = r.json()
    df = pd.DataFrame(r)
    return df

def generate_date_ranges(start_date, end_date, interval_days=15):
    """Generate date ranges between start_date and end_date with a given interval."""
    date_ranges = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date < end_date:
        next_date = current_date + timedelta(days=interval_days)
        if next_date > end_date:
            next_date = end_date
        date_ranges.append((current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')))
        current_date = next_date + timedelta(days=1)
    
    return date_ranges

# Generate date ranges from 2020 to 2023
date_ranges = generate_date_ranges('2019-01-01', '2023-10-30', interval_days=10)


# Load BTC data for multiple periods
btc_data_parts = []



crypto_symbols = [
    'BTCUSD', 'ETHUSD', 'BNBUSD', 'USDTUSD', 'XRPUSD', 'USDCUSD', 'ADAUSD', 'DOGEUSD', 'SOLUSD', 'TRXUSD',
    'DOTUSD', 'TONUSD', 'MATICUSD', 'LTCUSD', 'SHIBUSD', 'AVAXUSD', 'LINKUSD', 'ATOMUSD', 'XMRUSD', 'OKBUSD',
    'ETCUSD', 'XLMUSD', 'BCHUSD', 'FILUSD', 'LDOUSD', 'WBNBUSD', 'APTUSD', 'ARBUSD', 'NEARUSD', 'HBARUSD',
    'QNTUSD', 'VETUSD', 'CROUSD', 'ICPUSD', 'ALGOUSD', 'GRTUSD', 'EGLDUSD', 'MKRUSD', 'AAVEUSD', 'KLAYUSD',
    'STXUSD', 'FTMUSD', 'XECUSD', 'IMXUSD', 'AXSUSD', 'SANDUSD', 'TWTUSD', 'RNDRUSD', 'INJUSD', 'GMXUSD',
    'APEUSD', 'THETAUSD', 'EOSUSD', 'FLOWUSD', 'CHZUSD', 'WOOUSD', 'DYDXUSD', 'KASUSD', 'ZECUSD', 'CAKEUSD',
    'CFXUSD', 'RSRUSD', 'MINAUSD', 'OPUSD', 'SNXUSD', 'LUNCUSD', 'GMUSD', 'DASHUSD', 'ENSUSD', 'RUNEUSD',
    'NEXOUSD', 'CRVUSD', 'XRDUSD', 'FTTUSD', 'MIOTAUSD', 'COMPUSD', 'ZILUSD', 'BATUSD', 'KAVAUSD', '1INCHUSD',
    'CELOUSD', 'GALAUSD', 'QTUMUSD', 'HOTUSD', 'BTTUSD', 'YFIUSD', 'LRCUSD', 'BALUSD', 'ANKRUSD', 'FLUXUSD',
    'STORJUSD', 'RVNUSD', 'CVCUSD', 'POWRUSD', 'HNTUSD', 'OMGUSD', 'XYOUSD', 'WAVESUSD', 'BANDUSD', 'ZENUSD'
]

# Fetch data for each symbol and date range
for crypto in tqdm(crypto_symbols):
    for start_date, end_date in date_ranges:
        data_part = load_crypto_data(crypto, start_date, end_date)
        btc_data_parts.append(data_part)  # Update variable name if needed


# Combine all parts into a single DataFrame
btc_data = pd.concat(btc_data_parts).reset_index(drop=True)

# Save the combined data to a CSV file
btc_data.to_csv('crypto_historical_data.csv')
print("Data saved to crypto_historical_data.csv")

# Sort, process, and set up the dataset
btc_data.sort_values(by='date', inplace=True)
btc_data['date'] = pd.to_datetime(btc_data['date'])
btc_data.set_index('date', inplace=True)



# Resample to daily data
daily_data = btc_data.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Create a target column: 1 for price increase, 0 for price decrease
daily_data['target'] = (daily_data['close'].shift(-1) > daily_data['close']).astype(int)

# Drop the last row (no target value)
daily_data = daily_data[:-1]

# Print the updated dataset size
print(f"Dataset size: {daily_data.shape}")

# Prepare features and target
X = daily_data[['open', 'high', 'low', 'close', 'volume']]
y = daily_data['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='tanh', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='tanh'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='tanh'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the model to a file
model.save('btc_price_predictor.h5')


# Plot predictions vs actuals
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Model Predictions vs Actual Values')
plt.legend()
plt.grid()
plt.show()

from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('btc_price_predictor.h5')

# Fetch data for the date range December 1st to today
new_data = load_crypto_data('BTCUSD', '2023-12-01', datetime.today().strftime('%Y-%m-%d'))

# Process the new data
new_data['date'] = pd.to_datetime(new_data['date'])
new_data.set_index('date', inplace=True)

# Resample to daily data
new_daily_data = new_data.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Create the target column for the new data
new_daily_data['target'] = (new_daily_data['close'].shift(-1) > new_daily_data['close']).astype(int)

# Drop the last row (no target value)
new_daily_data = new_daily_data[:-1]

# Prepare features and target for the new data
X_new = new_daily_data[['open', 'high', 'low', 'close', 'volume']]
y_new = new_daily_data['target']

# Scale the features using the same scaler used during training
X_new_scaled = scaler.transform(X_new)
# Make predictions on the new data
y_new_pred = (loaded_model.predict(X_new_scaled) > 0.5).astype(int).flatten()

# Evaluate the accuracy on the new data
new_accuracy = accuracy_score(y_new, y_new_pred)
print(f"Accuracy on new data: {new_accuracy:.2f}")

# Plot predictions vs actual values for the new data
plt.figure(figsize=(12, 6))
plt.plot(y_new.values, label='Actual', marker='o')
plt.plot(y_new_pred, label='Predicted', marker='x', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Model Predictions vs Actual Values (December 1st - Today)')
plt.legend()
plt.grid()
plt.show()
