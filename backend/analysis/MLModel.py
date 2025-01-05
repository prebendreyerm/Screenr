import pandas as pd
import sqlite3
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Database connection
DB_PATH = r"backend\data\financial_data.db"

def load_data_from_db():
    print("Loading data from the database...")
    with sqlite3.connect(DB_PATH) as conn:
        ratios = pd.read_sql_query("SELECT * FROM RatiosAnnual", conn)
        financial_growth = pd.read_sql_query("SELECT * FROM FinancialGrowthAnnual", conn)
        keymetrics = pd.read_sql_query("SELECT * FROM KeyMetricsAnnual", conn)
        price = pd.read_sql_query("SELECT * FROM HistoricalPricesAnnual", conn)
    print("Data loaded successfully.")
    return ratios, financial_growth, keymetrics, price

def preprocess_data(ratios, financial_growth, keymetrics, price):
    print("Preprocessing data...")

    # Merge the tables on symbol and date (or calendarYear if needed)
    combined = pd.merge(ratios, financial_growth, on=['symbol', 'date'], how='inner', suffixes=('_ratios', '_financial_growth'))
    combined = pd.merge(combined, keymetrics, on=['symbol', 'date'], how='inner', suffixes=('_combined', '_keymetrics'))
    combined = pd.merge(combined, price[['symbol', 'date', 'stockPrice']], on=['symbol', 'date'], how='inner')

    # Sort by symbol and date to ensure correct temporal alignment
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(by=['symbol', 'date'])

    # Handle duplicate columns that have been renamed by pandas during the merge
    combined = combined.loc[:, ~combined.columns.str.contains('_ratios|_financial_growth|_keymetrics|_combined')]

    # Drop calendarYear and period columns, if they exist
    combined = combined.drop(['calendarYear', 'period'], axis=1, errors='ignore')

    # Handle missing values by forward filling (you can adjust this based on your needs)
    combined = combined.dropna()

    print("Data preprocessing completed. Shape:", combined.shape)
    return combined

# Load data from database
ratios, financial_growth, key_metrics, price = load_data_from_db()

# Preprocess data
df = preprocess_data(ratios, financial_growth, key_metrics, price)

# Create lagged features
df['stockPrice_lagged'] = df.groupby('symbol')['stockPrice'].shift(1)  # Lag 1 day
df = df.dropna()

# Create the target variable: 1 if stock price increased, 0 if it decreased or stayed the same
df['price_direction'] = np.where(df['stockPrice'] > df['stockPrice_lagged'], 1, 0)

# Normalize the features
features = df.drop(columns=['symbol', 'date', 'stockPrice', 'stockPrice_lagged', 'price_direction'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into features (X) and target (y)
X = scaled_features
y = df['price_direction'].values

# No need to reshape to 3D for Dense model, just use 2D: (samples, features)
# X = X.reshape((X.shape[0], 1, X.shape[1]))  # Remove this reshaping step

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the model with Dense layers
model = Sequential()
model.add(Dense(units=128, activation='tanh', input_dim=X_train.shape[1]))  # First Dense layer
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(Dense(units=64, activation='tanh'))  # Second Dense layer
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(units=32, activation='tanh'))  # Third Dense layer
model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model and scaler
model.save('dense_stock_direction_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Predict on the test setimport pandas as pd
import sqlite3
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Database connection
DB_PATH = r"backend\data\financial_data.db"

def load_data_from_db():
    print("Loading data from the database...")
    with sqlite3.connect(DB_PATH) as conn:
        ratios = pd.read_sql_query("SELECT * FROM RatiosQuarter", conn)
        financial_growth = pd.read_sql_query("SELECT * FROM FinancialGrowthQuarter", conn)
        keymetrics = pd.read_sql_query("SELECT * FROM KeyMetricsQuarter", conn)
        price = pd.read_sql_query("SELECT * FROM HistoricalPricesQuarter", conn)
    print("Data loaded successfully.")
    return ratios, financial_growth, keymetrics, price

def preprocess_data(ratios, financial_growth, keymetrics, price):
    print("Preprocessing data...")

    # Merge the tables on symbol and date (or calendarYear if needed)
    combined = pd.merge(ratios, financial_growth, on=['symbol', 'date'], how='inner', suffixes=('_ratios', '_financial_growth'))
    combined = pd.merge(combined, keymetrics, on=['symbol', 'date'], how='inner', suffixes=('_combined', '_keymetrics'))
    combined = pd.merge(combined, price[['symbol', 'date', 'stockPrice']], on=['symbol', 'date'], how='inner')

    # Sort by symbol and date to ensure correct temporal alignment
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(by=['symbol', 'date'])

    # Handle duplicate columns that have been renamed by pandas during the merge
    combined = combined.loc[:, ~combined.columns.str.contains('_ratios|_financial_growth|_keymetrics|_combined')]

    # Drop calendarYear and period columns, if they exist
    combined = combined.drop(['calendarYear', 'period'], axis=1, errors='ignore')

    # Handle missing values by forward filling (you can adjust this based on your needs)
    combined = combined.dropna()

    print("Data preprocessing completed. Shape:", combined.shape)
    return combined

# Load data from database
ratios, financial_growth, key_metrics, price = load_data_from_db()

# Preprocess data
df = preprocess_data(ratios, financial_growth, key_metrics, price)

# Create lagged features
df['stockPrice_lagged'] = df.groupby('symbol')['stockPrice'].shift(1)  # Lag 1 day
df = df.dropna()

# Create the target variable: 1 if stock price increased, 0 if it decreased or stayed the same
df['price_direction'] = np.where(df['stockPrice'] > df['stockPrice_lagged'], 1, 0)

# Normalize the features
features = df.drop(columns=['symbol', 'date', 'stockPrice', 'stockPrice_lagged', 'price_direction'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# Split the data into features (X) and target (y)
X = scaled_features
y = df['price_direction'].values

# No need to reshape to 3D for Dense model, just use 2D: (samples, features)
# X = X.reshape((X.shape[0], 1, X.shape[1]))  # Remove this reshaping step

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# Build the model with Dense layers
model = Sequential()
model.add(Dense(units=128, activation='tanh', input_dim=X_train.shape[1]))  # First Dense layer
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(Dense(units=64, activation='tanh'))  # Second Dense layer
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(units=32, activation='tanh'))  # Third Dense layer
model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(f'Class weights: {class_weight_dict}')

# Train the model with class weights
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)

# Save the model and scaler
model.save('dense_stock_direction_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Display sample of 20 symbols with stock price, lagged price, actual classification, and predicted classification
sample_data = X_test[:20]  # Get first 20 rows of test set
sample_df = pd.DataFrame(sample_data)

sample_df['stockPrice'] = df['stockPrice'].iloc[-len(sample_data):].values
sample_df['stockPrice_lagged'] = df['stockPrice_lagged'].iloc[-len(sample_data):].values
sample_df['actual_classification'] = y_test[:20]
sample_df['predicted_classification'] = y_pred[:20]

print("Sample output for 20 symbols:")
print(sample_df[['stockPrice', 'stockPrice_lagged', 'actual_classification', 'predicted_classification']])

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Display sample of 20 symbols with stock price, lagged price, actual classification, and predicted classification
sample_data = X_test[:20]  # Get first 20 rows of test set
sample_df = pd.DataFrame(sample_data)

sample_df['stockPrice'] = df['stockPrice'].iloc[-len(sample_data):].values
sample_df['stockPrice_lagged'] = df['stockPrice_lagged'].iloc[-len(sample_data):].values
sample_df['actual_classification'] = y_test[:20]
sample_df['predicted_classification'] = y_pred[:20]

print("Sample output for 20 symbols:")
print(sample_df[['stockPrice', 'stockPrice_lagged', 'actual_classification', 'predicted_classification']])
