import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import numpy as np
import random

# Function to remove outliers based on IQR, ensuring only numerical columns are included
def remove_outliers(df, columns):
    numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Step 1: Load data from the SQLite database
conn = sqlite3.connect(r'backend\data\financial_data.db')
df_Ratios = pd.read_sql_query("SELECT * FROM RatiosQuarter", conn)
df_prices = pd.read_sql_query("SELECT * FROM HistoricalPricesQuarter", conn)
conn.close()  # Close the connection when done

# Step 2: Clean the data (drop NaNs and filter out irrelevant columns)
on_columns = ['symbol', 'date', 'period', 'calendarYear']  # Specify the columns to drop
df_Ratios_filtered = df_Ratios.drop(columns=on_columns)  # Drop symbol, date, period, and calendarYear for features

# Merge the Ratios and Prices DataFrames
final_df = pd.merge(df_Ratios, df_prices[['symbol', 'date', 'stockPrice']], on=['symbol', 'date'], how='inner')

# Step 3: Calculate the change in stock price for each symbol
final_df['price_change'] = final_df.groupby('symbol')['stockPrice'].diff()

# Step 4: Create the binary target variable (1 for price increase, 0 for decrease)
final_df['price_increase'] = (final_df['price_change'] > 0).astype(int)

# Drop rows with NaN values that result from the price change calculation
final_df = final_df.dropna(subset=['price_increase'])

# Apply the outlier removal function on the filtered DataFrame
final_df = remove_outliers(final_df, df_Ratios_filtered.columns)

print(final_df.describe())
print(final_df)
# # Step 5: Select features and target variable
# X = final_df[df_Ratios_filtered.columns]  # Use all remaining columns (ratios) as features
# y = final_df['price_increase']  # Target: 1 for increase, 0 for decrease
# print(X.describe())
# print(X)

# # # Step 6: Normalize the features (modify the line)
# # min_max_scaler = MinMaxScaler()
# # X_scaled = X.copy()  # Create a copy of X to avoid modifying the original
# # X_scaled[df_Ratios_filtered.columns] = min_max_scaler.fit_transform(X_scaled[df_Ratios_filtered.columns])

# # # Step 7: Split dataset into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # # Function to train and evaluate the model
# # def train_model_with_columns(selected_columns):
# #     # Ensure the selected columns exist in the DataFrame
# #     X_train_subset = X_train[selected_columns]
# #     X_test_subset = X_test[selected_columns]

# #     # Define the classification model
# #     model = Sequential([
# #         Input(shape=(X_train_subset.shape[1],)),  # Define input shape correctly
# #         Dense(64, activation='relu'),
# #         Dense(32, activation='relu'),
# #         Dense(16, activation='relu'),
# #         Dense(1, activation='sigmoid')
# #     ])

# #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# #     # Train the model
# #     history = model.fit(X_train_subset, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

# #     # Evaluate the model
# #     _, accuracy = model.evaluate(X_test_subset, y_test, verbose=0)

# #     # Return the accuracy and the selected columns
# #     return accuracy, selected_columns

# # # Loop to select random column subsets and train models
# # best_accuracy = 0
# # best_columns = []
# # for i in range(100):  # Run the experiment 100 times with random subsets
# #     # Randomly select a subset of columns
# #     selected_columns = random.sample(list(df_Ratios_filtered.columns), k=random.randint(5, len(df_Ratios_filtered.columns)))
    
# #     # Train and evaluate the model on this subset of columns
# #     accuracy, selected_cols = train_model_with_columns(selected_columns)

# #     # Print progress and stop if accuracy reaches 0.8
# #     print(f"Iteration {i+1}: Accuracy = {accuracy:.4f} with columns {selected_cols}")

# #     if accuracy >= 0.8:
# #         print(f"Early stopping: Found promising columns {selected_cols} with accuracy {accuracy:.4f}")
# #         best_accuracy = accuracy
# #         best_columns = selected_cols
# #         break  # Stop if we achieve the target accuracy

# # if best_accuracy >= 0.8:
# #     print(f"\nBest columns: {best_columns} with accuracy {best_accuracy:.4f}")
# # else:
# #     print(f"\nNo columns achieved 0.8 or higher accuracy. Best accuracy achieved: {best_accuracy:.4f}")
