import sqlite3
import pandas as pd
import os

# Paths
db_path = r'backend\data\financial_data.db'
baselines_dir = r'backend\data\Baselines'

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all files in the Baselines directory
baseline_files = [f for f in os.listdir(baselines_dir) if f.startswith("baseline") and f.endswith(".csv")]

# Function to prepare a strategy for insertion
def prepare_strategy(columns, baseline_df):
    strategy = {col: None for col in columns}  # Default all to None
    for _, row in baseline_df.iterrows():
        scoring_column = row['scoring_columns']
        boolean_value = row['booleans']
        if scoring_column in strategy:
            strategy[scoring_column] = boolean_value
    return strategy

# Get all columns in the ScoringStrategies table
cursor.execute("PRAGMA table_info(ScoringStrategies)")
columns = [row[1] for row in cursor.fetchall() if row[1] not in ('id', 'sector')]

# Process each baseline file
for file in baseline_files:
    # Extract sector name from the filename
    if file == "baseline.csv":
        sector = "General"  # Default for baseline.csv
    else:
        sector = file.replace("baseline_", "").replace(".csv", "").strip()

    # Read the baseline file
    file_path = os.path.join(baselines_dir, file)
    baseline_df = pd.read_csv(file_path)

    # Prepare the strategy
    strategy = prepare_strategy(columns, baseline_df)

    # Insert the strategy into the database
    placeholders = ", ".join(["?" for _ in strategy])
    columns_part = ", ".join(strategy.keys())
    insert_query = f"INSERT INTO ScoringStrategies (sector, {columns_part}) VALUES (?, {placeholders})"

    values = [sector] + list(strategy.values())
    cursor.execute(insert_query, values)

# Commit all changes and close the connection
conn.commit()
conn.close()

print("All baseline strategies inserted successfully!")
