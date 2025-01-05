import sqlite3
import pandas as pd

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(r'backend\data\financial_data.db')
cursor = conn.cursor()

# Clear the Transactions table
cursor.execute('DELETE FROM Transactions')
cursor.execute('DELETE FROM Holdings')

# Commit the changes
conn.commit()

# Optionally, fetch and print the contents of the Transactions table to verify it's empty
port = pd.read_sql('SELECT * FROM Transactions', conn)
print(port)

# Close the connection
conn.close()
