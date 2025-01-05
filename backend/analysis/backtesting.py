import sqlite3
import pandas as pd
from pathlib import Path

pd.set_option('display.max_rows', 500)

# Set up database connection
db_path = Path('backend/data/financial_data.db')
conn = sqlite3.connect(db_path)

# Function to fetch data as DataFrame
def fetch_data_as_dataframe(query):
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df
    except sqlite3.Error as e:
        print(f"An error occurred while fetching data: {e}")
        return None

# Drop rows with NaN values
def clean_data(df):
    df = df.dropna()
    return df

# Ensure the DataFrame is sorted by ticker and year
def sort_data(df):
    df['calendarYear'] = df['calendarYear'].astype(int)  # Convert calendarYear to integers
    df = df.sort_values(by=['calendarYear', 'symbol'])
    return df

# Combine multiple database tables into a single DataFrame
def combined_df():
    # Fetch data from tables
    df_assets = fetch_data_as_dataframe('SELECT * FROM Assets')
    df_ratios = fetch_data_as_dataframe('SELECT * FROM RatiosAnnual')
    df_keyMetrics = fetch_data_as_dataframe('SELECT * FROM KeyMetricsAnnual')
    df_financial = fetch_data_as_dataframe('SELECT * FROM FinancialGrowthAnnual')
    df_prices = fetch_data_as_dataframe('SELECT * FROM HistoricalPricesAnnual')

    # Merge the data on common columns (e.g., 'symbol', 'date')
    df = pd.merge(df_ratios, df_keyMetrics, on=['symbol', 'date'], how='inner')
    df = pd.merge(df, df_financial, on=['symbol', 'date'], how='inner')
    df = pd.merge(df, df_prices, on=['symbol', 'date'], how='inner')
    df = pd.merge(df, df_assets[['symbol', 'name', 'industry', 'sector']], on='symbol', how='left')

    # Remove columns with underscores in their names
    df = df.loc[:, ~df.columns.str.contains('_')]
    return df

# Function to get the scoring strategy from the database for a given sector
def get_scoring_strategy(sector):
    with sqlite3.connect(db_path) as conn:
        strategy = pd.read_sql_query(
            "SELECT * FROM ScoringStrategies WHERE strategy_name = ?",
            conn,
            params=(sector,)
        )
        return strategy


def return_to_multiplier(percentage):
    return (100 + percentage) / 100


# Function to calculate yearly returns using the scoring strategy
def calculate_yearly_returns(df, scoring_columns, ascending_flags):
    df = df.copy()
    # Rank values for each scoring column
    ranks = [df[col].rank(pct=True, ascending=asc) for col, asc in zip(scoring_columns, ascending_flags)]
    df['score'] = sum(ranks) / len(scoring_columns)

    yearly_returns = {}
    years = df['calendarYear'].unique()

    for year in sorted(years)[:-1]:
        current_year_data = df[df['calendarYear'] == year]
        if current_year_data.empty:
            continue
        next_year_data = df[df['calendarYear'] == year + 1]

        total_initial_investment = current_year_data['stockPrice'].sum()
        total_return = 0
        for index, row in current_year_data.iterrows():
            stock_symbol = row['symbol']
            buy_price = row['stockPrice']
            next_year_stock_data = next_year_data[next_year_data['symbol'] == stock_symbol]
            if next_year_stock_data.empty:
                continue

            sell_price = next_year_stock_data['stockPrice'].values[0]
            return_for_stock = (sell_price - buy_price) / buy_price
            total_return += return_for_stock

        sector_yearly_return = (total_return / total_initial_investment) * 100

        yearly_returns[year] = sector_yearly_return

    return yearly_returns

# Combine the data, clean and sort it
df = combined_df()
df = clean_data(df)
df = sort_data(df)

# List of sectors to loop through
sectors = ['Industrials', 'Communication Services', 'Consumer Defensive', 'Healthcare',
           'Consumer Cyclical', 'Technology', 'Basic Materials', 'Energy',
           'Financial Services', 'Real Estate', 'Utilities']

# Initialize an empty list to store dictionaries for each year
returns_data = []

# Loop over each sector and calculate returns using the scoring strategy from the database
for sector in sectors:
    print(f"Calculating returns for sector: {sector}")
    df_sector = df[df['sector'] == sector]
    
    # Get the scoring strategy from the database
    strategy = get_scoring_strategy(sector)
    if strategy.empty:
        print(f"No scoring strategy found for sector: {sector}, skipping...")
        continue
    
    # Extract columns used and ranking direction
    scoring_columns = strategy['columns_used'].iloc[0].split(',')
    ascending_flags = [flag == 'True' for flag in strategy['ranking_direction'].iloc[0].split(',')]

    # Calculate the yearly returns for this sector
    sector_yearly_returns = calculate_yearly_returns(
        df_sector, scoring_columns, ascending_flags
    )

    # For each year, add the sector return to the dictionary for the row
    for year in sorted(sector_yearly_returns.keys()):
        year_data = next((item for item in returns_data if item['Year'] == year), None)
        if year_data is None:
            year_data = {'Year': year}
            returns_data.append(year_data)

        # Add the sector return for the year
        year_data[sector] = sector_yearly_returns[year]

# Convert the list of dictionaries into a DataFrame
returns_df = pd.DataFrame(returns_data)

# Calculate the total return for each year across all sectors as multipliers
returns_df['Total Multiplier'] = returns_df[sectors].map(return_to_multiplier).prod(axis=1, skipna=True)

# Initialize portfolio value with $100 at the start
initial_portfolio_value = 100
portfolio_value = initial_portfolio_value

# Simulate the portfolio value by applying the total multiplier for each year
returns_df['Portfolio Value'] = None
for i, row in returns_df.iterrows():
    portfolio_value *= row['Total Multiplier']
    returns_df.at[i, 'Portfolio Value'] = portfolio_value

# Fetch the baseline strategy from the database
baseline_strategy = get_scoring_strategy('Baseline')
if baseline_strategy.empty:
    print("No baseline strategy found, skipping baseline comparison...")
else:
    # Extract columns used and ranking direction for the baseline strategy
    baseline_columns = baseline_strategy['columns_used'].iloc[0].split(',')
    baseline_ascending_flags = [flag == 'True' for flag in baseline_strategy['ranking_direction'].iloc[0].split(',')]

    # Backtest the baseline strategy using the same approach as sector-specific strategies
    baseline_yearly_returns = calculate_yearly_returns(
        df, baseline_columns, baseline_ascending_flags
    )

    # Calculate the total return for each year across all sectors as multipliers for baseline
    baseline_returns_df = pd.DataFrame(baseline_yearly_returns.items(), columns=['Year', 'Baseline Return'])
    baseline_returns_df['Baseline Multiplier'] = baseline_returns_df['Baseline Return'].apply(return_to_multiplier)

    # Simulate the baseline portfolio value by applying the baseline multiplier for each year
    baseline_portfolio_value = initial_portfolio_value
    baseline_portfolio_values = []
    for multiplier in baseline_returns_df['Baseline Multiplier']:
        baseline_portfolio_value *= multiplier
        baseline_portfolio_values.append(baseline_portfolio_value)

    # Add the baseline portfolio values to the returns DataFrame for comparison
    returns_df['Baseline Portfolio Value'] = baseline_portfolio_values

# Print the resulting DataFrame with Portfolio Value and Baseline Portfolio Value columns
print(returns_df[['Year', 'Portfolio Value', 'Baseline Portfolio Value']])

# Function to calculate yearly top-ranked stock and its returns
def calculate_top_ranked_stocks(df, scoring_columns, ascending_flags):
    top_stocks = []
    df = df.copy()
    years = df['calendarYear'].unique()

    for year in sorted(years)[:-1]:
        current_year_data = df[df['calendarYear'] == year]
        next_year_data = df[df['calendarYear'] == year + 1]
        if current_year_data.empty or next_year_data.empty:
            continue
        
        # Rank values for each scoring column and calculate the score
        ranks = [current_year_data[col].rank(pct=True, ascending=asc) for col, asc in zip(scoring_columns, ascending_flags)]
        current_year_data['score'] = sum(ranks) / len(scoring_columns)
        
        # Get the top-ranked stock
        top_stock = current_year_data.nlargest(1, 'score')
        top_stock_symbol = top_stock['symbol'].values[0]
        buy_price = top_stock['stockPrice'].values[0]
        
        # Get the sell price from the next year's data
        next_year_stock_data = next_year_data[next_year_data['symbol'] == top_stock_symbol]
        if next_year_stock_data.empty:
            continue
        sell_price = next_year_stock_data['stockPrice'].values[0]
        
        # Add the top stock data to the list
        top_stocks.append({
            'Year': year,
            'Sector': top_stock['sector'].values[0],
            'Symbol': top_stock_symbol,
            'Buy Price': buy_price,
            'Sell Price': sell_price
        })

    return pd.DataFrame(top_stocks)

# Calculate top-ranked stocks for each sector
all_top_stocks = pd.DataFrame()
for sector in sectors:
    df_sector = df[df['sector'] == sector]
    strategy = get_scoring_strategy(sector)
    if strategy.empty:
        print(f"No scoring strategy found for sector: {sector}, skipping...")
        continue
    
    scoring_columns = strategy['columns_used'].iloc[0].split(',')
    ascending_flags = [flag == 'True' for flag in strategy['ranking_direction'].iloc[0].split(',')]
    top_stocks_sector = calculate_top_ranked_stocks(df_sector, scoring_columns, ascending_flags)
    all_top_stocks = pd.concat([all_top_stocks, top_stocks_sector])

# Calculate top-ranked stock for the baseline strategy
baseline_strategy = get_scoring_strategy('Baseline')
if not baseline_strategy.empty:
    baseline_columns = baseline_strategy['columns_used'].iloc[0].split(',')
    baseline_ascending_flags = [flag == 'True' for flag in baseline_strategy['ranking_direction'].iloc[0].split(',')]
    baseline_top_stocks = calculate_top_ranked_stocks(df, baseline_columns, baseline_ascending_flags)
    baseline_top_stocks['Sector'] = 'All'  # Indicate that this is the baseline for all sectors
else:
    print("No baseline strategy found, skipping baseline top stock calculation...")

# Print the top-ranked stocks for each sector and for the baseline
print("Top-ranked stocks for each sector each year:")
print(all_top_stocks)
print("\nTop-ranked stock for the baseline each year:")
print(baseline_top_stocks)

# Function to calculate the returns based on top-ranked stocks
def calculate_returns(top_stocks, initial_capital):
    returns = []
    total_value = initial_capital
    for index, row in top_stocks.iterrows():
        buy_price = row['Buy Price']
        sell_price = row['Sell Price']
        sector_investment = initial_capital / len(sectors) if row['Sector'] != 'All' else initial_capital
        shares_bought = sector_investment / buy_price
        sector_return = shares_bought * sell_price
        total_value += sector_return - sector_investment
        returns.append({
            'Year': row['Year'],
            'Sector': row['Sector'],
            'Symbol': row['Symbol'],
            'Buy Price': buy_price,
            'Sell Price': sell_price,
            'Return': sector_return - sector_investment,
            'Total Value': total_value
        })
    return pd.DataFrame(returns)

# Calculate returns for the non-baseline case
non_baseline_initial_capital = 100
non_baseline_returns = calculate_returns(all_top_stocks, non_baseline_initial_capital)

# Calculate returns for the baseline case
baseline_initial_capital = 100
baseline_returns = calculate_returns(baseline_top_stocks, baseline_initial_capital)

# Print the returns for each sector and for the baseline
print("Returns for each sector each year (non-baseline):")
print(non_baseline_returns)
print("\nReturns for the baseline each year:")
print(baseline_returns)