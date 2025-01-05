import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

db_path = Path('backend/data/financial_data.db')
conn = sqlite3.connect(db_path)

# Function to fetch data as DataFrame
def fetch_data_as_dataframe(query):
    try:
        with sqlite3.connect(r'backend\data\financial_data.db') as conn:
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
# Ensure the DataFrame is sorted by ticker and year
def sort_data(df):
    df['calendarYear'] = df['calendarYear'].astype(int)  # Convert calendarYear to integers
    df = df.sort_values(by=['calendarYear', 'symbol'])
    return df


# Combine multiple database tables into a single DataFrame
def combined_df():
    df_assets = pd.read_sql_query('SELECT * FROM Assets', conn)
    df_ratios = pd.read_sql_query('SELECT * FROM RatiosAnnual', conn)
    df_keyMetrics = pd.read_sql_query('SELECT * FROM KeyMetricsAnnual', conn)
    df_financial = pd.read_sql_query('SELECT * FROM FinancialGrowthAnnual', conn)
    df_prices = pd.read_sql_query('SELECT * FROM HistoricalPricesAnnual', conn)

    df = pd.merge(df_ratios, df_keyMetrics, on=['symbol', 'date'], how='inner')
    df = pd.merge(df, df_financial, on=['symbol', 'date'], how='inner')
    df = pd.merge(df, df_prices, on=['symbol', 'date'], how='inner')
    df = pd.merge(df, df_assets[['symbol', 'name', 'industry', 'sector']], on='symbol', how='left')

    df = df.loc[:, ~df.columns.str.contains('_')]
    return df

sectors =['Industrials', 'Communication Services', 'Consumer Defensive', 'Healthcare',
 'Consumer Cyclical', 'Technology', 'Basic Materials', 'Energy',
 'Financial Services', 'Real Estate', 'Utilities']
sector = 'Industrials'
df = combined_df()
df = clean_data(df)
df = sort_data(df)
if sector != 'Baseline':
    df = df[df['sector'] == sector]

# Select numerical columns for scoring
scoring_columns = df.select_dtypes(include=['number']).columns.tolist()

# Function to plot yearly returns
def plot_returns(yearly_returns, baseline_returns):
    years = list(yearly_returns.keys())
    returns = list(yearly_returns.values())

    # Set the colors for the bars based on whether the return is positive or negative
    bar_colors = ['g' if return_value >= 0 else 'r' for return_value in returns]

    plt.figure(figsize=(12, 6))
    plt.bar(years, returns, color=bar_colors, label='Yearly Returns')
    plt.title('Yearly Returns vs Baseline Strategy')
    plt.xlabel('Year')
    plt.ylabel('Return (%)')
    plt.grid(True)
    plt.xticks(years)
    plt.legend()
    plt.show()


# Function to calculate scores and perform backtesting
def calculate_scores_and_backtest(df, scoring_columns, ascending_flags):
    df = df.copy()
    ranks = [df[col].rank(pct=True, ascending=asc) for col, asc in zip(scoring_columns, ascending_flags)]
    df['score'] = sum(ranks) / len(scoring_columns)

    initial_capital = 100
    total_value = initial_capital
    investment_strategy = []
    years = df['calendarYear'].unique()
    positive_return_years = 0

    for year in sorted(years)[:-1]:
        current_year_data = df[df['calendarYear'] == year]
        if current_year_data.empty:
            continue
        next_year_data = df[df['calendarYear'] == year + 1]

        for _, best_stock in current_year_data.sort_values(by='score').iterrows():
            next_year_stock_data = next_year_data[next_year_data['symbol'] == best_stock['symbol']]
            if next_year_stock_data.empty:
                continue

            buy_price = best_stock['stockPrice']
            shares_bought = total_value / buy_price
            total_value = 0
            
            sell_price = next_year_stock_data['stockPrice'].values[0]
            total_value += shares_bought * sell_price

            gain_pct = (sell_price - buy_price) / buy_price * 100
            investment_strategy.append({
                'year': year,
                'symbol': best_stock['symbol'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'total_value': total_value,
                'gain_pct': gain_pct
            })

            if gain_pct > 0:
                positive_return_years += 1
            break

    overall_return_pct = ((total_value - initial_capital) / initial_capital) * 100
    positive_return_ratio = positive_return_years / len(years)

    # Add positive return ratio as an additional return metric
    return pd.DataFrame(investment_strategy), total_value, overall_return_pct, positive_return_ratio



def compare_with_baseline(strategy_results, baseline_strategy_name, median_return_baseline, baseline_std_dev, sector):
    updated = False
    for strategy_name, (investment_results, total_value, overall_return_pct, positive_return_ratio, strategy_columns) in strategy_results.items():
        # Calculate the median return of the strategy
        median_return = np.median(investment_results['gain_pct'])
        std_dev = np.std(investment_results['gain_pct'])

        # Check if the strategy satisfies all criteria
        if (
            median_return >= median_return_baseline and
            std_dev <= baseline_std_dev and
            positive_return_ratio >= positive_return_ratio_baseline and positive_return_ratio >= 0.5
        ):
            columns_used_str = ','.join(strategy_columns)
            ranking_direction_str = ','.join(map(str, [True] + [False] * (len(strategy_columns) - 1)))

            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ScoringStrategies (strategy_name, columns_used, ranking_direction, sector, median_return, std_dev)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (strategy_name, columns_used_str, ranking_direction_str, sector, median_return, std_dev))

            print(f"Baseline strategy updated to {strategy_name} due to superior median returns, stability, and positive return ratio.")
            updated = True
            break
    return updated

# Load or initialize the baseline strategy
baseline_strategy = conn.execute("""
    SELECT * FROM ScoringStrategies WHERE sector = ?
""", (sector,)).fetchone()

if baseline_strategy:
    # Load existing baseline strategy
    baseline_strategy_name = baseline_strategy[0]
    overall_return_pct_baseline = float(baseline_strategy[4])
    baseline_std_dev = float(baseline_strategy[5])
    positive_return_ratio_baseline = float(baseline_strategy[6])
else:
    # Initialize a new baseline strategy with random default values
    num_columns = random.randint(2, len(scoring_columns))
    selected_columns = random.sample(scoring_columns, num_columns)
    ascending_flags = [random.choice([True, False]) for _ in selected_columns]
    baseline_strategy_name = sector

    investment_results, total_value, overall_return_pct_baseline, positive_return_ratio_baseline = calculate_scores_and_backtest(
        df, selected_columns, ascending_flags
    )
    baseline_std_dev = np.std(investment_results['gain_pct'])

    # Save the new baseline strategy to the database
    with conn:
        conn.execute("""
            INSERT INTO ScoringStrategies (strategy_name, columns_used, ranking_direction, sector, overall_return, std_dev, positive_return_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            baseline_strategy_name,
            ','.join(selected_columns),
            ','.join(map(str, ascending_flags)),
            sector,
            overall_return_pct_baseline,
            baseline_std_dev,
            positive_return_ratio_baseline
        ))

# Run simulations to improve upon the baseline strategy
num_simulations = 1000
tested_strategies = set()

for i in tqdm(range(num_simulations)):
    num_columns = random.randint(2, len(scoring_columns))
    selected_columns = random.sample(scoring_columns, num_columns)
    ascending_flags = [random.choice([True, False]) for _ in selected_columns]

    strategy_identifier = (tuple(selected_columns), tuple(ascending_flags))
    if strategy_identifier in tested_strategies:
        continue

    tested_strategies.add(strategy_identifier)
    investment_results, total_value, overall_return_pct, positive_return_ratio = calculate_scores_and_backtest(
        df, selected_columns, ascending_flags
    )

    # Compare with the current baseline strategy and update if criteria are met
    if (
        overall_return_pct >= overall_return_pct_baseline and
        np.std(investment_results['gain_pct']) <= baseline_std_dev and
        positive_return_ratio >= positive_return_ratio_baseline and positive_return_ratio >= 0.5
    ):
        overall_return_pct_baseline = overall_return_pct
        baseline_std_dev = np.std(investment_results['gain_pct'])
        positive_return_ratio_baseline = positive_return_ratio

        # Update the baseline strategy in the database
        with conn:
            conn.execute("""
                UPDATE ScoringStrategies
                SET columns_used = ?, ranking_direction = ?, overall_return = ?, std_dev = ?, positive_return_ratio = ?
                WHERE sector = ?
            """, (
                ','.join(selected_columns),
                ','.join(map(str, ascending_flags)),
                overall_return_pct_baseline,
                baseline_std_dev,
                positive_return_ratio_baseline,
                sector
            ))

        print(f"Updated Baseline strategy with superior strategy.")
        print(f"New overall return: {overall_return_pct_baseline:.2f}%")
        print(f"New standard deviation: {baseline_std_dev:.2f}")
        print(f"Positive return ratio: {positive_return_ratio_baseline:.2f}")


# Get the yearly returns for the baseline strategy
yearly_returns_baseline = investment_results.groupby('year')['gain_pct'].sum()
# Plot the results
plot_returns(yearly_returns_baseline.to_dict(), yearly_returns_baseline.to_list())
print("Simulation and comparison with baseline completed.")

