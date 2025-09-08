import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ---------------------
# Load data
# ---------------------
conn = sqlite3.connect('backend/data/financial_data.db')

df = pd.read_sql_query('SELECT * FROM keyMetricsAnnual', conn)
df2 = pd.read_sql_query('SELECT * FROM historicalPricesAnnual', conn)
df = df.merge(df2, on=['symbol', 'date'], how='inner')

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol','date'])

# ---------------------
# Compute next-year change
# ---------------------
df['change'] = df.groupby('symbol')['stockPrice'].transform(lambda x: x.shift(-1) / x - 1)
df = df.dropna()  # drop rows where change cannot be computed
df = df[df['change'] > 0]

test = np.array(df['enterpriseValueOverEBITDA'])


# print(df['enterpriseValueOverEBITDA'].describe())
# print(df['pfcfRatio'].describe())


# ---------------------
# Select numeric columns (exclude identifiers)
# ---------------------
exclude_cols = ['symbol', 'date', 'calendarYear', 'period', 'debtToEquity', 'debtToAssets', 'netDebtToEBITDA', 'currentRatio',
       'interestCoverage', 'incomeQuality', 'dividendYield', 'payoutRatio',
       'salesGeneralAndAdministrativeToRevenue',
       'researchAndDdevelopementToRevenue', 'intangiblesToTotalAssets',
       'capexToOperatingCashFlow', 'capexToRevenue', 'capexToDepreciation',
       'stockBasedCompensationToRevenue', 'grahamNumber', 'roic',
       'returnOnTangibleAssets', 'grahamNetNet', 'workingCapital',
       'tangibleAssetValue', 'netCurrentAssetValue', 'investedCapital',
       'averageReceivables', 'averagePayables', 'averageInventory',
       'daysSalesOutstanding', 'daysPayablesOutstanding',
       'daysOfInventoryOnHand', 'receivablesTurnover', 'payablesTurnover',
       'inventoryTurnover', 'roe', 'capexPerShare', 'stockPrice',
       'numberOfShares', 'marketCapitalization', 'minusCashAndCashEquivalents',
       'addTotalDebt', 'enterpriseValue_y']
numeric_cols = df.select_dtypes(include='number').columns.difference(exclude_cols)

# ---------------------
# Plot all numeric distributions
# ---------------------
n_cols = 5  # number of plots per row
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  # compute required rows
plt.figure(figsize=(n_cols*5, n_rows*4))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    
    # Clip extreme outliers (1st and 99th percentile)
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    filtered = df[col].clip(lower, upper)
    
    # Check skewness: if highly skewed, use symlog x-scale
    skewness = filtered.skew()
    if abs(skewness) > 10:  # threshold for extreme skew
        sns.histplot(filtered + 1e-6, bins=50, kde=True, color='skyblue')
        plt.xscale('symlog')
    else:
        sns.histplot(filtered, bins=50, kde=True, color='skyblue')
    
    plt.title(col)
    plt.tight_layout()

plt.show()

