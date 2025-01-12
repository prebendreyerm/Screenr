import sqlite3
import pandas as pd
from pathlib import Path

def get_db_connection():
    db_path = Path('backend/data/financial_data.db')
    conn = sqlite3.connect(db_path)
    return conn


# Connect to SQLite database (or create it if it doesn't exist)
conn = get_db_connection()
cursor = conn.cursor()

# Create Assets table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    nordnetSymbol TEXT NOT NULL,
    name TEXT NOT NULL,
    sector TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS ScoringStrategies (
    strategy_name TEXT PRIMARY KEY,
    columns_used TEXT,
    ranking_direction TEXT,
    sector TEXT,
    overall_return REAL,
    std_dev REAL,
    positive_return_ratio REAL
)
''')



# Create AnnualRatios table
cursor.execute('''
CREATE TABLE IF NOT EXISTS AnnualRatios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT,
    calendarYear TEXT,
    period TEXT,
    currentRatio REAL,
    quickRatio REAL,
    cashRatio REAL,
    daysOfSalesOutstanding REAL,
    daysOfInventoryOutstanding REAL,
    operatingCycle REAL,
    daysOfPayablesOutstanding REAL,
    cashConversionCycle REAL,
    grossProfitMargin REAL,
    operatingProfitMargin REAL,
    pretaxProfitMargin REAL,
    netProfitMargin REAL,
    effectiveTaxRate REAL,
    returnOnAssets REAL,
    returnOnEquity REAL,
    returnOnCapitalEmployed REAL,
    netIncomePerEBT REAL,
    ebtPerEbit REAL,
    ebitPerRevenue REAL,
    debtRatio REAL,
    debtEquityRatio REAL,
    longTermDebtToCapitalization REAL,
    totalDebtToCapitalization REAL,
    interestCoverage REAL,
    cashFlowToDebtRatio REAL,
    companyEquityMultiplier REAL,
    receivablesTurnover REAL,
    payablesTurnover REAL,
    inventoryTurnover REAL,
    fixedAssetTurnover REAL,
    assetTurnover REAL,
    operatingCashFlowPerShare REAL,
    freeCashFlowPerShare REAL,
    cashPerShare REAL,
    payoutRatio REAL,
    operatingCashFlowSalesRatio REAL,
    freeCashFlowOperatingCashFlowRatio REAL,
    cashFlowCoverageRatios REAL,
    shortTermCoverageRatios REAL,
    capitalExpenditureCoverageRatio REAL,
    dividendPaidAndCapexCoverageRatio REAL,
    dividendPayoutRatio REAL,
    priceBookValueRatio REAL,
    priceToBookRatio REAL,
    priceToSalesRatio REAL,
    priceEarningsRatio REAL,
    priceToFreeCashFlowsRatio REAL,
    priceToOperatingCashFlowsRatio REAL,
    priceCashFlowRatio REAL,
    priceEarningsToGrowthRatio REAL,
    priceSalesRatio REAL,
    dividendYield REAL,
    enterpriseValueMultiple REAL,
    priceFairValue REAL
)
''')

# Create QuarterlyRatios table
cursor.execute('''
CREATE TABLE IF NOT EXISTS QuarterlyRatios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT,
    calendarYear TEXT,
    period TEXT,
    currentRatio REAL,
    quickRatio REAL,
    cashRatio REAL,
    daysOfSalesOutstanding REAL,
    daysOfInventoryOutstanding REAL,
    operatingCycle REAL,
    daysOfPayablesOutstanding REAL,
    cashConversionCycle REAL,
    grossProfitMargin REAL,
    operatingProfitMargin REAL,
    pretaxProfitMargin REAL,
    netProfitMargin REAL,
    effectiveTaxRate REAL,
    returnOnAssets REAL,
    returnOnEquity REAL,
    returnOnCapitalEmployed REAL,
    netIncomePerEBT REAL,
    ebtPerEbit REAL,
    ebitPerRevenue REAL,
    debtRatio REAL,
    debtEquityRatio REAL,
    longTermDebtToCapitalization REAL,
    totalDebtToCapitalization REAL,
    interestCoverage REAL,
    cashFlowToDebtRatio REAL,
    companyEquityMultiplier REAL,
    receivablesTurnover REAL,
    payablesTurnover REAL,
    inventoryTurnover REAL,
    fixedAssetTurnover REAL,
    assetTurnover REAL,
    operatingCashFlowPerShare REAL,
    freeCashFlowPerShare REAL,
    cashPerShare REAL,
    payoutRatio REAL,
    operatingCashFlowSalesRatio REAL,
    freeCashFlowOperatingCashFlowRatio REAL,
    cashFlowCoverageRatios REAL,
    shortTermCoverageRatios REAL,
    capitalExpenditureCoverageRatio REAL,
    dividendPaidAndCapexCoverageRatio REAL,
    dividendPayoutRatio REAL,
    priceBookValueRatio REAL,
    priceToBookRatio REAL,
    priceToSalesRatio REAL,
    priceEarningsRatio REAL,
    priceToFreeCashFlowsRatio REAL,
    priceToOperatingCashFlowsRatio REAL,
    priceCashFlowRatio REAL,
    priceEarningsToGrowthRatio REAL,
    priceSalesRatio REAL,
    dividendYield REAL,
    enterpriseValueMultiple REAL,
    priceFairValue REAL
)
''')

# Create KeyMetricsTTM table
cursor.execute('''
CREATE TABLE IF NOT EXISTS KeyMetricsTTM (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    revenuePerShareTTM REAL,
    netIncomePerShareTTM REAL,
    operatingCashFlowPerShareTTM REAL,
    freeCashFlowPerShareTTM REAL,
    cashPerShareTTM REAL,
    bookValuePerShareTTM REAL,
    tangibleBookValuePerShareTTM REAL,
    shareholdersEquityPerShareTTM REAL,
    interestDebtPerShareTTM REAL,
    marketCapTTM REAL,
    enterpriseValueTTM REAL,
    peRatioTTM REAL,
    priceToSalesRatioTTM REAL,
    pocfratioTTM REAL,
    pfcfRatioTTM REAL,
    pbRatioTTM REAL,
    ptbRatioTTM REAL,
    evToSalesTTM REAL,
    enterpriseValueOverEBITDATTM REAL,
    evToOperatingCashFlowTTM REAL,
    evToFreeCashFlowTTM REAL,
    earningsYieldTTM REAL,
    freeCashFlowYieldTTM REAL,
    debtToEquityTTM REAL,
    debtToAssetsTTM REAL,
    netDebtToEBITDATTM REAL,
    currentRatioTTM REAL,
    interestCoverageTTM REAL,
    incomeQualityTTM REAL,
    dividendYieldTTM REAL,
    dividendYieldPercentageTTM REAL,
    payoutRatioTTM REAL,
    salesGeneralAndAdministrativeToRevenueTTM REAL,
    researchAndDevelopementToRevenueTTM REAL,
    intangiblesToTotalAssetsTTM REAL,
    capexToOperatingCashFlowTTM REAL,
    capexToRevenueTTM REAL,
    capexToDepreciationTTM REAL,
    stockBasedCompensationToRevenueTTM REAL,
    grahamNumberTTM REAL,
    roicTTM REAL,
    returnOnTangibleAssetsTTM REAL,
    grahamNetNetTTM REAL,
    workingCapitalTTM REAL,
    tangibleAssetValueTTM REAL,
    netCurrentAssetValueTTM REAL,
    investedCapitalTTM REAL,
    averageReceivablesTTM REAL,
    averagePayablesTTM REAL,
    averageInventoryTTM REAL,
    daysSalesOutstandingTTM REAL,
    daysPayablesOutstandingTTM REAL,
    daysOfInventoryOnHandTTM REAL,
    receivablesTurnoverTTM REAL,
    payablesTurnoverTTM REAL,
    inventoryTurnoverTTM REAL,
    roeTTM REAL,
    capexPerShareTTM REAL,
    dividendPerShareTTM REAL,
    debtToMarketCapTTM REAL
)
''')



# Commit changes and close the connection
conn.commit()
conn.close()
