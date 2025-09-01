# Personal Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import yfinance as yf
from scipy.optimize import minimize


# ----------------------
# PRE-PROCESSING DATA 
# ----------------------


# Get Data From Yahoo
# https://tradingeconomics.com/zambia/government-bond-yield

tickers = [
    "ZMW=X",
    "HG=F", # Copper futures (reseach what futures are)
    "^GSPC", # S&P 500
    "GC=F"   # Gold futures 
]

#  Download 6 years of daily data till date

data = yf.download(tickers, start="2015-01-01", end="2025-08-01")["Close"]

# Clean the data , remove null and missing data

data = data.dropna()
print(data.head())

# ----------------------
# PROCESSING THE DATA 
# ----------------------

# Calculating the returns for each ticker 

returns = data.pct_change().dropna() # daily percentage returns 
mean_returns = returns.mean() * 252 # annulised average returns 
covariance_matrix = returns.cov() * 252 # annulised covariance between tickers 

print(f"Mean Returns: {mean_returns}")
print(f"Covariance Matrix: {covariance_matrix}")

# --------------------
# Monte Carlo Simulation
# --------------------

number_of_portfolios = 10000
results_matrix = np.zeros((3, number_of_portfolios)) # return, volatility, Sharpe ratio 
weights_array = []
risk_free_rate = .1952

for i in range(number_of_portfolios): 
    weights = np.random.random(len(tickers))
    weights /= sum(weights)
    weights_array.append(weights)

    portfolio_returns = np.dot(weights,mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    #The yield on Zambia Government Bond 10y held steady at 19.52% on August 22, 2025.
    sharpe_ratio = (portfolio_returns - risk_free_rate)/portfolio_volatility

    results_matrix[0,i] = portfolio_returns
    results_matrix[1,i] = portfolio_volatility
    results_matrix[2, i] = sharpe_ratio


# ------------------------
# GRAPHICAL REPRESENTATION
# ------------------------

plt.figure(figsize=(10,6))
plt.scatter(x=results_matrix[1,:],y=results_matrix[0,:], c=results_matrix[2,:], cmap='viridis', s=10)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Monte Carlo Porfolio Optimisation - Focused On Zambia")
max_sharpe_index = np.argmax(results_matrix[2])
plt.scatter(results_matrix[1,max_sharpe_index], results_matrix[0,max_sharpe_index], c="red", s=50, marker='*', label= "Max Sharpe")
plt.legend()
plt.show()

# ------------------------
# MOST FAVORABLE WEIGHTS
# ------------------------

weights_df = pd.DataFrame(weights_array, columns=tickers)
best_weights= weights_df.iloc[max_sharpe_index]
print("Best Portfolio Weights:")
print(best_weights)

# --------------
# OPTIMISATION 
# --------------

# Functions 

def portfolio_performance(weights, mean_return, covariance_matrix, risk_free_rate): 
    op_portfolio_returns = np.dot(weights, mean_return)
    op_portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    op_sharpe_ratio = (op_portfolio_returns - risk_free_rate) / op_portfolio_volatility

    return op_portfolio_returns, op_portfolio_volatility, op_sharpe_ratio

def negative_sharpe_ratio(weights, mean_return, covariance_matrix, risk_free_rate): 
    return -portfolio_performance(weights, mean_return, covariance_matrix, risk_free_rate)[2]

def portfolio_volatility(weights, mean_return, covariance_matrix, risk_free_rate): 
    return portfolio_performance(weights, mean_return, covariance_matrix, risk_free_rate)[1]

# weights must sum to 1 
constraints = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})

# weights must be between 0 to 1 
bounds = tuple((0,1) for _ in range(len(tickers)))
initial_guess = len(tickers) * [1./len(tickers)]

# --------------------
# Optimize for Max Sharpe Ratio
# --------------------

max_shape_ratio = minimize(negative_sharpe_ratio, initial_guess, 
                           args=(mean_returns, covariance_matrix, risk_free_rate),
                            method='SLSQP', bounds=bounds, constraints=constraints)

op_sharpe_return, op_sharpe_volalitlity, op_sharpe_ratio = portfolio_performance(max_shape_ratio.x, mean_returns, covariance_matrix, risk_free_rate)

print("Max Sharpe Ratio Portfolio:")
print("Weights:\n", dict(zip(tickers, max_shape_ratio.x)))
print("Return:\n", op_sharpe_return)
print("Volatility:\n", op_sharpe_volalitlity)
print("Sharpe Ratio:\n", op_sharpe_ratio)
