import os
from lib.IVVEnvironment import IVVEnvironment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import choice
import random
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

import torch


device = "cpu" if not torch.cuda.is_available() else 'cuda'

#Disable the warnings
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def plot_reward(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, linestyle='-')
    plt.title('Training - Reward evolution')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.savefig('reward.png')
    plt.show()

def plot_validation(profit, net_profit, trades):
    profit_mean = np.mean(profit)
    trades_mean = np.mean(trades)

    # Profit
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(profit) + 1), profit, label='Profit', linestyle='-')
    plt.plot(range(1, len(net_profit) + 1), net_profit, label='Net Profit', linestyle='-')
    plt.axhline(profit_mean, color='r', linestyle='--', label=f'Mean: {profit_mean:.2f}')
    plt.title(f'Profit and Net Profit\nAnnual mean: {profit_mean}')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('profit.png')
    plt.show()

    # Trades
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(trades) + 1), trades)
    plt.axhline(trades_mean, color='r', linestyle='--', label=f'Mean: {trades_mean:.2f}')
    plt.title(f'Number of trades per day\nAnnual mean: {trades_mean}')
    plt.xlabel('Episodes')
    plt.ylabel('Number of trades')
    plt.grid(True)
    plt.legend()
    plt.savefig('trades.png')
    plt.show()

def plot_best(episode, series, profit, buy, sell):
    fig = plt.figure(figsize = (15,5))
    plt.plot(series, color='r', lw=2.)
    plt.plot(series, 'o', markersize=10, color='m', label = 'Buying signal', markevery = buy)
    plt.plot(series, 'o', markersize=10, color='k', label = 'Selling signal', markevery = sell)
    plt.title('Total gains: %f'%(profit))
    plt.legend()
    plt.savefig('lib/'+str(episode)+'.png')

def data_info(filepath):
    data = pd.read_csv(filepath, sep=',', parse_dates=['DateTime'], index_col='DateTime')
    
    print('Train data observations: ', data.shape)
    print('Columns: ', data.columns.values)
    print('---------------')
    data.info()
    print('---------------')
    data.describe()

    print('Correlation matrix')
    corr_matrix = data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()

    return data

def data_visualization(data):
    print('---------------')
    print('Dataset visualization')
    sns.set_theme(style="whitegrid") 
    data['Date'] = data.index.date
 
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.plot(data['Date'], data['Open'], label='Open Price')
    plt.plot(data['Date'], data['High'], label='High Price')
    plt.plot(data['Date'], data['Low'], label='Low Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Over Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Volume'], label='Volume', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Volume Over Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data['Close'], kde=True)
    plt.title('Distribution of Closing Stock Price')
    plt.xlabel('Closing Stock Price')
    plt.ylabel('Frequency')
    plt.show()

    return data

def check_stationarity(series):
    print("Stationarity analysis: Augmented Dickey-Fuller test (ADF):")
    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print(str(key) + ': ' + str(value))
    if result[1] <= 0.05:
        print("The ADF test affirms that the time series is stationary.")
    else:
        print("The ADF test could not affirm whether or not the time series is stationary...")
 
def autocorrelation(sample):    
    # autocorrelation and partial autocorrelation plots
    _, axes = plt.subplots(2, figsize=(16, 9))
    plot_acf(sample, lags=21, ax=axes[0])
    plot_pacf(sample, lags=21, ax=axes[1])
    plt.show()

def statistic_analysis(data):
    # Outliers
    print("Outliers")
    z_scores = np.abs((data['Close'] - data['Close'].mean()) / data['Close'].std())
    outliers = data[z_scores > 3]
    print(outliers)

    # Decompose the stock prices into trend, seasonality, and noise components
    result = seasonal_decompose(data['Close'], model='additive', period=1, extrapolate_trend='freq')
    result.plot()
    plt.show()