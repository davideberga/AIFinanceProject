import numpy as np
import math
from typing import List
import pandas as pd


# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		close = line.split(",")[4]
		if close != 'null':
			vec.append(float(line.split(",")[5]))

	return vec

def readData(file_csv):
	dataset = pd.read_csv(file_csv, index_col=0, parse_dates=[0], header=0)
	dataset.isnull().values.any()
	dataset=dataset.fillna(method='ffill')
	
	grouped_by_day = dataset.groupby(pd.Grouper(freq='D'))

	days = []
	for _, day in grouped_by_day:
		if(len(day) > 0):
			day_preprocessed = day.drop(columns=['High', 'Open', 'Low', 'Volume'])
			days.append(day_preprocessed['Close'].tolist())
	return days

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])
