import pyfolio as pf
import pandas as pd
import numpy as np

# Definire gli oggetti Timestamp per le due righe
data1 = pd.Timestamp('2007-01-03 00:00:00+00:00')
data2 = pd.Timestamp('2007-01-04 00:00:00+00:00')

# Definire i valori float per le due righe
float_value1 = 0.0345
float_value2 = -0.0345

data = [{'timestamp': data1, 'price': float_value1}, 
        {'timestamp': data2, 'price': float_value2}]  
  
df = pd.DataFrame.from_records(data,index=['timestamp'])
print(df)

stock_rets = pf.utils.get_symbol_rets('FB')
print(stock_rets.head())

idx = pd.date_range("2018-01-01", periods=5, freq="h")
ts = pd.Series(range(len(idx)), index=idx, dtype=np.float64)

print(ts)

print(ts.head())

pf.create_returns_tear_sheet(ts)

# Creare il DataFrame
df = pd.DataFrame({'euro': [float_value1, float_value2]}, index=[data1, data2])

print(df)

pf.create_returns_tear_sheet(df['euro'])
