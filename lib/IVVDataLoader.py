import random
from typing import List
from pandas import read_csv
import pandas as pd
from torch.utils.data import Dataset

class IVVDataset(Dataset):
    def __init__(self, file_csv, colums_to_drop=[]):

        if isinstance(file_csv, list):
            accumulator = []
            for csv in file_csv:
                df = read_csv(csv, index_col=0, parse_dates=[0], header=0)
                accumulator.append(df)
            self.dataset = pd.concat(accumulator, axis=0)
        else:
            self.dataset = read_csv(file_csv, index_col=0, parse_dates=[0], header=0)

        
        self.dataset.isnull().values.any()
        self.dataset=self.dataset.fillna(method='ffill')

        self.open = self.dataset['Open'].astype('float')
        self.close = self.dataset['Close'].astype('float')
        self.high = self.dataset['High'].astype('float')
        self.low = self.dataset['Low'].astype('float')
        self.volume = self.dataset['Volume'].astype('float')

        self.dataset['bar_hc'] = self.high - self.close
        self.dataset['bar_ho'] = self.high - self.open
        self.dataset['bar_hl'] = self.high - self.low
        self.dataset['bar_cl'] = self.close - self.low
        self.dataset['bar_ol'] = self.open - self.low
        self.dataset['bar_co'] = self.close - self.open
        self.dataset['bar_mov'] = self.dataset['Close'] - self.dataset['Close'].shift(1)

        self.dataset['adj_open'] = self.dataset['Open'] / self.close
        self.dataset['adj_high'] = self.dataset['High'] / self.close
        self.dataset['adj_low'] = self.dataset['Low'] / self.close
        self.dataset['adj_close'] = self.dataset['Close'] / self.close


        # Group data by day
        self.grouped_by_day = self.dataset.groupby(pd.Grouper(freq='D'))

        self.days : List[pd.DataFrame] = []
        for date, day in self.grouped_by_day:
            if(len(day) > 0):
                day_preprocessed = self._preprocess_day(day, colums_to_drop)
                self.days.append(day_preprocessed)

        # random.shuffle(self.days)

    def _preprocess_day(self, day: pd.DataFrame, colums_to_drop: List[str]) -> pd.DataFrame:
        return day.drop(columns=colums_to_drop)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        return self.days[idx].to_numpy()