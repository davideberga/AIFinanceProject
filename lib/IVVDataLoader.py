import random
from typing import List
from pandas import read_csv
import pandas as pd
from torch.utils.data import Dataset

class IVVDataset(Dataset):
    def __init__(self, file_csv, colums_to_drop=[]):
        self.dataset = read_csv(file_csv, index_col=0, parse_dates=[0], header=0)
        self.dataset.isnull().values.any()
        self.dataset=self.dataset.fillna(method='ffill')

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