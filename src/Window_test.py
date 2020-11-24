import sys
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from src.util.Features import Features
from src.util.Tickers import Tickers

class Window:
    def __init__(self, start_date: date, window_length: timedelta):
        self.start_date: date = start_date
        self.window_length: timedelta = window_length
        self.data: DataFrame = pd.read_csv('../resources/all_data.csv', header=[0, 1, 2], index_col=0, parse_dates=True)
        self.current_window = self.data[start_date, start_date + window_length]
