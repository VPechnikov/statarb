from datetime import date, timedelta, datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.util.Features import Features
from src.util.Tickers import Tickers
from src.DataRepository import Universes


class Window:
    def __init__(self, window_start: datetime, trading_win_len: int):
        """
        @param window_start: datetime object to indicate the starting date for the window
        @param trading_win_len: the length of the window as integer
        """
        # we need to use datetime for start_date as this is how dates are parsed automatically
        self.data: DataFrame = pd.read_csv('../resources/all_data.csv', header=[0, 1, 2], index_col=0, parse_dates=True)

        self.window_start: datetime = window_start
        self.trading_win_len: int = trading_win_len
        self.window_end: datetime = self.__get_nth_working_day_ahead(self.window_start, self.trading_win_len)
        self.__update_current_window()

    def __update_current_window(self):
        """
        @return: updates the current_window and removes the dead_tickers
        """
        final_day = self.__get_nth_working_day_ahead(target=self.window_end, n=1)
        self.current_window = self.data.loc[self.window_start: final_day]
        self.remove_dead_tickers()

    def check_date_equality(self, d1: date, d2: date) -> bool:
        """
        the function returns True if the day, month, year are the same
        """
        return (d1.day == d2.day and
                d1.month == d2.month and
                d1.year == d2.year)

    # Yimiao, please see the functions below
    def __get_nth_working_day_ahead(self, target: datetime, n: int) -> datetime:
        # here we want to be able to identify the index of the self.today date (i) and return the date which
        # corresponds to the index of i+n
        # we need to search for the index of today's date in self.data
        idx = np.where(self.data.index == target)[0]
        last_day = self.data.index[-1]
        return min(self.data.iloc[idx + n].index[0], last_day)  # so that we do not run out of data

    def roll_forward_one_day(self):
        """
        @return: updates the current_window and shifts the data by 1 day forward
        """
        # here we want to increment the self.today date by calling __get_nth_wortking_day_ahead() with n = 1
        next_starting_day = self.__get_nth_working_day_ahead(target=self.window_start, n=1)
        self.window_start = next_starting_day
        self.__update_current_window()

    def get_data(self,
                 universe: Optional[List[Universes]] = None,
                 tickers: Optional[List[Tickers]] = None,
                 features: Optional[List[Features]] = None) -> DataFrame:
        """
        @param universe: Universes.SNP or Universes.ETFs
        @param tickers: List of tickers to get data for, if not specified returns data for all tickers
        @param features: List of features to get data for, if not specified returns data for all features
        @return:DataFrame
        """
        if universe is None:
            if tickers is None and features is None:
                return self.current_window

            elif tickers is None:
                return self.current_window.loc[:, pd.IndexSlice[:, :, features]]

            elif features is None:
                return self.current_window.loc[:, pd.IndexSlice[:, tickers, :]]

        else:
            if tickers is None and features is None:
                return self.current_window[universe]

            if tickers is None:
                return self.current_window.loc[:, pd.IndexSlice[universe, :, features]]

            if features is None:
                return self.current_window.loc[:, pd.IndexSlice[universe, tickers, :]]

    def remove_dead_tickers(self):
        """
        @return: updates the value of current_window to only contain the data
        for tickers where there are no missing values
        """
        all_tickers = set([i[1] for i in self.data.columns])  # since data has MultiIndex columns,
        dead_tickers = set()
        for ticker in all_tickers:
            column = self.current_window.loc[:, pd.IndexSlice[:, ticker]]
            if any(column.isna().sum()):
                dead_tickers += set(ticker)
        alive_tickers = all_tickers - dead_tickers
        self.current_window = self.current_window.loc[:pd.IndexSlice[:, alive_tickers]]
