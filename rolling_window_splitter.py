#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:20:56 2023

@author: rishisingh
"""

import pandas as pd
from datetime import timedelta

class RollingWindowSplitter:
    def __init__(self, data, train_window_size='3Y', test_window_size='1Y'):
        """
        Initializes the RollingWindowSplitter with the dataset and window sizes.
        :param data: pandas.DataFrame
        :param train_window_size: str
        :param test_window_size: str
        """
        self.data = data
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size

    def split(self):
        """
        Splits the data using a rolling window approach.
        Yields training and testing datasets for each window.
        """
        start_date = self.data.index.min()
        end_date = self.data.index.max()

        current_start = start_date

        while current_start + pd.DateOffset(years=int(self.train_window_size[:-1])) <= end_date:
            train_end = current_start + pd.DateOffset(years=int(self.train_window_size[:-1])) - timedelta(days=1)
            test_end = train_end + pd.DateOffset(years=int(self.test_window_size[:-1]))

            # Ensuring the test_end does not exceed the dataset
            test_end = min(test_end, end_date)

            # Define training and testing sets
            train_data = self.data[current_start:train_end]
            test_data = self.data[train_end + timedelta(days=1):test_end]

            # Yield the train and test sets for this window
            yield train_data, test_data

            # Move the window forward by one year
            current_start += pd.DateOffset(years=1)