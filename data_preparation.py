#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 23:13:25 2023

@author: rishisingh
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreparation:
    def __init__(self, stock_data, stock_analysis):
        """
        Initializes the MLPreparation object with stock data.
        :param stock_data: pandas.DataFrame
        """
        self.stock_data = stock_data
        self.stock_analysis = stock_analysis

    def create_lag_features(self, lags=1):
        """
        Creates lag features for the specified number of lags.
        :param lags: int
        """
        for lag in range(1, lags + 1):
            self.stock_data[f'Close_lag{lag}'] = self.stock_data['Close'].shift(lag)

    def create_rolling_window_features(self, window=5):
        """
        Creates rolling window features like mean and standard deviation.
        :param window: int
        """
        self.stock_data[f'Rolling_mean_{window}'] = self.stock_data['Close'].rolling(window=window).mean()
        self.stock_data[f'Rolling_std_{window}'] = self.stock_data['Close'].rolling(window=window).std()

    def normalize_features(self, feature_names):
        """
        Normalizes the specified features using StandardScaler.
        :param feature_names: list
        """
        scaler = StandardScaler()
        self.stock_data[feature_names] = scaler.fit_transform(self.stock_data[feature_names].values)
        

    def add_technical_indicators(self):
        """
        Adds technical indicators from StockAnalysis to the dataset.
        """
        # Assuming StockAnalysis has methods to calculate indicators
        # and returns them as Series or adds them directly to stock_data
        self.stock_analysis.calculate_moving_average(50)
        self.stock_analysis.calculate_rsi()
        self.stock_analysis.calculate_macd()
        # Add more indicators as required

    def create_labels(self, future_period=1):
        """
        Creates labels for a classification task.
        :param future_period: int
        """
        self.stock_data['Future_Close'] = self.stock_data['Close'].shift(-future_period)
        self.stock_data['Direction'] = (self.stock_data['Future_Close'] > self.stock_data['Close']).astype(int)

    def prepare_data(self):
        """
        Returns the prepared dataset after all processing steps.
        """
        # Drop NaN values created by shifts, rolling windows, etc.
        self.stock_data.dropna(inplace=True)
        return self.stock_data