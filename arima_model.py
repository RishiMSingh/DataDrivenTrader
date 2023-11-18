#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 01:37:00 2023

@author: rishisingh
"""

# arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize the ARIMA Model with a specified order.
        :param order: tuple (p, d, q) for the ARIMA model
        """
        self.order = order
        self.model = None

    def fit(self, training_series):
        """
        Fit the ARIMA model to the training data.
        :param training_series: pandas.Series, the time series data to fit the model on
        """
        self.model = ARIMA(training_series, order=self.order)
        self.model = self.model.fit()

    def forecast(self, steps=1):
        """
        Forecast future values using the fitted model.
        :param steps: int, number of future steps to forecast
        :return: Forecasted values
        """
        return self.model.forecast(steps=steps)

    def evaluate(self, actual, forecast):
        """
        Evaluate the model's performance.
        :param actual: Actual values
        :param forecast: Forecasted values
        :return: dict containing MSE and R2 score
        """
        mse = mean_squared_error(actual, forecast)
        r2 = r2_score(actual, forecast)
        return {'mse': mse, 'r2': r2}

    

    