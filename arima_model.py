#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 01:37:00 2023

@author: rishisingh
"""

# arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize the ARIMA Model with specified order.
        :param order: tuple (p, d, q) where p is the order of the AR term,
                      d is the degree of differencing, and q is the order of the MA term.
        """
        self.order = order
        self.model = None

    def fit(self, training_series):
        """
        Fit the ARIMA model to the training data.
        :param training_series: pandas.Series
        """
        self.model = ARIMA(training_series, order=self.order)
        self.model = self.model.fit()

    def forecast(self, steps=1):
        """
        Make forecasts using the fitted model.
        :param steps: int, number of steps to forecast
        :return: Forecasted values
        """
        forecast = self.model.forecast(steps=steps)
        return forecast
    
    def evaluate(self, X_test, y_actual):
        """
        Evaluate the model's performance.
        :param X_test: Test features
        :param y_actual: Actual target values
        :return: dict containing MSE and R2 score
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_actual, predictions)
        r2 = r2_score(y_actual, predictions)
        return {'mse': mse, 'r2': r2}
    

    