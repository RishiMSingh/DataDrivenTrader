#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:23:26 2023

@author: rishisingh
"""

from stock_data import StockData
from stock_analysis import StockAnalysis
from data_preparation import DataPreparation
from rolling_window_splitter import RollingWindowSplitter
from sklearn.metrics import mean_squared_error, r2_score
from LinearRegression import LinearRegression
from arima_model import ARIMAModel
import numpy as np

# Example usage
stock = StockData('AAPL', '2000-01-01', '2023-01-01')
stock.fetch_data()
stock.clean_data()
stock.save_data('AAPL_stock_data.csv')

# Access the data
data = stock.get_data()


# Analyze the stock data
analysis = StockAnalysis(stock.get_data())
analysis.calculate_moving_average(50)
analysis.calculate_rsi()
analysis.plot_price_and_ma(50)
analysis.plot_rsi()

analysis.calculate_bollinger_bands()
analysis.plot_bollinger_bands()

analysis.calculate_macd()
analysis.plot_macd()

analysis.plot_candlestick()
analysis.calculate_fibonacci_retracement_levels()
analysis.plot_fibonacci_retracement_levels()

# Prepare data for data
data_prep = DataPreparation(stock.get_data(), analysis)
data_prep.add_technical_indicators()    # Add indicators like RSI, MACD from StockAnalysis
data_prep.create_lag_features(lags=3)   # Create lag features
data_prep.create_rolling_window_features(window=5)  # Create rolling window features
# Normalize features including the new technical indicators
feature_cols = ['Close', 'Volume', 'Close_lag1', 'Rolling_mean_5', 'Rolling_std_5', 'RSI', 'MACD']  # Add all feature column names here
data_prep.normalize_features(feature_cols)
data_prep.create_labels(future_period=1)  # Create labels for supervised learning

prepared_data = data_prep.prepare_data()

# Initialize the RollingWindowSplitter with the prepared data
splitter = RollingWindowSplitter(prepared_data)

# Specify the label column name
label_column = 'Direction'

# Perform rolling window split and model training/testing
idx_count = 0
results = []
arima_results = []

for train_df, test_df in splitter.split():
    print(train_df.columns)



# for train_df, test_df in splitter.split():
#     idx_count += 1
#     # Splitting the training and testing data
#     X_train, y_train = train_df.drop(label_column, axis=1), train_df[label_column]
#     X_test, y_test = test_df.drop(label_column, axis=1), test_df[label_column]
    
    
#     # Create and train the linear regression model
#     lr_model = LinearRegression(learning_rate=0.0001, n_iterations=500)
#     lr_model.fit(X_train, y_train)
    
#     # Evaluate the model on the test set
#     lr_model.evaluate(X_test, y_test)
    
    
#     # Initialize and train the ARIMA model
#     arima_model = ARIMAModel(order=(1, 1, 1))
#     arima_model.fit(train_df['Close'])
#     arima_forecast = arima_model.forecast(steps=len(test_df))
#     arima_results.append(arima_model.evaluate(test_df['Close'], arima_forecast))

