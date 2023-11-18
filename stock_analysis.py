#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:26:19 2023

@author: rishisingh
"""

import pandas as pd 
import matplotlib.pyplot as plt
import mplfinance as mpf

class StockAnalysis:
    
    def __init__(self, stock_data):
        """
        Initializes the StockAnalysis object with stock data.
        :param stock_data: pandas.DataFrame
        """
        self.stock_data = stock_data

    def calculate_moving_average(self, window_size):
        """
        Calculates the moving average for the specified window size.
        :param window_size: int
        """
        ma_column_name = f'SMA_{window_size}'
        self.stock_data[ma_column_name] = self.stock_data['Close'].rolling(window=window_size).mean()

    def calculate_rsi(self, period=14):
        """
        Calculates the Relative Strength Index (RSI) for the given period.
        :param period: int
        """
        delta = self.stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        self.stock_data['RSI'] = 100 - (100 / (1 + rs))

    def plot_price_and_ma(self, window_size):
        """
        Plots the closing prices along with the moving average.
        :param window_size: int
        """
        ma_column_name = f'SMA_{window_size}'
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['Close'], label='Close Price')
        plt.plot(self.stock_data[ma_column_name], label=f'{window_size}-day SMA')
        plt.title('Stock Price and Moving Average')
        plt.legend()
        plt.show()

    def plot_rsi(self):
        """
        Plots the Relative Strength Index (RSI).
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['RSI'], label='RSI')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.show()
        
    def calculate_bollinger_bands(self, window_size=20, num_std_dev=2):
        """
        Calculates Bollinger Bands.
        :param window_size: int
        :param num_std_dev: int
        """
        self.stock_data['SMA'] = self.stock_data['Close'].rolling(window=window_size).mean()
        std_dev = self.stock_data['Close'].rolling(window=window_size).std()
        self.stock_data['Upper_Band'] = self.stock_data['SMA'] + (std_dev * num_std_dev)
        self.stock_data['Lower_Band'] = self.stock_data['SMA'] - (std_dev * num_std_dev)

    def calculate_macd(self, short_window=12, long_window=26, signal=9):
        """
        Calculates the Moving Average Convergence Divergence (MACD).
        :param short_window: int
        :param long_window: int
        :param signal: int
        """
        self.stock_data['MACD'] = self.stock_data['Close'].ewm(span=short_window, adjust=False).mean() - \
                                  self.stock_data['Close'].ewm(span=long_window, adjust=False).mean()
        self.stock_data['MACD_Signal'] = self.stock_data['MACD'].ewm(span=signal, adjust=False).mean()

    def plot_bollinger_bands(self):
        """
        Plots Bollinger Bands.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['Close'], label='Close Price')
        plt.plot(self.stock_data['SMA'], label='Moving Average')
        plt.plot(self.stock_data['Upper_Band'], label='Upper Bollinger Band')
        plt.plot(self.stock_data['Lower_Band'], label='Lower Bollinger Band')
        plt.title('Bollinger Bands')
        plt.legend()
        plt.show()

    def plot_macd(self):
        """
        Plots the MACD and MACD Signal line.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['MACD'], label='MACD')
        plt.plot(self.stock_data['MACD_Signal'], label='MACD Signal Line')
        plt.title('MACD')
        plt.legend()
        plt.show()
        
    def plot_candlestick(self, window_size=100):
        """
        Plots a candlestick chart for the given window size.
        :param window_size: int
        """
        plot_data = self.stock_data[-window_size:]
        mpf.plot(plot_data, type='candle', style='charles', volume=True)

    def calculate_fibonacci_retracement_levels(self):
        """
        Calculates Fibonacci Retracement Levels as single values.
        """
        max_price = self.stock_data['Close'].max()
        min_price = self.stock_data['Close'].min()
        difference = max_price - min_price
        self.fibonacci_level_1 = max_price - difference * 0.236
        self.fibonacci_level_2 = max_price - difference * 0.382
        self.fibonacci_level_3 = max_price - difference * 0.618

    def plot_fibonacci_retracement_levels(self):
        """
        Plots Fibonacci Retracement Levels.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['Close'], label='Close Price')
        plt.axhline(y=self.fibonacci_level_1, color='r', linestyle='-', label='Fibonacci Level 0.236')
        plt.axhline(y=self.fibonacci_level_2, color='g', linestyle='-', label='Fibonacci Level 0.382')
        plt.axhline(y=self.fibonacci_level_3, color='b', linestyle='-', label='Fibonacci Level 0.618')
        plt.title('Fibonacci Retracement Levels')
        plt.legend()
        plt.show()