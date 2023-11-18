#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:23:26 2023

@author: rishisingh
"""

import yfinance as yf
import pandas as pd


class StockData:
    def __init__(self, ticker_symbol, start_date, end_date):
        """
        Initializes the StockData object with ticker symbol, start and end dates for data fetching.
        :param ticker_symbol: str
        :param start_date: str
        :param end_date: str
        """
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        """
        Fetches historical stock data using yfinance.
        """
        self.data = yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date, interval="1d")

    def clean_data(self):
        """
        Cleans the fetched data by handling missing values.
        """
        self.data.fillna(method='ffill', inplace=True)

    def save_data(self, file_name):
        """
        Saves the cleaned data to a CSV file.
        :param file_name: str
        """
        self.data.to_csv(file_name)

    def get_data(self):
        """
        Returns the fetched and cleaned stock data.
        :return: pandas.DataFrame
        """
        return self.data
    
    