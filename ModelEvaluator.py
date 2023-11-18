#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 01:17:16 2023

@author: rishisingh
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = y_true - y_pred

    def calculate_mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def calculate_rmse(self):
        return np.sqrt(self.calculate_mse())

    def calculate_r2(self):
        return r2_score(self.y_true, self.y_pred)

    def plot_residuals_vs_predicted(self):
        plt.scatter(self.y_pred, self.residuals)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuals vs Predicted Values')
        plt.show()

    def plot_histogram_residuals(self):
        plt.hist(self.residuals, bins=20)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        plt.show()

    def plot_actual_vs_predicted(self):
        plt.scatter(self.y_true, self.y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.plot([self.y_true.min(), self.y_true.max()], [self.y_true.min(), self.y_true.max()], 'k--', lw=3)
        plt.show()
