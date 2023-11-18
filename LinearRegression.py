#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:46:55 2023

@author: rishisingh
"""

import numpy as np
from ModelEvaluator import ModelEvaluator
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=10, debug=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.debug = debug

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Adjust learning rate based on number of features
        adjusted_learning_rate = self.learning_rate / n_features

        # Gradient Descent
        for i in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (model - y))
            db = (1 / n_samples) * np.sum(model - y)

            if np.isnan(dw).any() or np.isnan(db):
                print("NaN detected in gradients at iteration", i)
                break

            self.weights -= adjusted_learning_rate * dw
            self.bias -= adjusted_learning_rate * db

            if self.debug and i % 100 == 0:
                print(f"Iteration {i}: Weights: {self.weights}, Bias: {self.bias}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def evaluate(self, X_test, y_test, verbose=True, plot=False):
        predictions = self.predict(X_test)
        evaluator = ModelEvaluator(y_test, predictions)
        
        if verbose:
            print("MSE:", evaluator.calculate_mse())
            print("RMSE:", evaluator.calculate_rmse())
            print("R^2:", evaluator.calculate_r2())

        if plot:
            evaluator.plot_residuals_vs_predicted()
            evaluator.plot_histogram_residuals()
            evaluator.plot_actual_vs_predicted()
        
        return {'mse': evaluator.calculate_mse(), 'r2': evaluator.calculate_r2()}
