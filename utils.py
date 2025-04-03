# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:13:11 2025

@author: LZ166
"""

import numpy as np
import pandas as pd
from scipy.special import gamma, digamma
import math



def wrmsse(Y_pred,Y_test,Y_train):
    list_rmsse=np.array([rmsse(Y_pred[i],Y_test[i],Y_train[i]) for i in range(0,Y_pred.shape[0])])
    weights=np.array([np.median(y) for y in Y_train])
    
    valid_ids=[i for i in range(0,len(list_rmsse)) if not math.isinf(list_rmsse[i])]
    weights=weights[valid_ids]/np.sum(weights[valid_ids])
    return np.dot(list_rmsse[valid_ids],weights),weights,list_rmsse

def rmsse(preds, actuals, y_train):
    """
    Calculates the Root Mean Squared Scaled Error (RMSSE).

    Args:
        preds (np.array): Forecasted values.
        actuals (np.array): Actual values.
        y_train (np.array): Training data for calculating the scaling factor.

    Returns:
        np.array: RMSSE values for each series.
    """
    mse = np.mean((preds - actuals)**2)
    scale = np.mean((y_train[1:] - y_train[:-1])**2,dtype=np.float64) # Mean Squared Error of naive forecast on training data
    scale=scale if scale>1 else 1
    rmsse_val = np.sqrt(mse / (scale))
    return rmsse_val

    

def create_lag_data(df,lags=range(1,15)):
    df_lag = df.copy()
    for lag in lags:
        df_lag[f'lag_{lag}'] = df['sales'].shift(lag)
    df_lag['ema_3d'] = round(df['sales'].ewm(span=3).mean().shift(1))
    df_lag['ema_7d'] = round(df['sales'].ewm(span=7).mean().shift(1))
    df_lag['ema_14d'] = round(df['sales'].ewm(span=14).mean().shift(1)) 
    df_lag['ema_28d'] = round(df['sales'].ewm(span=28).mean().shift(1)) 
    df_lag['std_3d']=round(df['sales'].rolling(window=3).std().shift(1))
    df_lag['std_7d']=round(df['sales'].rolling(window=7).std().shift(1))
    df_lag['std_14d']=round(df['sales'].rolling(window=14).std().shift(1))
    df_lag['std_28d']=round(df['sales'].rolling(window=28).std().shift(1))
    return df_lag.dropna()

def neg_binomial_obj(preds, dtrain):
    y = dtrain.get_label()
    r = 1.5  # Dispersion parameter (you might need to estimate this)
    mu = np.exp(preds)  # Link function: log link

    grad = (r + y) * (digamma(r + y) - digamma(r) - np.log(r) + np.log(mu))
    hess = (r + y) * (1/mu) * (1 - digamma(r + y) + digamma(r) + np.log(r) - np.log(mu))

    return grad, hess

# Define the evaluation metric (optional)
def neg_binomial_eval(preds, dtrain):
    y = dtrain.get_label()
    r = 1.0  # Same dispersion parameter as in the objective
    mu = np.exp(preds)
    loglik = gamma(r + y) - gamma(y + 1) - gamma(r) + r * np.log(r) - r * np.log(r + mu) + y * np.log(mu) - y * np.log(r + mu)
    return 'neg_loglik', -np.mean(loglik)
