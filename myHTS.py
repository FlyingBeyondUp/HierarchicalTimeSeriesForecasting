# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 11:13:23 2025

@author: LZ166
"""
# Idea:
# Hierarchical Time Series Forecast
# Example: Walmart's data of sales of goods
# Three different levels: Location (CA_1), department, and item
# up-bottom: predict the aggregate sales at location level and then allocate among depts and items 
#            according to the predicted ratios;
# bottom-up: predict the sales of items and then obtain the aggregate sales at dept and location levels
#            To get the sales of items: 1. train model for each item; 2. use item id as an input factor.
# middle-out: predict the aggregate sales of goods at department level and then aggregate to get the sales 
#             at location level while disaggregate to item levels according to the predicted ratios.
# The loss function: weighted rooted mean squared scaled error.
#                    weight: y_train_i/y_train_total for department and item level,
#                            for location level, simply set weight=1.
#                    scaled: np.mean((y_train[1:] - y_train[:-1])**2).
# The prediction model for each time series should be specified ( like xgboost ).
# Furthermore: an extra level that aggregates over all locations can be added as the top level, but is not necessary at this stage.

# Structure of the program:
# The hierarchical structure can be naturally expressed as a tree, which can be represented by the summing matrix.
# The time series forecasting for each node itself can be tackled in the same manner. 
# The aggregate and disaggregate steps can be realized to be the same for all possible levels.

from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import scipy.sparse as sp

class NodeForecastor:
    def __init__(self,model=XGBRegressor(
        objective="count:poisson",  
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )):
        self.model=model
        return
    def fit(self,X_train,y_train):
        self.model.fit(X_train,y_train)
        return
    
    def forecast(self,X_init,time_horizon):
        Y_pred=[]
        X=X_init.copy()
        for t in range(0,time_horizon):
            Y_pred.append(self.model.predict())
            X=self.update(X, Y_pred[-1])
        return Y_pred
        
    def update(self,X,Y):
        ...

class HierarchicalTimeSeriesTree:
    def __init__(self,df,tree_structure,create_features):
        self.df=df
        self.tree_structure=tree_structure
        self.create_features

    def get_train_data(self,node_levels):
        
        ...
    def get_SMatrix(self):
        ...
    def get_children_p(self,node_level):
        ...
    def split_time_series(self):
        ...

        
        
def forecastHTS(hts,model_class,T,node_levels=[]):
    '''
    hts: HierarchicalTimeSeriesTree object.
    model_class: NodeForecastor class, trained by .fit method and predict by .predict method.
    T: The time horizon of forecasting.
    node_levels: 
    specify which levels of nodes are needed to forecast the time series of the current level.
    The order should follow the structure of the hierarchy of the original data.
    example: From the top to the bottom, the levels are location(root),cat_id,dept_id,item_id.
    
             For bottom_up method, item_id should be included:
             node_levels=['item'] means that the training data has the same dept_id
             and thus only the item id needs to be added as a categorical variable.
             The number models need to be trained is the number of depts.
             An extra function will be added to deal with more complex tree structure.
             
             However, if node_levels=['dept','item'], then the data with different dept_id
             and the same cat_id will be organized together as the training data, 
             thus both of dept_id and item_id should be used as the categorical variables.
             The number of models need to be trained is the number of categories.
             
             For middle_out method, levels other than item_id are used.
             The predicted time series of the nodes in the current level are 
             then disaggregated to lower levels according to children nodes relation
             until the leaf nodes level is reached.
             Then, the time series for all different hierarchies are obtained by multiplying SMatrix.
             If node_levels=['dept'], then the models are trained by time series of different depts with
             the same category id, thus the number of models need to be trained is n_category.
    '''
    X_train,Y_train=hts.get_train_data(node_levels) # X_train shape: (the number of models, number of past time steps*number of nodes, number of features)
    model.fit(X_train,Y_train)
    X_init=model.update(X_train[-1],Y_train[-1]) # X_init has shape (Nl,p), where p is the number of features.
    Y_pred=[model.forecast(X_init[i]) for i in range(0,X_init.shape[0])]    # Y_pred has shape (Nl,T), where Nl is the number of nodes at the current level.
    
    i0=hts.tree_structure.index(node_levels[-1])
    for i in range(i0,len(hts.tree_structure)):
        p=hts.get_children_p(hts.tree_structure[i])   # p: (number of nodes, number of child nodes, time steps)
        Y_next=p[0]@sp.block_diag(Y_pred[0])
        for j in range(1,len(p)):
            Y_next=sp.vstack(Y_next,p[j]@sp.block_diag(Y_pred[j]))
        Y_pred=Y_next.copy()
        
    SMatrix=hts.get_SMatrix()
    whole_time_series=SMatrix@Y_pred
    return hts.split_time_series(whole_time_series)
        
    

    
    
    
    
    
    
    
    
    
    
    