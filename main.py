# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:37:33 2025

@author: LZ166
"""

import pandas as pd
import matplotlib.pyplot as plt
from myHTS import HierarchicalTimeSeriesTree,TreeForecastor,NodeForecastor,update
from utils import rmsse,wrmsse
import numpy as np
from sklearn.preprocessing import LabelEncoder

df_date=pd.read_csv("calendar.csv")
df_sales=pd.read_csv("sales_train_evaluation.csv")
df_price=pd.read_csv("sell_prices.csv")

sales_per_store = {name: group.copy() for name, group in df_sales.groupby('store_id')}

def preprocessing(df_sales,df_date,tree_level,time_horizon):
    df=pd.melt(df_sales,id_vars=tree_level,value_vars=[f'd_{i}' for i in range(1,time_horizon)],var_name='d',value_name='sales')
    df=pd.merge(df,df_date[['wm_yr_wk','wday','month','d']],on='d',how='left')
    df.drop('d',axis=1,inplace=True)
    df['wm_yr_wk']=df['wm_yr_wk']%100
    
    le = LabelEncoder()
    for level in tree_level:
        df[level]=le.fit_transform(df[level])
    return df


tree_level=['cat_id','dept_id','item_id']
node_levels=['item_id']

df_example=preprocessing(sales_per_store['CA_1'],df_date,tree_level,time_horizon=1913)
train_time=[f'd_{i}' for i in range(1,1914)]
test_time=[f'd_{i}' for i in range(1914,1942)]
Y_test=np.array(sales_per_store['CA_1'][test_time])
Y_train=np.array(sales_per_store['CA_1'][train_time])
# groups = df_example.groupby('cat_id')
# dict_df_groups={name:group.copy() for name,group in groups}

hts=HierarchicalTimeSeriesTree(df_example, tree_level)
train_data=hts.get_train_data(node_levels)

tree_forecastor=TreeForecastor(NodeForecastor)
tree_forecastor.fit(hts,node_levels,train_data)

X_init=[]
for i,(_,nodes_id,X_train,y_train) in enumerate(train_data):
    print(i)
    print(X_train.shape)
    T_train = X_train.shape[0] // len(nodes_id)
    X_init.append(
            update(X_train[len(nodes_id)*(T_train-1):len(nodes_id)*T_train], y_train[len(nodes_id)*(T_train-1):len(nodes_id)*T_train],n_lags=28))

Y_pred=tree_forecastor.forecast(X_init, T=28)
Y_pred_aggregate=np.concatenate([Y_pred[i].T for i in range(0,len(Y_pred))])
wrmsse_value,weights,rmsses=wrmsse(Y_pred_aggregate, Y_test, Y_train)

X_init=[]
for i,(_,nodes_id,X_train,y_train) in enumerate(train_data):
    print(i)
    print(X_train.shape)
    T_train = X_train.shape[0] // len(nodes_id)
    X_init.append(
            update(X_train[len(nodes_id)*(T_train-29):len(nodes_id)*(T_train-28)], y_train[len(nodes_id)*(T_train-29):len(nodes_id)*(T_train-28)],n_lags=28))
Y_pred_train=tree_forecastor.forecast(X_init, T=28)
Y_pred_aggregate_train=np.concatenate([Y_pred[i].T for i in range(0,len(Y_pred))])
wrmsse_value_train,weights,rmsses_train=wrmsse(Y_pred_aggregate_train, Y_train[:,-28:], Y_train)
list_valid_id=np.where(rmsses_train<1)
weights_valid=weights[list_valid_id]
rmsses_valid=rmsses[list_valid_id]
wrmsse_valid=weights_valid.dot(rmsses_valid)/np.sum(weights_valid)

# train_data=hts.get_train_data(node_levels)
# Y_pred=forecastHTS(hts,T=28,node_levels=node_levels)