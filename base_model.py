# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:27:35 2025

@author: LZ166
"""

import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from utils import rmsse, create_lag_data, neg_binomial_obj
from statsmodels.tsa.stattools import acf



df_date=pd.read_csv("calendar.csv")
df_sales=pd.read_csv("sales_train_evaluation.csv")
df_price=pd.read_csv("sell_prices.csv")

sales_per_store = {name: group.copy() for name, group in df_sales.groupby('store_id')}

example_sales=sales_per_store['CA_1']
example_aggregate_sales=example_sales[[f'd_{i}' for i in range(1,1942)]].sum()



example_item_sales=example_sales[0][6:]
print(f'max daily sales: {example_item_sales.max()}')
print(f'mean daily sales: {example_item_sales.mean()}')
print(f'std daily sales: {example_item_sales.std()}')
print(f'skew daily sales: {example_item_sales.skew()}')

df=pd.DataFrame({'sales':example_aggregate_sales.values,'wday':df_date.wday[:1941],'month':df_date.month[:1941],'week':df_date.wm_yr_wk[:1941]%100})
wday_sales=df.groupby('wday')['sales'].sum()
month_sales=df.groupby('month')['sales'].sum()
week_sales=df.groupby('week')['sales'].sum()
plt.figure()
plt.xlabel('weekday')
plt.ylabel('cumulative aggregate sales')
plt.bar(wday_sales.index,wday_sales.values)
plt.savefig('weekday_sales.pdf', bbox_inches='tight')
plt.bar(month_sales.index,month_sales.values)
plt.bar(week_sales.index,week_sales.values)

acf_values, confint = acf(df['sales'], nlags=30, alpha=0.05)  # 95% confidence interval

# 3. Plot the ACF to visually inspect significant lags
plt.figure(figsize=(10, 5))
plt.stem(np.arange(len(acf_values)), acf_values)
plt.fill_between(np.arange(len(acf_values)), confint[:, 0] - acf_values, confint[:, 1] - acf_values,
                 color='gray', alpha=0.3)
plt.title('ACF Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
plt.savefig('ACF.pdf',bbox_inches='tight')


# base model: for week day i, use the averaged value of the sales data 
# in the previous four weeks on week day i as the prediction
df_given,df_target=df[:1913],df[1913:]
for i in range(1913,1941):
    sales_pred=(df_given.sales[i-7]+df_given.sales[i-14]+df_given.sales[i-21]+df_given.sales[i-28]).item()/4
    df_given=pd.concat([df_given,pd.DataFrame([sales_pred],columns=df_given.columns)],ignore_index=True)
print(rmsse(df_given[1913:].sales, df_target['sales'], df[:1913].sales.values))
plt.xlabel('eva_date')
plt.ylabel('aggregate sales')
plt.plot(range(0,28),df_given[1913:]['sales'],label='predicted')
plt.plot(range(0,28),df_target['sales'],label='accurate')
plt.legend()


df_week=pd.DataFrame({'sales':week_sales.values,'week':week_sales.index%100})





# df['days']=np.sin(2*np.pi*df_date.wday.values[0:1941]/7)
# df['weeks']=np.sin(2*np.pi*(df_date.wm_yr_wk.values[0:1941]%100)/52)
# df['months']=np.sin(2*np.pi*df_date.month.values[0:1941]/12)
df['event1']=np.array(~df_date.event_name_1[0:1941].isna().astype(bool))
df['event2']=np.array(~df_date.event_name_2[0:1941].isna().astype(bool))
event_sales1=df.groupby('event1')['sales'].sum()
event_sales2=df.groupby('event2')['sales'].sum()


# remember to shift the dates by one.
lags=range(1,29)
df_lag=create_lag_data(df,lags)

features =[f'lag_{lag}' for lag in lags]+['week','wday','month']
X = df_lag[features].values
y = df_lag['sales'].values

train_size=len(y)-28
X_train,y_train=X[:train_size],y[:train_size]
X_test,y_test=X[train_size:],y[train_size:]

tree = DecisionTreeRegressor(criterion="poisson",random_state=0,max_depth=10)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
print(rmsse(y_pred_tree, y_test, y_train))
# mae_tree = mean_absolute_error(y_test, y_pred_tree)
# print("Mean Absolute Test Error tree:", mae_tree)

adaboost_model = AdaBoostRegressor(
    tree,
    n_estimators=500,       # Adjust as needed
    learning_rate=0.001,    # Adjust as needed
    random_state=0
)

# 5. Train the AdaBoost Model
adaboost_model.fit(X_train, y_train)

# 6. Make Predictions
y_pred_adaboost = adaboost_model.predict(X_test)
print(rmsse(y_pred_adaboost, y_test, y_train))

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(criterion="poisson",random_state=0, n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(rmsse(y_pred_rf, y_test, y_train))


model = xgb.XGBRegressor(
    objective="count:poisson",  # Specify Tweedie objective
    # tweedie_variance_power=1.1,  # Adjust the variance power (see explanation below)
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, y_train/np.std(y_train),
    # eval_set=eval_set,
    verbose=False
)
y_pred_xgb = model.predict(X_test)*np.std(y_train)
print("rmsse (xgb):", rmsse(model.predict(X_train)*np.std(y_train), y_train, y_train))
print("rmsse (xgb):", rmsse(y_pred_xgb, y_test, y_train))
# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
# mae_xgb_train=mean_absolute_error(y_train, [round(y) for y in model.predict(X_train)])
# print("Mean Absolute Train Error (xgb):", mae_xgb_train)
# print("Mean Absolute Test Error (xgb):", mae_xgb)




# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': neg_binomial_obj,  # Use the custom objective
    # 'eval_metric': neg_binomial_eval, # Use the custom eval metric
    'eta': 0.001,
    'max_depth': 10,
    'seed': 42
}

# Train the model
model = xgb.train(params, dtrain,obj=neg_binomial_obj, num_boost_round=200,
                  evals=[(dtrain, 'train'), (dtest, 'test')],
                   verbose_eval=False)

# Make predictions (remember to exponentiate to get the actual predictions)
y_pred_xgb_nb = [round(y) for y in np.exp(model.predict(dtest))]
mae_xgb_nb = mean_absolute_error(y_test, y_pred_xgb_nb)
print("Mean Absolute Test Error tree:", mae_xgb_nb)

plt.plot(range(0,28),y_pred_tree,label='pred_tree')
plt.plot(range(0,28),y_pred_xgb,label='pred_xgb')
plt.plot(range(0,28),y_pred_xgb_nb,label='pred_xgb_nb')
plt.plot(range(0,28),y_test,label='test')
plt.legend()












