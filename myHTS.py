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
from multiprocessing import Pool, cpu_count

n_lags=28
def create_features(df, n_nodes, n_lags=n_lags):
    """
    Create lag features and target for time series forecasting.
    
    Parameters:
    - df: DataFrame with columns 'date', 'sales', and categorical columns in cat_levels.
    - n_nodes: the number of nodes in the forecasted level
    - n_lags: Number of lagged sales features to include (default is 7).
    
    Returns:
    - X: Numpy array of features, shape (number of samples=n_nodes*n_train_time_steps, number of features).
    - y: Numpy array of targets, shape (number of samples,).
    """
    df_lag=df.copy()
    for lag in range(1,n_lags):
        df_lag[f'lag_{lag}']=df['sales'].shift(lag*n_nodes)
    df_lag=df_lag.dropna()
    return np.array(df_lag.drop('sales',axis=1)),np.array(df_lag.sales)

def update(X, y_new,n_lags=n_lags):
    """Update the feature vector with the new prediction."""
    # X format: [lag1, lag2, ..., lagN, categorical_features...]
    lags = X[:,-n_lags:]
    cats = X[:,:X.shape[1]-n_lags]
    new_lags = np.roll(lags[:], 1)
    new_lags[:,0] = y_new
    return np.concatenate([cats,new_lags],axis=1)

def get_init(start,end,train_data):
    X_init=[]
    for i,(_,nodes_id,X_train,y_train) in enumerate(train_data):
        print(i)
        print(X_train.shape)
        # T_train = X_train.shape[0] // len(nodes_id)
        X_init.append(
                update(X_train[len(nodes_id)*start:len(nodes_id)*end], y_train[len(nodes_id)*(start-1):len(nodes_id)*end],n_lags=28))

    return X_init
    

class NodeForecastor:
    def __init__(self, model=None):
        # Default to XGBoost with Poisson objective for count data
        self.model = model if model else XGBRegressor(
            objective="count:poisson",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=15,
            random_state=42
        )
        self.n_lags = n_lags  # Number of lagged features

    def fit(self, X_train, y_train):
        """Train the model on the provided data."""
        self.model.fit(X_train, y_train)

    def forecast(self, X_init, time_horizon):
        """Forecast T steps ahead recursively."""
        Y_pred = []
        X = X_init.copy()
        for _ in range(time_horizon):
            y_new = self.model.predict(X)
            Y_pred.append(y_new)
            X = update(X, y_new)
        return np.array(Y_pred)

    

class HierarchicalTimeSeriesTree:
    def __init__(self, df, tree_structure):
        self.df = df.copy()
        self.tree_structure = tree_structure  # e.g., ['store', 'dept', 'item']
        self.create_features = create_features  # Function to generate X, y


    def get_train_data(self, node_levels):
        """Prepare training data for each group."""
        forecast_level = node_levels[-1]
        i0 = self.tree_structure.index(node_levels[0])
        grouping_levels = self.tree_structure[:i0]  # Levels above node_levels

        train_data = []
        if not grouping_levels:
            group_key = ['__total__']
            groups = [None]
        else:
            group_key = grouping_levels
            groups = self.df.groupby(group_key)

        for group_name, group_df in groups:
            group_id = group_name if grouping_levels else '__total__'
            sub_df = group_df.copy()
            for level in node_levels:
                sub_df[level] = sub_df[level].astype('category')
            node_ids = sub_df[forecast_level].cat.categories if forecast_level in sub_df else [group_id]
            
            X, y = self.create_features(sub_df, n_nodes=len(node_ids)) # The shape of X is (T_train*N,p), 
                                                             # T_train is the training time step
                                                             # N is the number of nodes included in the node_levels for one group, 
                                                             # p is the number of features.
            train_data.append((group_id, node_ids, X, y))
        return train_data

    def get_SMatrix(self):
        """Construct the summing matrix S."""
        bottom_level = self.tree_structure[-1]
        bottom_nodes = self.df[bottom_level].unique()
        n_bottom = len(bottom_nodes)
        total_nodes = sum(len(self.df[level].unique()) for level in self.tree_structure) + 1
        S = sp.lil_matrix((total_nodes, n_bottom))

        idx = 0
        S[0, :] = 1  # Total level
        idx += 1

        bottom_idx_map = {node: i for i, node in enumerate(bottom_nodes)}
        for level in self.tree_structure:
            nodes = self.df[level].unique()
            for node in nodes:
                if level == bottom_level:
                    S[idx, bottom_idx_map[node]] = 1
                else:
                    children = self.df[self.df[level] == node][bottom_level].unique()
                    for child in children:
                        S[idx, bottom_idx_map[child]] = 1
                idx += 1
        return S.tocsr()

    def get_children_p(self, node_level):
        """Compute proportions of child nodes."""
        next_level_idx = self.tree_structure.index(node_level) + 1
        if next_level_idx >= len(self.tree_structure):
            return {}
        next_level = self.tree_structure[next_level_idx]
        parents = self.df[node_level].unique()
        p_dict = {}
        for parent in parents:
            children = self.df[self.df[node_level] == parent][next_level].unique()
            child_sales = [
                self.df[(self.df[node_level] == parent) & (self.df[next_level] == child)]['sales'].sum()
                for child in children
            ]
            total = sum(child_sales) + 1e-6
            p_dict[parent] = np.array(child_sales) / total
        return p_dict

    def split_time_series(self, whole_time_series):
        """Split aggregated forecasts into hierarchical levels."""
        idx = 0
        forecasts = {'total': whole_time_series[idx]}
        idx += 1
        for level in self.tree_structure:
            nodes = self.df[level].unique()
            forecasts[level] = {}
            for node in nodes:
                forecasts[level][node] = whole_time_series[idx]
                idx += 1
        return forecasts


# Parallel training and forecasting function
# def train_and_forecast_group(args):
#     group_id, node_ids, X_train, Y_train, model_class, T = args
#     # Number of nodes in this group
#     Nl = len(node_ids)
#     T_train = X_train.shape[0] // Nl  # Time steps per node used for training
#                                       # the last of which will be used as the init value of features. 
    
#     # Train one model per group
#     model=model_class()
#     model.fit(X_train, Y_train)
    
#     # Prepare initial inputs for forecasting
#     X_init = [
#         model.update(X_train[i], Y_train[i])
#         for i in range(Nl*(T_train-1),Nl*T_train)
#     ]
#     # Forecast for each node in the group
#     Y_pred_group = [model.forecast(X_init[i], T) for i in range(Nl)]
#     return Y_pred_group

def train_group(args):
    group_id, node_ids, X_train, Y_train, model_class = args
    # Number of nodes in this group

    # Train one model per group
    model=model_class()
    model.fit(X_train, Y_train)

    return model


class TreeForecastor:
    def __init__(self,NodeForecastor):
        self.NodeForecastor=NodeForecastor
        self.models=[]
        
    def fit(self,hts,node_levels,train_data):
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
        
        # Number of models = number of groups num_models = len(train_data)
        
        # Parallelize training and forecasting
        num_cores = cpu_count()
        with Pool(num_cores) as pool:
            args = [(group_id, node_ids, X, y, self.NodeForecastor) for group_id, node_ids, X, y in train_data]
            self.models = pool.map(train_group, args)
        

        # i0=hts.tree_structure.index(node_levels[-1])
        # for i in range(i0,len(hts.tree_structure)):
        #     p=hts.get_children_p(hts.tree_structure[i])   # p: (number of nodes, number of child nodes, time steps)
        #     Y_next=p[0]@sp.block_diag(Y_pred[0])
        #     for j in range(1,len(p)):
        #         Y_next=sp.vstack(Y_next,p[j]@sp.block_diag(Y_pred[j]))
        #     Y_pred=Y_next.copy()
            
        # SMatrix=hts.get_SMatrix()
        # whole_time_series=SMatrix@Y_pred
        # return hts.split_time_series(whole_time_series)
        return
    
    def forecast(self,X_init,T):
        Y_pred_group = [self.models[i].forecast(X_init[i], T) for i in range(len(X_init))]
        return Y_pred_group


# Example usage (assuming df and create_features are defined)
# df = pd.DataFrame({...})
# def create_features(df, levels): ...
# hts = HierarchicalTimeSeriesTree(df, ['location', 'dept', 'item'], create_features)
# model_template = NodeForecastor()
# forecasts = forecastHTS(hts, model_template, T=10, node_levels=['item'])
        
    

    
    
    
    
    
    
    
    
    
    
    