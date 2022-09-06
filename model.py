# Importing libraries

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from utils import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PowerTransformer, StandardScaler
import pickle

# Loading the training and test dataset

X = pd.read_csv('train.csv', index_col='id')
X_test = pd.read_csv('test.csv', index_col='id')

# Separting the independent and dependent features 

X.dropna(axis=0, subset=['rent'], inplace=True)
y = X.rent
y_train = y 
X.drop(['rent'], axis=1, inplace=True)



# We are only keeping the columns that have numerical features and the categorical features whose cardinality is less than 100
# As they are easy on the computers hardware when dealing with their encodings

numerical_cols = check_numerical(X)
categorical_cols = check_categorical(X)

my_cols = numerical_cols + categorical_cols
X_train = X[my_cols].copy()

X_test = X_test[my_cols].copy()



# We create a pipeline for numerical features which impute the missing values with median since they are robust to outliers
# We are also performing the scaling to bring down the data in the same range as the missing values

numerical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='median')), 
                                          ('sc', StandardScaler())])
# We are creating a pipeline whihc would impute the categorical data as the most frequently used value in the categorical features array
# and later we do the one hot encoding

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# combine the preprocessing of numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Initiating the process of hyperparameter tuning using grid search

model_cat = CatBoostRegressor()
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model_cat)
                             ])
# parameter grids
# parameters =  {
#               'model__learning_rate': [.01, .005], 
#               'model__max_depth': [5, 15],
#               'model__subsample': [0.4, 0.7],
#               'model__colsample_bytree': [0.7],
#               'model__n_estimators': [1000, 4000],
#               'model__reg_lambda':  [0.9]}

# xgb_grid = GridSearchCV(my_pipeline,parameters,cv = 2,n_jobs = 5,verbose=True)
# print("best parameters: ", xgb_grid.best_params_)

# Fitting the data to training pipeline
my_pipeline.fit(X_train,
         y_train)

# Using pickle to dave the proposed CatBoost model
with open('pickle_model', 'wb') as f:
    pickle.dump(my_pipeline, f)

# Making predictions on test dataset

with open('pickle_model', 'rb') as f:
    model = pickle.load(f)

# Making the predictions

prediction = model.predict(X_test)
preds = pd.DataFrame(prediction, columns=['rent'])
preds.to_csv('rent.csv', index = False) 
