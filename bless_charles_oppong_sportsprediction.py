# -*- coding: utf-8 -*-
"""Bless_Charles_Oppong_SportsPrediction

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yshcEVVf0l1wCRfB-EUQ-oWRJf8Tg91w
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')

"""Question 1
Demonstrating data preparation and feature extraction process
"""

#code to load the training dataset
players=pd.read_csv('/content/drive/My Drive/male_players (legacy) (1).csv')
players.head()

#code to drop some of the columns that are not needed to predict a players overall rating
players.drop(['body_type','work_rate','player_positions','real_face','value_eur','wage_eur','league_id','club_loaned_from','nation_position','player_face_url','fifa_update_date','fifa_update','fifa_version','long_name','player_url','player_id','dob','ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb','gk','short_name','club_team_id','club_joined_date','nationality_id','nation_team_id','nation_jersey_number','player_tags','player_traits'],axis=1,inplace=True)
players.head()

#code to drop the columns that have missing values greater than 30%
threshold = 0.3
values_missing= players.isnull().mean()
column= values_missing[values_missing > threshold].index
#code to drop the columns
players = players.drop(columns=column)
players.head()

#code to identify all the categorical columns that the dataset has
non_numeric=players.select_dtypes(exclude=np.number)

#code to encode and impute the selected columns
non_numeric=pd.get_dummies(non_numeric).astype(int)
non_numeric=non_numeric.ffill()
#non_numeric.head()
non_numeric.isnull().sum().sum()

#code to get the numeric columns in the dataset
numeric=players.select_dtypes(include=np.number)

#code to impute the missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp=IterativeImputer(max_iter=5,random_state=0)
numeric=pd.DataFrame(np.round(imp.fit_transform(numeric)),columns=numeric.columns)

numeric.isnull().sum().sum()

import pandas as pd
#code to join the numeric and non_numeric columns which have been cleaned
new_players=pd.concat([numeric,non_numeric],axis=1)
#new_players.head()
#code to check that there are no NAN values in the dataset
new_players.isnull().sum().sum()

#code to pick the dependent variable from the dataset
y=new_players['overall']
x=new_players.drop('overall',axis=1)

from sklearn.preprocessing import StandardScaler

#code to scale the data
scale=StandardScaler()
scaled=scale.fit_transform(x)

x=pd.DataFrame(scaled,columns=x.columns)

"""Question 2
Feature subsets that show maximum correlation with the dependent variable
"""

#code to find the correlation between the other features and the dependent variable
correlation = x.corrwith(y)

correlation.sort_values(ascending= False)

#code to select the features that have a correlation with the dependent variable aboe 0.4 or less than 0.4
possible_features = correlation[correlation.abs()>0.4].index
x=x[possible_features]
x.head()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

"""Question 3:
Creating and training a suitable machine learning model with cross-validation
"""

#code to train using xgboost
cv=KFold(n_splits=3)
parameters = {
    "subsample":[0.5, 0.75, 1],
    "max_depth": [2,5, 6, 12],
    "learning_rate": [0.3, 0.1, 0.03],
    "n_estimators":[100,500,1000]
}

xg= xgb.XGBRegressor(n_jobs=-1)

cv_gridsearch= GridSearchCV(xg, param_grid=parameters, scoring='neg_mean_squared_error', cv=cv)
cv_gridsearch.fit(xtrain, ytrain)

chosen_model=cv_gridsearch.best_estimator_
chosen_model
# code to predict
predictions_XG = chosen_model.predict(xtest)
predictions_XG

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

"""Question 4:
Measuring a model's performance
"""

#code to find the mean squared error
mse_XG=mean_squared_error(ytest,predictions_XG)
mse_XG

#code to find the mean absolute error
mae_XG=mean_absolute_error(ytest,predictions_XG)
mae_XG

#code to train using linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)
predictions_R=lr.predict(xtest)

#code to find the mean squared error
mse_lr=mean_squared_error(ytest,predictions_R)
mse_lr

#code to find the mean absolute error
mae_lr=mean_absolute_error(ytest,predictions_R)
mae_lr

#code to train using gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

gb=GradientBoostingRegressor()

parameters_gb ={
    "max_depth":[3,4,5],
    "learning_rate":[0.01,0.1,0.2],
    "n_estimators":[100,200,300]}

cv_gridsearchG=GridSearchCV(gb,parameters_gb,cv=5,scoring="neg_mean_squared_error",n_jobs=-1)
cv_gridsearchG.fit(xtrain,ytrain)
possibleModel=cv_gridsearchG.best_estimator_
possibleModel

# code to predict
predictions_GB = possibleModel.predict(xtest)

#code to calculate the mean squared error
mse_GB=mean_squared_error(ytest,predictions_GB)
mse_GB

#code to calculate the mean absolute error
mae_GB=mean_absolute_error(ytest,predictions_GB)
mae_GB

"""Question 5:
Testing the model using the players_22 dataset
"""

#code to load the testing dataset
playerstest =pd.read_csv('/content/drive/My Drive/players_22.csv')
playerstest.head()

#code to drop unneccessay columns from the testing dataset
playerstest.drop(['sofifa_id','player_url','short_name','long_name','player_positions','value_eur','wage_eur','dob','club_team_id','club_position','club_loaned_from','club_joined','nationality_id','nation_position','work_rate','body_type','real_face','player_tags','player_traits','ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb','gk','player_face_url','club_logo_url','club_flag_url','nation_logo_url','nation_flag_url'],axis=1,inplace=True)
playerstest.head()

#code to drop columns that have missing values above the 30% threshold for the testing data
threshold = 0.3
values_missing= playerstest.isnull().mean()
column= values_missing[values_missing > threshold].index
#code to drop the columns
playerstest = playerstest.drop(columns=column)
playerstest.head()

#code to get the non numeric data from the dataset
non_numeric_test=playerstest.select_dtypes(exclude=np.number)

#code to encode and impute the non numeric columns in the dataset
non_numeric_test=pd.get_dummies(non_numeric_test).astype(int)
non_numeric_test=non_numeric_test.ffill()
#non_numeric.head()
non_numeric_test.isnull().sum().sum()

#code to get the numeric columns in the data set
numeric_test=playerstest.select_dtypes(include=np.number)

#code to impute the missing values in the numeric columns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp=IterativeImputer(max_iter=3,random_state=0)
numeric_test=pd.DataFrame(np.round(imp.fit_transform(numeric_test)),columns=numeric_test.columns)

numeric_test.isnull().sum().sum()

import pandas as pd
#code to join the numeric and non numeric columns
new_players_test=pd.concat([numeric_test,non_numeric_test],axis=1)
#new_players.head()
#code to check if there are still missing values in the dataset
new_players_test.isnull().sum().sum()

#code to pick the dependent variable
y_test=new_players_test['overall']

#code to extract the features with high correlation from the players 22 dataset based on observations in the male players dataset
#code will also take corresponding features from the training and testing dataset
new_test=new_players_test[possible_features]

new_test.head()

from sklearn.preprocessing import StandardScaler

#scale the data
sc = StandardScaler()
scaled=sc.fit_transform(new_test)

new_test=pd.DataFrame(scaled,columns=new_test.columns)

#code to make predictions on the testing data
predictions_XGB = chosen_model.predict(new_test)

#code to calculate the mean absolute error
mae_XGB=mean_absolute_error(y_test,predictions_XGB)
mae_XGB

#code to calculate the mean squared error
mse_XGB=mean_squared_error(y_test,predictions_XGB)
mse_XGB

"""Question 6: Deploy the model"""

#code to save the model in a pickle file
import pickle as pkl
XG_model = chosen_model

file_path="C:\\Users\\user\\Downloads"
with open(file_path, 'wb') as file:
    pkl.dump(XG_model, file)

