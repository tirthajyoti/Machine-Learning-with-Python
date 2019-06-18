#########################################
# Model Training file 
#########################################
"""
import numpy as np
import pandas as pd

import os 
import json
import io
import requests

import dill as pickle

#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

url="https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/USA_Housing.csv"
s=requests.get(url).content

df = pd.read_csv(io.StringIO(s.decode('utf-8')))
print("Dataset is loaded from the URL: {}".format(url))
#print(df.head())
print()

# Make a list of data frame column names
l_column = list(df.columns) # Making a list out of column names
len_feature = len(l_column) # Length of column vector list

# Put all the numerical features in X and Price in y, 
# Ignore Address which is string for linear regression
X = df[l_column[0:len_feature-2]]
y = df[l_column[len_feature-2]]

print("Feature set size:",X.shape)
print("Variable set size:",y.shape)
print()
print("Features variables: ",l_column[0:len_feature-2])
print()

# Create X and y train and test splits in one command using a split ratio and a random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print("Training feature set size:",X_train.shape)
print("Test feature set size:",X_test.shape)
print("Training variable set size:",y_train.shape)
print("Test variable set size:",y_test.shape)
print()

# Model fit and training
lm = LinearRegression() # Creating a Linear Regression object 'lm'
lm.fit(X_train,y_train)
print("Model training done...")
print()

# Print the intercept and coefficients of the linear model
print("The intercept term of the linear model:", round(lm.intercept_,3))
print("The coefficients of the linear model:", [round(c,3) for c in lm.coef_])
print()

# R-square coefficient
train_pred=lm.predict(X_train)
print("R-squared value of this fit (on the training set):",round(metrics.r2_score(y_train,train_pred),3))
# Test score
test_score=lm.score(X_test,y_test)
print("Test score: ",round(test_score,3))
print()

# Main (model saving in pickle format)
if __name__ == '__main__':
	filename = 'lm_model_v1.pk'
	print("Now saving the model to a serialized format (pickle)...")
	with open('models/'+filename, 'wb') as file:
		pickle.dump(lm, file)
	# Save some of the test data in a CSV
	print("Saving test data to a file...")
	print()
	X_test.to_csv("data/housing_test.csv")
	
