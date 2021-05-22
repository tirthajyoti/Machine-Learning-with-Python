import pandas as pd 
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

NUM_FEATURES = 10
NUM_SAMPLES = 1000

# Make data
def make_data():
    X,y = make_regression(n_samples=NUM_SAMPLES,n_features=NUM_FEATURES,
                      n_informative=NUM_FEATURES,noise=0.5)
    data = pd.DataFrame(X,columns=['X'+str(i) for i in range(1,NUM_FEATURES+1)],dtype=np.float16)
    data['y']=np.array(y,dtype=np.float16)
    return data

# Test/Train
def test_train(data):
    X_train,y_train = data.iloc[:int(NUM_SAMPLES/2)].drop(['y'],axis=1),data.iloc[:int(NUM_SAMPLES/2)]['y']
    X_test,y_test = data.iloc[int(NUM_SAMPLES/2):].drop(['y'],axis=1),data.iloc[int(NUM_SAMPLES/2):]['y']
    return (X_train,y_train,X_test,y_test)

# Fitting
def fitting(X_train,y_train):
    mlp = MLPRegressor(max_iter=50)
    mlp.fit(X_train,y_train)
    del X_train
    del y_train
    return mlp

# Saving model
def save(mlp):
    with open('MultiLayerPerceptron.sav',mode='wb') as f:
        pickle.dump(mlp,f)
        
def model_run(model,testfile):
    """
    Loads and runs a sklearn linear model
    """
    mlp = pickle.load(open(model, 'rb'))
    X_test = pd.read_csv(testfile)
    _= mlp.predict(X_test)
    return None

if __name__ == '__main__':
    data = make_data()
    X_train,y_train,X_test,y_test = test_train(data)
    X_test.to_csv("Test.csv",index=False)
    mlp = fitting(X_train,y_train)
    save(mlp)
    model_run('MultiLayerPerceptron.sav','Test.csv')