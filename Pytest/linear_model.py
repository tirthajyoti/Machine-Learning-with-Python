from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import load, dump
import numpy

def train_linear_model(X,y,
                       test_frac=0.2, 
                       filename='trained_linear_model'):
    """
    Trains a simple linear regression model with scikit-learn
    """

    try:
        assert isinstance(X,numpy.ndarray), "X must be a Numpy array"
        assert isinstance(y,numpy.ndarray), "y must be a Numpy array"
        assert isinstance(test_frac,float), "Test set fraction must be a floating point number"
        assert test_frac < 1.0, "Test set fraction must be between 0.0 and 1.0"
        assert test_frac > 0, "Test set fraction must be between 0.0 and 1.0"
        assert isinstance(filename, str), "Filename must be a string"
        assert X.shape[0] == y.shape[0], "Row numbers of X and y data must be identical"

        # Shaping
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        # Test/train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=42)
        # Instantiate
        model = LinearRegression()
        # Fit
        model.fit(X_train, y_train)
        # Save
        fname = filename+'.sav'
        dump(model, fname)
        # Compute scores
        r2_train = model.score(X_train,y_train)
        r2_test = model.score(X_test,y_test)
        # Return scores in a dictionary
        return {'Train-score':r2_train, 'Test-score': r2_test}
    
    except AssertionError as msg: 
        print(msg)
        return msg    