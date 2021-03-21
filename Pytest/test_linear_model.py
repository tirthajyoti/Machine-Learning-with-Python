# Pytest module for testing linear regression model function
# Dr. Tirthajyoti Sarkar, Fremont, CA


from joblib import load, dump
import numpy as np
from linear_model import train_linear_model
from os import path
import sklearn
from sklearn.model_selection import train_test_split
import pytest
import math


def random_data_constructor(noise_mag=1.0):
    """
    Random data constructor utility for tests
    """
    num_points = 100
    X = 10*np.random.random(size=num_points)
    y = 2*X+3+2*noise_mag*np.random.normal(size=num_points)
    return X,y

#-------------------------------------------------------------------

def fixed_data_constructor():
    """
    Fixed data constructor utility for tests
    """
    num_points = 100
    X = np.linspace(1,10,num_points)
    y = 2*X+3
    return X,y

#-------------------------------------------------------------------

def test_model_return_object():
    """
    Tests the returned object of the modeling function
    """
    X,y = random_data_constructor()
    scores = train_linear_model(X,y)
    
    #=================================
    # TEST SUITE
    #=================================
    # Check the return object type
    assert isinstance(scores, dict)
    # Check the length of the returned object
    assert len(scores) == 2
    # Check the correctness of the names of the returned dict keys
    assert 'Train-score' in scores and 'Test-score' in scores

#-------------------------------------------------------------------

def test_model_return_vals():
    """
    Tests for the returned values of the modeling function
    """
    X,y = random_data_constructor()
    scores = train_linear_model(X,y)
    
    #=================================
    # TEST SUITE
    #=================================
    # Check returned scores' type
    assert isinstance(scores['Train-score'], float)
    assert isinstance(scores['Test-score'], float)
    # Check returned scores' range
    assert scores['Train-score'] >= 0.0
    assert scores['Train-score'] <= 1.0
    assert scores['Test-score'] >= 0.0
    assert scores['Test-score'] <= 1.0

#-------------------------------------------------------------------

def test_model_save_load():
    """
    Tests for the model saving process
    """
    X,y = random_data_constructor()
    filename = 'testing'
    _ = train_linear_model(X,y, filename=filename)
        
    #=================================
    # TEST SUITE
    #=================================
    # Check the model file is created/saved in the directory
    assert path.exists('testing.sav')
    # Check that the model file can be loaded properly 
    # (by type checking that it is a sklearn linear regression estimator)
    loaded_model = load('testing.sav')
    assert isinstance(loaded_model, sklearn.linear_model._base.LinearRegression)

#-------------------------------------------------------------------

def test_loaded_model_works():
    """
    Tests if the loading of the model works correctly
    """
    X,y = fixed_data_constructor()
    if len(X.shape) == 1:
        X = X.reshape(-1,1)
    if len(y.shape) == 1:
        y = y.reshape(-1,1)
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)
    loaded_model = load('testing.sav')

    #=================================
    # TEST SUITE
    #=================================
    # Check that test and train scores are perfectly equal to 1.0
    assert scores['Train-score'] == 1.0
    assert scores['Test-score'] == 1.0
    # Check that trained model predicts the y (almost) perfectly given X
    # Note the use of np.testing function instead of standard 'assert'
    # To handle numerical precision issues, we should use the `assert_allclose` function instead of any equality check
    np.testing.assert_allclose(y,loaded_model.predict(X))

#-------------------------------------------------------------------

def test_model_works_data_range_sign_change():
    """
    Tests for functionality with data scaled high and low
    """
    # Small-valued data
    X,y = fixed_data_constructor()
    X = 1.0e-9*X
    y = 1.0e-9*y
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)
    
    # Check that test and train scores are perfectly equal to 1.0
    assert scores['Train-score'] == 1.0
    assert scores['Test-score'] == 1.0

    # Large-valued data
    X,y = fixed_data_constructor()
    X = 1.0e9*X
    y = 1.0e9*y
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)
    
    # Check that test and train scores are perfectly equal to 1.0
    assert scores['Train-score'] == 1.0
    assert scores['Test-score'] == 1.0

    # X-values are flipped
    X,y = fixed_data_constructor()
    X = -1 * X
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)
    
    # Check that test and train scores are perfectly equal to 1.0
    assert scores['Train-score'] == 1.0
    assert scores['Test-score'] == 1.0

    # y-values are flipped
    X,y = fixed_data_constructor()
    y = -1 * y
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)
    
    # Check that test and train scores are perfectly equal to 1.0
    assert scores['Train-score'] == 1.0
    assert scores['Test-score'] == 1.0

#-------------------------------------------------------------------

def test_noise_impact():
    """
    Tests functionality with low and high noise data and expected change in the R^2 score
    """
    X,y = random_data_constructor(noise_mag=0.5)
    filename = 'testing'
    scores_low_noise = train_linear_model(X,y, filename=filename)

    X,y = random_data_constructor(noise_mag=5.0)
    filename = 'testing'
    scores_high_noise = train_linear_model(X,y, filename=filename)

    # Check that R^2 scores from high-noise input is less than that of low-noise input
    assert scores_high_noise['Train-score'] < scores_low_noise['Train-score']
    assert scores_high_noise['Test-score'] < scores_low_noise['Test-score']

#-------------------------------------------------------------------

def test_additive_invariance():
    """
    Tests additive invariance 
    i.e. adding constant numbers to X or y array does not change the model coefficients
    """
    X,y = random_data_constructor(noise_mag=0.5)
    filename = 'testing'
    
    _ = train_linear_model(X,y, filename=filename)
    m = load('testing.sav')
    coeff_no_additive = float(m.coef_)

    X = X + 100
    _ = train_linear_model(X,y, filename=filename)
    m = load('testing.sav')
    coeff_X_additive = float(m.coef_)

    y = y - 100
    _ = train_linear_model(X,y, filename=filename)
    m = load('testing.sav')
    coeff_y_additive = float(m.coef_)

    # Check that model coefficients for default and additive data are same (very close)
    # Note the use of math.isclose function
    assert math.isclose(coeff_no_additive, coeff_X_additive, rel_tol=1e-6)
    assert math.isclose(coeff_no_additive, coeff_y_additive, rel_tol=1e-6)

#-------------------------------------------------------------------

def test_wrong_input_raises_assertion():
    """
    Tests for various assertion cheks written in the modeling function
    """
    X,y = random_data_constructor()
    filename = 'testing'
    scores = train_linear_model(X,y, filename=filename)

    #=================================
    # TEST SUITE
    #=================================
    # Test that it handles the case of: X is a string
    msg = train_linear_model('X',y)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "X must be a Numpy array"
    # Test that it handles the case of: y is a string
    msg = train_linear_model(X,'y')
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "y must be a Numpy array"
    # Test that it handles the case of: test_frac is a string
    msg = train_linear_model(X,y, test_frac='0.2')
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be a floating point number"
    # Test that it handles the case of: test_frac is within 0.0 and 1.0
    msg = train_linear_model(X,y, test_frac=-0.2)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be between 0.0 and 1.0"
    msg = train_linear_model(X,y, test_frac=1.2)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be between 0.0 and 1.0"
    # Test that it handles the case of: filename for model save a string
    msg = train_linear_model(X,y, filename = 2.0)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Filename must be a string"
    # Test that function is checking input vector shape compatibility
    X = X.reshape(10,10)
    msg = train_linear_model(X,y, filename='testing')
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Row numbers of X and y data must be identical"

#-------------------------------------------------------------------

def test_raised_exception():
    """
    Tests for raised exception with pytest.raises context manager
    """
    # ValueError
    with pytest.raises(ValueError):
        # Insert a np.nan into the X array
        X,y = random_data_constructor()    
        X[1] = np.nan
        filename = 'testing'
        scores = train_linear_model(X,y, filename=filename)
        # Insert a np.nan into the y array
        X,y = random_data_constructor()
        y[1] = np.nan
        filename = 'testing'
        scores = train_linear_model(X,y, filename=filename)
        
    with pytest.raises(ValueError) as exception:
        # Insert a string into the X array
        X,y = random_data_constructor()
        X[1] = "A string"
        filename = 'testing'
        scores = train_linear_model(X,y, filename=filename)
        assert "could not convert string to float" in str(exception.value)