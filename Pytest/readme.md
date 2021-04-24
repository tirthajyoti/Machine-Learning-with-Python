## A sample Pytest module for a Scikit-learn model training function

![scheme](https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Pytest/Overall-scheme.png)

### How to run Pytest

- Install pytest `pip install pytest`

- Copy/clone the two Python scripts from this directory
- The `linear_model.py` has a single function that trains a simple linear regression model using scikit-learn. Note that it has basic assertion tests and `try-except` construct to handle potential input errors.
- The `test_linear_model.py` file is the test module which acts as the input to the Pytest program.
- Run `pytest test_linear_model.py -v` on your terminal to run the tests. You should see something like following,

```
======================================================================================================= test session starts ======================================================================================================== 
platform win32 -- Python 3.9.1, pytest-6.2.2, py-1.10.0, pluggy-0.13.1 -- c:\program files\python39\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\TirthajyotiSarkar\Documents\Python Notebooks\Pytest
plugins: anyio-2.0.2
collected 9 items                                                                                                                                                                                                                    

test_linear_model.py::test_model_return_object PASSED                                                                                                                                                                         [ 11%] 
test_linear_model.py::test_model_return_vals PASSED                                                                                                                                                                           [ 22%] 
test_linear_model.py::test_model_save_load PASSED                                                                                                                                                                             [ 33%] 
test_linear_model.py::test_loaded_model_works PASSED                                                                                                                                                                          [ 44%] 
test_linear_model.py::test_model_works_data_range_sign_change PASSED                                                                                                                                                          [ 55%] 
test_linear_model.py::test_noise_impact PASSED                                                                                                                                                                                [ 66%] 
test_linear_model.py::test_additive_invariance PASSED                                                                                                                                                                         [ 77%] 
test_linear_model.py::test_wrong_input_raises_assertion PASSED                                                                                                                                                                [ 88%] 
test_linear_model.py::test_raised_exception PASSED                                                                                                                                                                            [100%] 

========================================================================================================= warnings summary ========================================================================================================= 
..\..\..\..\..\..\program files\python39\lib\site-packages\win32\lib\pywintypes.py:2
  c:\program files\python39\lib\site-packages\win32\lib\pywintypes.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
    import imp, sys, os

-- Docs: https://docs.pytest.org/en/stable/warnings.html
=================================================================================================== 9 passed, 1 warning in 0.96s =================================================================================================== 
```

### What does it mean?

- The terminal message (above) indicates that 9 tests were run (corresponding to the 9 functions in the `test_linear_model.py` module) and all of them passed.

- It also shows the order of the tests run (this is because you included the `- v` argument on the command line while running `pytest` command). Pytest allows you to randomize the testing sequence but that discussion is for another day.

### Notes on the test module

- Note, how the `test_linear_model.py` contains 9 functions with names starting with `test...`. Those contain the actual test code. 
- It also has a couple of data constructor functions (`random_data_constructor` and `fixed_data_constructor`) whose names do not start with `test...` and they are ignored by Pytest. The `random_data_constructor` even takes a `noise_mag` argument which is used to control the magnitude of noise to test the expected behavior of a linear regression algorithm. Refer to the `test_noise_impact` function for this.

- Note that we need to import a variety of libraries to test all kind of things e.g. we imported libraries like `joblib`, `os`, `sklearn`, `numpy`, and of course, the `train_linear_model` function from the `linear_model` module.
- Note the **clear and distinctive names** for the testing functions e.g. `test_model_return_object()` which only checks the returned object from the `train_linear_model` function, or the `test_model_save_load()` which checks whether the saved model can be loaded properly (but does not try to make predictions or anything). **Always write short and crisp test functions with a singular focus**. 
- For checking the predictions i.e. whether the trained model really works or not, we have the `test_loaded_model_works()` function which uses a fixed data generator with no noise (as compared to other cases, where we can use a random data generator with random noise). It passes on the fixed `X` and `y` data, loads the trained model, checks if the R^2 ([regression coefficient](https://www.geeksforgeeks.org/python-coefficient-of-determination-r2-score/)) scores are perfectly equal to 1.0 (true for a fixed dataset with no noise) and then compare the model predictions with the original ground truth `y` vector. 
- Note, how the aforementioned function uses a **special Numpy testing function `np.testing.assert_allclose` instead of the regular `assert` statement**. This is to avoid any potential numerical precision issues associated with the model data i.e. Numpy arrays and the prediction algorithm involving linear algebra operations.
- The `test_model_works_data_range_sign_change` function tests the expected behavior of a linear regression estimator - that the regression scores will still be 1.0 no matter the range of the data (scaling the data by 10e-9 or 10e+9). It also changes the expected behavior if the data flips sign somehow.
- The `test_additive_invariance` function tests the additive invariance property of the linear regression estimator. **Similarly, you should think about the special properties of your ML estimator and write customized tests for them**.
- Take a look at the `test_wrong_input_raises_assertion` function which tests **if the original modeling function raises the correct Exception and messages** when fed with various types of wrong input arguments. Always remember that **Pytest (or any testing module for that matter) is not to be used for error-handling** (e.g. a wrong input type). The developer (ML sofware engineer in this case) must write the necessary error-checking and handling code. In this example, we write a bunch of `assert` statements in the original `train_linear_model` function and wrap them around a whole `try...except` block. The test code only checks if we are returning the `AssertionType` error for those wrong input cases and if the correct error message is being printed for the end user.
- Finally, the `test_raised_exception` function tests the **rise of other exception types (i.e. different from the `AssertionError` that could be raised by the `assert` statements in the function module)** during runtime using a special Pytest feature - [the `pytest.raises` context manager](https://docs.pytest.org/en/reorganize-docs/new-docs/user/pytest_raises.html).
