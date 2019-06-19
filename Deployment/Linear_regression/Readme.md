## How to serve a linear regression model on a web API using Flask and Gunicorn
I assume that you already have Python 3.6+ installed on your machine.<br>
Clone/copy all the files in this folder into a single folder on your computer.

#### Install required libraries

**`$ pip install requirements.txt`**

Then follow the steps below,

#### Run the training script,

**`$ python training_housing.py`**

This would do the following,
* Read the dataset from the `/data/USA_housing.csv`
* Train the linear regression model
* Save the model in a serialized format in the `/models/` folder
* Save the test dataset (sampled from the full dataset) in the `/data/housing_test.csv` file

#### Fire up the gunicorn server by running,

**`$ gunicorn --bind 0.0.0.0:5000 server_lm:app`**

This will start the HTTP server interface and run an `/predict` API endpoint.<br>
The exact address of this API is `http://localhost:5000/predict`.<br>

But we do not need to open any browser.
We will take advantage of this endpoint by passing data to it in JSON format from another script. 

#### Run the script sending test data to the API and requesting predictions

**`$ python request_pred.py`**

This should print the predicted values on your terminal.

#### Note 
For demo purpose, only 6 data points are passed on to the prediction server. **If you want to get preditcions for more points, simply edit the index of the `test_df` DataFrame in the following code segment in the `request_pred.py`.**

```
# Read test dataset
test_df = pd.read_csv("data/housing_test.csv")
# For demo purpose, only 6 data points are passed on to the prediction server endpoint.
# Feel free to change this index or use the whole test dataset
test_df = test_df.iloc[40:46]
```
