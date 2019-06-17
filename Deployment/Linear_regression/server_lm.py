import os
import pandas as pd
import dill as pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
	"""
	API Call
	Pandas dataframe (sent as a payload) from API Call
	"""
	try:
		test_json = request.get_json()
		test = pd.read_json(test_json)
		print("The test data received are as follows...")
		print(test)
		print()

	except Exception as e:
		raise e

	clf = 'lm_model_v1.pk'
	
	if test.empty:
		return(bad_request())
	else:
		#Load the saved model
		print("Loading the model...")
		loaded_model = None
		with open('./models/'+clf,'rb') as f:
			loaded_model = pickle.load(f)

		print("The model has been loaded...doing predictions now...")
		print()
		predictions = loaded_model.predict(test)
			
		prediction_series = pd.Series(predictions)
		response = jsonify(prediction_series.to_json())
		response.status_code = 200
		return (response)

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp
