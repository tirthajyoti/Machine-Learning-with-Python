"""Filename: hello_py.py
  """

from flask import Flask

app = Flask(__name__)

@app.route('/users/<string:username>')
def hello_py(username=None):
	return("Hello {}!".format(username))