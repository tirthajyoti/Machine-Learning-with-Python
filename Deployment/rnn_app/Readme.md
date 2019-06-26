## Web-app generating patent abstract like text using a Recurrent Neural Net (RNN)

### Requirements
* [Flask](http://flask.pocoo.org/)
* [Keras](http://keras.io/)
* TensorFlow
* Numpy
* [wtforms](https://wtforms.readthedocs.io/en/stable/)
* json
* [Jinja](http://jinja.pocoo.org/)

---

### How to run
I assume that you already have Python 3.6+ installed on your machine.<br>
Clone/copy all the files in this folder into a single folder on your computer.

**`$ pip install requirements.txt`**

Then type,

**`$ python keras_server.py`**

After this the server should start and be available at `http://localhost:5000`. Open your browser and go to this address to see the model output.

### Source
["Deploying a Keras Deep Learning Model as a Web Application in Python"](https://towardsdatascience.com/deploying-a-keras-deep-learning-model-as-a-web-application-in-p-fc0f2354a7ff) by [Will Kohersen](https://willk.online/)
