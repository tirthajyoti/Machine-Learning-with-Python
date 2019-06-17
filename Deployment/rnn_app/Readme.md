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
Clone/copy all the files in this folder into a single folder on your computer and run.

**`$ python keras_server.py`**

**Special Note**: The `models` folder is currently empty. You have to download and keep the [model file from this link](https://github.com/WillKoehrsen/recurrent-neural-networks/blob/master/models/train-embeddings-rnn.h5?raw=true) in that folder.

### Source
["Deploying a Keras Deep Learning Model as a Web Application in Python"](https://towardsdatascience.com/deploying-a-keras-deep-learning-model-as-a-web-application-in-p-fc0f2354a7ff) by [Will Kohersen](https://willk.online/)
