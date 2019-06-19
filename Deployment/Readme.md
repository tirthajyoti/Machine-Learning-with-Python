# Machine learning models deployment examples

Machine Learning models are often best served ... as an [RESTful API](https://restfulapi.net/) to be used by an external user. A little bit of web programming (starting a [Microservice](https://smartbear.com/solutions/microservices/) for example) is needed to wrap around your core ML model. We show examples of such ML API and how to create them in this section.

---

### [A linear regression example with US Housing data](https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Deployment/Linear_regression)
[Flask](http://flask.pocoo.org/) and [Gunicorn](https://gunicorn.org/) are used to run a simple HTTP server/interface. User can start the server, which functions as the model endpoint API, and then execute another Python script to send test data and request predictions from the pre-trained model.

---

### [A text generating app with recurrent neural network](https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Deployment/rnn_app)
A HTTP web app, complete with [WTForms](https://wtforms.readthedocs.io/en/stable/crash_course.html), CSS and aided by [Jinja templating](http://jinja.pocoo.org/docs/2.10/), is served to the user using [Flask](http://flask.pocoo.org/). Upon submitting the request via this web form, text is generated based on a pre-trained [recurrent neural network (RNN)](https://skymind.ai/wiki/lstm) with [word embedding](http://deeplearning.net/tutorial/rnnslu.html).
