# Python Machine Learning Notebooks
Essential codes for jump-starting machine learning/data science with Python

## Essential tutorial-type notebooks on Pandas and Numpy
* Jupyter notebooks covering a wide range of functions and operations on the topics of NumPy, Pandans, Seaborn, matplotlib etc.

## Tutorial-type notebooks covering regression, classification, clustering, dimensionality reduction, and some basic neural network algorithms

### Regression
* Simple linear regression with t-statistic generation
* Multiple ways to do linear regression in Python and their speed comparison ([check the article I wrote on freeCodeCamp](https://medium.freecodecamp.org/data-science-with-python-8-ways-to-do-linear-regression-and-measure-their-speed-b5577d75f8b))
* Multi-variate regression with regularization
* Polynomial regression with how to use ***scikit-learn pipeline feature*** ([check the article I wrote on *Towards Data Science*](https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49))
* Decision trees and Random Forest regression (showing how the Random Forest works as a robust/regularized meta-estimator rejecting overfitting)

### Classification
* Logistic regression/classification
* _k_-nearest neighbor classification
* Decision trees and Random Forest Classification
* Support vector machine classification
* Naive Bayes classification

### Clustering
* _K_-means clustering
* Affinity propagation (showing its time complexity and the effect of damping factor)
* Mean-shift technique (showing its time complexity and the effect of noise on cluster discovery)
* Hierarchical clustering with Dendograms showing how to choose optimal number of clusters
* DBSCAN (showing how it can generically detect areas of high density irrespective of cluster shapes, which the k-means fails to do)

### Dimensionality reduction
* Principal component analysis

### Deep Learning/Neural Network
* Demo notebook to illustrate the superiority of deep neural network for complex nonlinear function approximation task.
* Step-by-step building of 1-hidden-layer and 2-hidden-layer dense network using basic TensorFlow methods 

## Basic interactive controls demo
* Demo on how to integrate basic interactive controls (slider bars, drop-down menus, check-boxes etc.) in a Jupyter notebook and use them for interactive machine learning task

## Run Jupyter using Docker

The https://github.com/machine-learning-helpers/docker-python-jupyter project builds a Docker image so that the (your) Jupyter notebooks can be run out-of-the-box on almost any platform in a few minutes.

It gives something like:

* Initialization of the Git repository for the Jupyter notebooks:
```
$ mkdir -p ~/dev/ml
$ cd ~/dev/ml
$ git clone https://github.com/tirthajyoti/PythonMachineLearning.git
```

Initialization of the Docker image to run those Jupyter notebooks:
```
$ docker pull artificialintelligence/python-jupyter
```
Usgae
```
$ cd ~/dev/ml/PythonMachineLearning
$ docker run -d -p 9000:8888 -v ${PWD}:/notebook -v ${PWD}:/data artificialintelligence/python-jupyter
```
And then you can open http://localhost:9000 in your browser.

Any modification to the notebooks may be committed to the Git repository (if you are registered as a contributor), and/or submitted as a pull request.
Shutdown
```
$ docker ps
CONTAINER ID        IMAGE                                   COMMAND                  CREATED             STATUS              PORTS                    NAMES
431b12a93ccf        artificialintelligence/python-jupyter   "/bin/sh -c 'jupyt..."   4 minutes ago       Up 4 minutes        0.0.0.0:9000->8888/tcp   friendly_euclid
$ docker kill 431b12a93ccf 
```
--------------------------------------------------------------------------------------------------------------------
**You can [add me on LinkedIn here](https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7/)**
