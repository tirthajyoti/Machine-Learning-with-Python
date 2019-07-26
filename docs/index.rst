.. image:: https://readthedocs.org/projects/machine-learning-with-python/badge/?version=latest
   :width: 20 %
.. image:: https://img.shields.io/badge/License-BSD%202--Clause-orange.svg
   :width: 20 %
   
Python Machine Learning Notebooks (Tutorial style)
==================================================

Dr. Tirthajyoti Sarkar, Fremont, CA (`Please feel free to add me on
LinkedIn
here <https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7>`__)

--------------

Requirements
===============

-  Python 3.5+
-  NumPy (``$ pip install numpy``)
-  Pandas (``$ pip install pandas``)
-  Scikit-learn (``$ pip install scikit-learn``)
-  SciPy (``$ pip install scipy``)
-  Statsmodels (``$ pip install statsmodels``)
-  MatplotLib (``$ pip install matplotlib``)
-  Seaborn (``$ pip install seaborn``)
-  Sympy (``$ pip install sympy``)

--------------

You can start with this article that I wrote in Heartbeat magazine (on
Medium platform): 

`“Some Essential Hacks and Tricks for Machine Learning
with
Python” <https://heartbeat.fritz.ai/some-essential-hacks-and-tricks-for-machine-learning-with-python-5478bc6593f2>`__

.. image:: https://cookieegroup.com/wp-content/uploads/2018/10/2-1.png"
   :width: 500px
   :align: center
   :height: 350px
   :alt: alternate text

Essential tutorial-type notebooks on Pandas and Numpy
=======================================================

Jupyter notebooks covering a wide range of functions and operations on
the topics of NumPy, Pandans, Seaborn, matplotlib etc.

-  `Basics of Numpy
   array <https://github.com/tirthajyoti/PythonMachineLearning/blob/master/Pandas%20and%20Numpy/Basics%20of%20Numpy%20arrays.ipynb>`__

-  `Basics of Pandas
   DataFrame <https://github.com/tirthajyoti/PythonMachineLearning/blob/master/Pandas%20and%20Numpy/Basics%20of%20Pandas%20DataFrame.ipynb>`__

-  `Basics of Matplotlib and Descriptive
   Statistics <https://github.com/tirthajyoti/PythonMachineLearning/blob/master/Pandas%20and%20Numpy/Basics%20of%20Matplotlib%20and%20Descriptive%20Statistics.ipynb>`__

--------------

Regression
===============
.. image:: https://slideplayer.com/slide/6053182/20/images/10/Simple+Linear+Regression+Model.jpg
   :width: 400px
   :align: center
   :height: 300px
   :alt: alternate text

-  Simple linear regression with t-statistic generation

-  `Multiple ways to perform linear regression in Python and their speed
   comparison <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Linear_Regression_Methods.ipynb>`__
   (`check the article I wrote on
   freeCodeCamp <https://medium.freecodecamp.org/data-science-with-python-8-ways-to-do-linear-regression-and-measure-their-speed-b5577d75f8b>`__)

-  `Multi-variate regression with
   regularization <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Multi-variate%20LASSO%20regression%20with%20CV.ipynb>`__

-  Polynomial regression using ***scikit-learn pipeline feature***
   (`check the article I wrote on *Towards Data
   Science* <https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49>`__)
-  Decision trees and Random Forest regression (showing how the Random
   Forest works as a robust/regularized meta-estimator rejecting
   overfitting)

-  `Detailed visual analytics and goodness-of-fit diagnostic tests for a
   linear regression
   problem <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Regression_Diagnostics.ipynb>`__

--------------

Classification
===============

.. image:: https://qph.fs.quoracdn.net/main-qimg-914b29e777e78b44b67246b66a4d6d71
   :width: 500px
   :align: center
   :height: 350px
   :alt: alternate text

-  Logistic regression/classification

-  *k*-nearest neighbor classification
-  Decision trees and Random Forest Classification
-  Support vector machine classification (`check the article I wrote
   in Towards Data Science on SVM and sorting
   algorithm <https://towardsdatascience.com/how-the-good-old-sorting-algorithm-helps-a-great-machine-learning-technique-9e744020254b>`__)

-  Naive Bayes classification

--------------

Clustering
===============

.. image:: https://i.ytimg.com/vi/IJt62uaZR-M/maxresdefault.jpg
   :width: 500px
   :align: center
   :height: 350px
   :alt: alternate text

-  *K*-means clustering
-  Affinity propagation (showing its time complexity and the effect of
   damping factor)
-  Mean-shift technique (showing its time complexity and the effect of
   noise on cluster discovery)
-  DBSCAN (showing how it can generically detect areas of high density
   irrespective of cluster shapes, which the k-means fails to do)
-  Hierarchical clustering with Dendograms showing how to choose optimal
   number of clusters

--------------

Dimensionality reduction
===========================

.. image:: https://i.ytimg.com/vi/QP43Iy-QQWY/maxresdefault.jpg
   :width: 500px
   :align: center
   :height: 350px
   :alt: alternate text

-  Principal component analysis

--------------

Deep Learning/Neural Network
==============================

-  `Demo notebook to illustrate the superiority of deep neural network
   for complex nonlinear function approximation
   task <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Function%20Approximation%20by%20Neural%20Network/Polynomial%20regression%20-%20linear%20and%20neural%20network.ipynb>`__
-  Step-by-step building of 1-hidden-layer and 2-hidden-layer dense
   network using basic TensorFlow methods

--------------

Random data generation using symbolic expressions
======================================================

-  How to use `Sympy package <https://www.sympy.org/en/index.html>`__ to
   generate random datasets using symbolic mathematical expressions.

-  Here is my article on Medium on this topic: `Random regression and
   classification problem generation with symbolic
   expression <https://towardsdatascience.com/random-regression-and-classification-problem-generation-with-symbolic-expression-a4e190e37b8d>`__

--------------

Simple deployment examples (serving ML models on web API)
============================================================

-  `Serving a linear regression model through a simple HTTP server
   interface <https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Deployment/Linear_regression>`__.
   User needs to request predictions by executing a Python script. Uses
   ``Flask`` and ``Gunicorn``.

-  `Serving a recurrent neural network (RNN) through a HTTP
   webpage <https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Deployment/rnn_app>`__,
   complete with a web form, where users can input parameters and click
   a button to generate text based on the pre-trained RNN model. Uses
   ``Flask``, ``Jinja``, ``Keras``/``TensorFlow``, ``WTForms``.

--------------

Object-oriented programming with machine learning
======================================================

Implementing some of the core OOP principles in a machine learning
context by `building your own Scikit-learn-like estimator, and making it
better <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/OOP_in_ML/Class_MyLinearRegression.ipynb>`__.

See my articles on Medium on this topic.

-  `Object-oriented programming for data scientists: Build your ML
   estimator <https://towardsdatascience.com/object-oriented-programming-for-data-scientists-build-your-ml-estimator-7da416751f64>`__
-  `How a simple mix of object-oriented programming can sharpen your
   deep learning
   prototype <https://towardsdatascience.com/how-a-simple-mix-of-object-oriented-programming-can-sharpen-your-deep-learning-prototype-19893bd969bd>`__

.. |License| image:: https://img.shields.io/badge/License-BSD%202--Clause-orange.svg
   :target: https://opensource.org/licenses/BSD-2-Clause
.. |GitHub forks| image:: https://img.shields.io/github/forks/tirthajyoti/Machine-Learning-with-Python.svg
   :target: https://github.com/tirthajyoti/Machine-Learning-with-Python/network
.. |GitHub stars| image:: https://img.shields.io/github/stars/tirthajyoti/Machine-Learning-with-Python.svg
   :target: https://github.com/tirthajyoti/Machine-Learning-with-Python/stargazers
