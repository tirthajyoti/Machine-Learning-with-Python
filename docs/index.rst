|License| |GitHub forks| |GitHub stars|

Python Machine Learning Notebooks (Tutorial style)
==================================================

### Dr. Tirthajyoti Sarkar, Fremont, CA (`Please feel free to add me on LinkedIn here <https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7>`__)
--------------------------------------------------------------------------------------------------------------------------------------------------

Requirements
~~~~~~~~~~~~

-  **Python 3.5+**
-  **NumPy (``$ pip install numpy``)**
-  **Pandas (``$ pip install pandas``)**
-  **Scikit-learn (``$ pip install scikit-learn``)**
-  **SciPy (``$ pip install scipy``)**
-  **Statsmodels (``$ pip install statsmodels``)**
-  **MatplotLib (``$ pip install matplotlib``)**
-  **Seaborn (``$ pip install seaborn``)**
-  .. rubric:: **Sympy (``$ pip install sympy``)**
      :name: sympy-pip-install-sympy

You can start with this article that I wrote in Heartbeat magazine (on
Medium platform): ### `“Some Essential Hacks and Tricks for Machine
Learning with
Python” <https://heartbeat.fritz.ai/some-essential-hacks-and-tricks-for-machine-learning-with-python-5478bc6593f2>`__

Essential tutorial-type notebooks on Pandas and Numpy
-----------------------------------------------------

Jupyter notebooks covering a wide range of functions and operations on
the topics of NumPy, Pandans, Seaborn, matplotlib etc.

`Basic Numpy operations <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Basics%20of%20Numpy%20arrays.ipynb>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Basic Pandas operations <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Basics%20of%20Pandas%20DataFrame.ipynb>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Basics of visualization with Matplotlib and descriptive stats <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Basics%20of%20Matplotlib%20and%20Descriptive%20Statistics.ipynb>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Advanced Pandas operations <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Advanced%20Pandas%20Operations.ipynb>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`How to read various data sources <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Read_data_various_sources/How%20to%20read%20various%20sources%20in%20a%20DataFrame.ipynb>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`PDF reading and table processing demo <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Read_data_various_sources/PDF%20table%20reading%20and%20processing%20demo.ipynb>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`How fast are Numpy operations compared to pure Python code? <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/How%20fast%20are%20NumPy%20ops.ipynb>`__ (Read my `article <https://towardsdatascience.com/why-you-should-forget-for-loop-for-data-science-code-and-embrace-vectorization-696632622d5f>`__ on Medium related to this topic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Fast reading from Numpy using .npy file format <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Numpy_Reading.ipynb>`__ (Read my `article <https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161>`__ on Medium on this topic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tutorial-type notebooks covering regression, classification, clustering, dimensionality reduction, and some basic neural network algorithms
-------------------------------------------------------------------------------------------------------------------------------------------

Regression
~~~~~~~~~~

-  Simple linear regression with t-statistic generation

-  `Multiple ways to perform linear regression in Python and their speed
   comparison <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Linear_Regression_Methods.ipynb>`__
   (`check the article I wrote on
   freeCodeCamp <https://medium.freecodecamp.org/data-science-with-python-8-ways-to-do-linear-regression-and-measure-their-speed-b5577d75f8b>`__)

-  `Multi-variate regression with
   regularization <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Multi-variate%20LASSO%20regression%20with%20CV.ipynb>`__

-  Polynomial regression using **scikit-learn pipeline feature** (`check
   the article I wrote on Towards Data
   Science <https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49>`__)

-  Decision trees and Random Forest regression (showing how the Random
   Forest works as a robust/regularized meta-estimator rejecting
   overfitting)

-  `Detailed visual analytics and goodness-of-fit diagnostic tests for a
   linear regression
   problem <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Regression_Diagnostics.ipynb>`__

--------------

Classification
~~~~~~~~~~~~~~

-  Logistic regression/classification

-  *k*-nearest neighbor classification

-  Decision trees and Random Forest Classification

-  Support vector machine classification (`check the article I wrote in
   Towards Data Science on SVM and sorting
   algorithm <https://towardsdatascience.com/how-the-good-old-sorting-algorithm-helps-a-great-machine-learning-technique-9e744020254b>`__\ **)**

-  Naive Bayes classification

--------------

Clustering
~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

-  Principal component analysis

--------------

Deep Learning/Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `Demo notebook to illustrate the superiority of deep neural network
   for complex nonlinear function approximation
   task <https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Function%20Approximation%20by%20Neural%20Network/Polynomial%20regression%20-%20linear%20and%20neural%20network.ipynb>`__
-  Step-by-step building of 1-hidden-layer and 2-hidden-layer dense
   network using basic TensorFlow methods

--------------

Random data generation using symbolic expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  How to use `Sympy package <https://www.sympy.org/en/index.html>`__ to
   generate random datasets using symbolic mathematical expressions.

-  .. rubric:: Here is my article on Medium on this topic: `Random
      regression and classification problem generation with symbolic
      expression <https://towardsdatascience.com/random-regression-and-classification-problem-generation-with-symbolic-expression-a4e190e37b8d>`__
      :name: here-is-my-article-on-medium-on-this-topic-random-regression-and-classification-problem-generation-with-symbolic-expression

Simple deployment examples (serving ML models on web API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
