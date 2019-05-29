## Complexity and Learning curves
Modern enterprise of data science is powered at its core by advanced statistical modeling and machine learning (ML). Countless courses and articles teach budding data scientists about the concepts of training and test set, cross-validation, and computation of error metrics such as confusion matrix and F1 score. 

However, a good modeling does not end with achieving a high accuracy in the test set. It is just the start. There are tasks, often underappreciated, which can make a ML modeling robust and ready for production level scaling. Learners and practitioners of data science should imbue these tasks in their modeling pipeline as much as possible to project themselves as someone who cares not only about the algorithmic performance but also how the data science pipeline ultimately helps solving a business or scientific problem.

In this repo, we have notebooks outlining two such simple yet effective techniques and how they can be enmeshed with widely popular ML algorithms for modeling task.

### Learning and complexity curves

Complexity and learning curve analyses are essentially are part of the visual analytics that a data scientist must perform using the available dataset for comparing the merits of various ML algorithms. 

Often, a data scientist has a plethora of ML algorithms to choose from for a given business question and a specific data set. 

“Should I use logistic regression or k-nearest-neighbor for the classification task? How about a Support Vector Machine classifier? But what kernel to choose?” 

Ultimately, a lot of experiment with the data and algorithms are needed to construct a good methodology. That is the true spirit of the data science process i.e. experimentation with the core elements (the dataset and the processing algorithms) in a scientific manner. In support of that process, sampling and time complexity analyses can often guide a data scientist, in a systematic manner, to the choice of a suitable ML algorithm for the job at hand.

---

**Learning curve**: Graphs that compares the performance of a model on training and testing data over a varying number of training instances. 

We should generally see performance improve as the number of training points increases. 

**Complexity curve**: Graphs that show the model performance over training and validation set for varying degree of model complexity 
(e.g. degree of polynomial for linear regression, number of layers or neurons for neural networks, 
number of estimator trees for a Boosting algorithm or Random Forest)

Complexity curve allows us to verify when a model has learned as much as it can about the data without fitting to the noise. 
The optimum learning (given the fixed data) occurs when,

* The performances on the training and testing sets reach a plateau
* There is a consistent gap between the two error rates

The key is to find the sweet spot that minimizes bias and variance by finding the right level of model complexity.
Of course with more data any model can improve, and different models may be optimal.

### Following is the intuitive illustration of model complexity curve from Andrew Ng's Machine Learning course
![complexity_curve](https://raw.githubusercontent.com/tirthajyoti/PythonMachineLearning/master/Images/Complexity_curve_example.PNG)
