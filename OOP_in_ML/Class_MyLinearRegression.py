import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class Metrics:
    """
    Methods for computing useful regression metrics
    
    sse: Sum of squared errors
    sst: Total sum of squared errors (actual vs avg(actual))
    r_squared: Regression coefficient (R^2)
    adj_r_squared: Adjusted R^2
    mse: Mean sum of squared errors
    AIC: Akaike information criterion
    BIC: Bayesian information criterion
    """

    def sse(self):
        """Returns sum of squared errors (model vs. actual)"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        squared_errors = (self.resid_) ** 2
        self.sq_error_ = np.sum(squared_errors)
        return self.sq_error_

    def sst(self):
        """Returns total sum of squared errors (actual vs avg(actual))"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        avg_y = np.mean(self.target_)
        squared_errors = (self.target_ - avg_y) ** 2
        self.sst_ = np.sum(squared_errors)
        return self.sst_

    def r_squared(self):
        """Returns calculated value of r^2"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        self.r_sq_ = 1 - self.sse() / self.sst()
        return self.r_sq_

    def adj_r_squared(self):
        """Returns calculated value of adjusted r^2"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        self.adj_r_sq_ = 1 - (self.sse() / self.dfe_) / (self.sst() / self.dft_)
        return self.adj_r_sq_

    def mse(self):
        """Returns calculated value of mse"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        self.mse_ = np.mean((self.predict(self.features_) - self.target_) ** 2)
        return self.mse_

    def aic(self):
        """
        Returns AIC (Akaike information criterion)
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.aic

    def bic(self):
        """
        Returns BIC (Bayesian information criterion)
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.bic

    def print_metrics(self):
        """Prints a report of the useful metrics for a given model object"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        items = (
            ("sse:", self.sse()),
            ("sst:", self.sst()),
            ("mse:", self.mse()),
            ("r^2:", self.r_squared()),
            ("adj_r^2:", self.adj_r_squared()),
            ("AIC:", self.aic()),
            ("BIC:", self.bic()),
        )
        for item in items:
            print("{0:8} {1:.4f}".format(item[0], item[1]))

    def summary_metrics(self):
        """Returns a dictionary of the useful metrics"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        metrics = {}
        items = (
            ("sse", self.sse()),
            ("sst", self.sst()),
            ("mse", self.mse()),
            ("r^2", self.r_squared()),
            ("adj_r^2:", self.adj_r_squared()),
            ("AIC:", self.aic()),
            ("BIC:", self.bic()),
        )
        for item in items:
            metrics[item[0]] = item[1]
        return metrics


class Inference:
    """
    Inferential statistics: 
        standard error, 
        p-values
        t-test statistics
        F-statistics and p-value of F-test
    """

    def __init__():
        pass

    def std_err(self):
        """
        Returns standard error values of the features
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.bse

    def pvalues(self):
        """
        Returns p-values of the features
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.pvalues

    def tvalues(self):
        """
        Returns t-test values of the features
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.tvalues

    def ftest(self):
        """
        Returns the F-statistic of the overall regression and corresponding p-value
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return (lm.fvalue, lm.f_pvalue)


class Diagnostics_plots:
    """
    Diagnostics plots and methods
    
    Arguments:
    fitted_vs_residual: Plots fitted values vs. residuals
    fitted_vs_features: Plots residuals vs all feature variables in a grid
    histogram_resid: Plots a histogram of the residuals (can be normalized)
    shapiro_test: Performs Shapiro-Wilk normality test on the residuals
    qqplot_resid: Creates a quantile-quantile plot for residuals comparing with a normal distribution    
    """

    def __init__():
        pass

    def fitted_vs_residual(self):
        """Plots fitted values vs. residuals"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        plt.title("Fitted vs. residuals plot", fontsize=14)
        plt.scatter(self.fitted_, self.resid_, edgecolor="k")
        plt.hlines(
            y=0,
            xmin=np.amin(self.fitted_),
            xmax=np.amax(self.fitted_),
            color="k",
            linestyle="dashed",
        )
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.show()

    def fitted_vs_features(self):
        """Plots residuals vs all feature variables in a grid"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        num_plots = self.features_.shape[1]
        if num_plots % 3 == 0:
            nrows = int(num_plots / 3)
        else:
            nrows = int(num_plots / 3) + 1
        ncols = 3
        fig, ax = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
        axes = ax.ravel()
        for i in range(num_plots, nrows * ncols):
            axes[i].set_visible(False)
        for i in range(num_plots):
            axes[i].scatter(
                self.features_.T[i],
                self.resid_,
                color="orange",
                edgecolor="k",
                alpha=0.8,
            )
            axes[i].grid(True)
            axes[i].set_xlabel("Feature X[{}]".format(i))
            axes[i].set_ylabel("Residuals")
            axes[i].hlines(
                y=0,
                xmin=np.amin(self.features_.T[i]),
                xmax=np.amax(self.features_.T[i]),
                color="k",
                linestyle="dashed",
            )
        plt.show()

    def histogram_resid(self, normalized=True):
        """Plots a histogram of the residuals (can be normalized)"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        if normalized:
            norm_r = self.resid_ / np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        num_bins = min(20, int(np.sqrt(self.features_.shape[0])))
        plt.title("Histogram of the normalized residuals")
        plt.hist(norm_r, bins=num_bins, edgecolor="k")
        plt.xlabel("Normalized residuals")
        plt.ylabel("Count")
        plt.show()

    def shapiro_test(self, normalized=True):
        """Performs Shapiro-Wilk normality test on the residuals"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        from scipy.stats import shapiro

        if normalized:
            norm_r = self.resid_ / np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        _, p = shapiro(norm_r)
        if p > 0.01:
            print("The residuals seem to have come from a Gaussian process")
        else:
            print(
                "The residuals does not seem to have come from a Gaussian process.\nNormality assumptions of the linear regression may have been violated."
            )

    def qqplot_resid(self, normalized=True):
        """Creates a quantile-quantile plot for residuals comparing with a normal distribution"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        from scipy.stats import probplot

        if normalized:
            norm_r = self.resid_ / np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        plt.title("Q-Q plot of the normalized residuals")
        probplot(norm_r, dist="norm", plot=plt)
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Residual quantiles")
        plt.show()


class Data_plots:
    """
    Methods for data related plots
    
    pairplot: Creates pairplot of all variables and the target
    plot_fitted: Plots fitted values against the true output values from the data
    """

    def __init__():
        pass

    def pairplot(self):
        """Creates pairplot of all variables and the target using the Seaborn library"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None

        print("This may take a little time. Have patience...")
        from seaborn import pairplot
        from pandas import DataFrame

        df = DataFrame(np.hstack((self.features_, self.target_.reshape(-1, 1))))
        pairplot(df)
        plt.show()

    def plot_fitted(self, reference_line=False):
        """
        Plots fitted values against the true output values from the data
        
        Arguments:
        reference_line: A Boolean switch to draw a 45-degree reference line on the plot
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        plt.title("True vs. fitted values", fontsize=14)
        plt.scatter(y, self.fitted_, s=100, alpha=0.75, color="red", edgecolor="k")
        if reference_line:
            plt.plot(y, y, c="k", linestyle="dotted")
        plt.xlabel("True values")
        plt.ylabel("Fitted values")
        plt.grid(True)
        plt.show()


class Outliers:
    """
    Methods for plotting outliers, leverage, influence points
    
    cook_distance: Computes and plots Cook's distance
    influence_plot: Creates the influence plot
    leverage_resid_plot: Plots leverage vs normalized residuals' square
    """

    def __init__():
        pass

    def cook_distance(self):
        """Computes and plots Cook\'s distance"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence as influence

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        inf = influence(lm)
        (c, p) = inf.cooks_distance
        plt.figure(figsize=(8, 5))
        plt.title("Cook's distance plot for the residuals", fontsize=14)
        plt.stem(np.arange(len(c)), c, markerfmt=",", use_line_collection=True)
        plt.grid(True)
        plt.show()

    def influence_plot(self):
        """Creates the influence plot"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        fig, ax = plt.subplots(figsize=(10, 8))
        fig = sm.graphics.influence_plot(lm, ax=ax, criterion="cooks")
        plt.show()

    def leverage_resid_plot(self):
        """Plots leverage vs normalized residuals' square"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        fig, ax = plt.subplots(figsize=(10, 8))
        fig = sm.graphics.plot_leverage_resid2(lm, ax=ax)
        plt.show()


class Multicollinearity:
    """
    Methods for checking multicollinearity in the dataset features
    
    vif:Computes variance influence factors for each feature variable
    """

    def __init__():
        pass

    def vif(self):
        """Computes variance influence factors for each feature variable"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as vif,
        )

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        for i in range(self.features_.shape[1]):
            v = vif(np.matrix(self.features_), i)
            print("Variance inflation factor for feature {}: {}".format(i, round(v, 2)))


class MyLinearRegression(
    Metrics, Diagnostics_plots, Data_plots, Outliers, Multicollinearity, Inference
):
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept_ = fit_intercept
        self.is_fitted = False
        self.features_ = None
        self.target_ = None

    def __repr__(self):
        return "I am a Linear Regression model!"

    def ingest_data(self, X, y):
        """
       Ingests the given data
        
        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # features and data
        self.features_ = X
        self.target_ = y

    def fit(self, X=None, y=None, fit_intercept_=True):
        """
        Fit model coefficients.
        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """

        if X != None:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            self.features_ = X
        if y != None:
            self.target_ = y

        # degrees of freedom of population dependent variable variance
        self.dft_ = self.features_.shape[0] - 1
        # degrees of freedom of population error variance
        self.dfe_ = self.features_.shape[0] - self.features_.shape[1] - 1

        # add bias if fit_intercept is True
        if self.fit_intercept_:
            X_biased = np.c_[np.ones(self.features_.shape[0]), self.features_]
        else:
            X_biased = self.features_
        # Assign target_ to a local variable y
        y = self.target_

        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self.fit_intercept_:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        # Predicted/fitted y
        self.fitted_ = np.dot(self.features_, self.coef_) + self.intercept_

        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals

        # Set is_fitted to True
        self.is_fitted = True

    def fit(self, X=None, y=None, fit_intercept_=True):
        """
        Fits model coefficients.
        
        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        fit_intercept: Boolean, whether an intercept term will be included in the fit
        """

        if X != None:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            self.features_ = X
        if y != None:
            self.target_ = y

        # degrees of freedom of population dependent variable variance
        self.dft_ = self.features_.shape[0] - 1
        # degrees of freedom of population error variance
        self.dfe_ = self.features_.shape[0] - self.features_.shape[1] - 1

        # add bias if fit_intercept is True
        if self.fit_intercept_:
            X_biased = np.c_[np.ones(self.features_.shape[0]), self.features_]
        else:
            X_biased = self.features_
        # Assign target_ to a local variable y
        y = self.target_

        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self.fit_intercept_:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        # Predicted/fitted y
        self.fitted_ = np.dot(self.features_, self.coef_) + self.intercept_

        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals

        # Set is_fitted to True
        self.is_fitted = True

    def fit_dataframe(self, X, y, dataframe, fit_intercept_=True):
        """
        Fit model coefficients from a Pandas DataFrame.
        
        Arguments:
        X: A list of columns of the dataframe acting as features. Must be only numerical.
        y: Name of the column of the dataframe acting as the target
        fit_intercept: Boolean, whether an intercept term will be included in the fit
        """

        assert (
            type(X) == list
        ), "X must be a list of the names of the numerical feature/predictor columns"
        assert (
            type(y) == str
        ), "y must be a string - name of the column you want as target"

        self.features_ = np.array(dataframe[X])
        self.target_ = np.array(dataframe[y])

        # degrees of freedom of population dependent variable variance
        self.dft_ = self.features_.shape[0] - 1
        # degrees of freedom of population error variance
        self.dfe_ = self.features_.shape[0] - self.features_.shape[1] - 1

        # add bias if fit_intercept is True
        if self.fit_intercept_:
            X_biased = np.c_[np.ones(self.features_.shape[0]), self.features_]
        else:
            X_biased = self.features_
        # Assign target_ to a local variable y
        y = self.target_

        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self.fit_intercept_:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        # Predicted/fitted y
        self.fitted_ = np.dot(self.features_, self.coef_) + self.intercept_

        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals

        # Set is_fitted to True
        self.is_fitted = True

    def predict(self, X):
        """Output model prediction.
        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.predicted_ = self.intercept_ + np.dot(X, self.coef_)
        return self.predicted_

    def run_diagnostics(self):
        """Runs diagnostics tests and plots"""
        Diagnostics_plots.fitted_vs_residual(self)
        Diagnostics_plots.histogram_resid(self)
        Diagnostics_plots.qqplot_resid(self)
        print()
        Diagnostics_plots.shapiro_test(self)

    def outlier_plots(self):
        """Creates various outlier plots"""
        Outliers.cook_distance(self)
        Outliers.influence_plot(self)
        Outliers.leverage_resid_plot(self)
