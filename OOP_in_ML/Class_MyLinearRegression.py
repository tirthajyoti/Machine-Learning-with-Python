import numpy as np
import matplotlib.pyplot as plt

class Metrics:
              
    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.resid_) ** 2
        self.sq_error_ = np.sum(squared_errors)
        return self.sq_error_
        
    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.target_)
        squared_errors = (self.target_ - avg_y) ** 2
        self.sst_ = np.sum(squared_errors)
        return self.sst_
    
    def r_squared(self):
        '''returns calculated value of r^2'''
        self.r_sq_ = 1 - self.sse()/self.sst()
        return self.r_sq_
    
    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2'''
        self.adj_r_sq_ = 1 - (self.sse()/self.dfe_) / (self.sst()/self.dft_)
        return self.adj_r_sq_
    
    def mse(self):
        '''returns calculated value of mse'''
        self.mse_ = np.mean( (self.predict(self.features_) - self.target_) ** 2 )
        return self.mse_
    
    def pretty_print_stats(self):
        '''returns report of statistics for a given model object'''
        items = ( ('sse:', self.sse()), ('sst:', self.sst()), 
                 ('mse:', self.mse()), ('r^2:', self.r_squared()), 
                  ('adj_r^2:', self.adj_r_squared()))
        for item in items:
            print('{0:8} {1:.4f}'.format(item[0], item[1]))


class Diagnostics_plots:
    
    def __init__():
        pass
    
    def fitted_vs_residual(self):
        '''Plots fitted values vs. residuals'''
        plt.title("Fitted vs. residuals plot",fontsize=14)
        plt.scatter(self.fitted_,self.resid_,edgecolor='k')
        plt.hlines(y=0,xmin=np.amin(self.fitted_),xmax=np.amax(self.fitted_),color='k',linestyle='dashed')
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.show()
    
    def fitted_vs_features(self):
        '''Plots residuals vs all feature variables in a grid'''
        num_plots = self.features_.shape[1]
        if num_plots%3==0:
            nrows = int(num_plots/3)
        else:
            nrows = int(num_plots/3)+1
        ncols = 3
        fig, ax = plt.subplots(nrows, ncols, figsize=(15,nrows*3.5))
        axes = ax.ravel()
        for i in range(num_plots,nrows*ncols):
            axes[i].set_visible(False)
        for i in range(num_plots):
            axes[i].scatter(self.features_.T[i],self.resid_,color='orange',edgecolor='k',alpha=0.8)
            axes[i].grid(True)
            axes[i].set_xlabel("Feature X[{}]".format(i))
            axes[i].set_ylabel("Residuals")
            axes[i].hlines(y=0,xmin=np.amin(self.features_.T[i]),xmax=np.amax(self.features_.T[i]),
                           color='k',linestyle='dashed')
        plt.show()
        
    def histogram_resid(self,normalized=True):
        '''Plots a histogram of the residuals (can be normalized)'''
        if normalized:
            norm_r=self.resid_/np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        num_bins=min(20,int(np.sqrt(self.features_.shape[0])))
        plt.title("Histogram of the normalized residuals")
        plt.hist(norm_r,bins=num_bins,edgecolor='k')
        plt.xlabel("Normalized residuals")
        plt.ylabel("Count")
        plt.show()
    
    def shapiro_test(self,normalized=True):
        '''Performs Shapiro-Wilk normality test on the residuals'''
        from scipy.stats import shapiro
        if normalized:
            norm_r=self.resid_/np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        _,p = shapiro(norm_r)
        if p > 0.01:
            print("The residuals seem to have come from a Gaussian process")
        else:
            print("The residuals does not seem to have come from a Gaussian process.\nNormality assumptions of the linear regression may have been violated.")
        
    def qqplot_resid(self,normalized=True):
        '''Creates a quantile-quantile plot for residuals comparing with a normal distribution'''
        from scipy.stats import probplot
        if normalized:
            norm_r=self.resid_/np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        plt.title("Q-Q plot of the normalized residuals")
        probplot(norm_r,dist='norm',plot=plt)
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Residual quantiles")
        plt.show()


class Data_plots:
    
    def __init__():
        pass
    
    def pairplot(self):
        '''Creates pairplot of all variables and the target using the Seaborn library'''
        
        print ("This may take a little time. Have patience...")
        from seaborn import pairplot
        from pandas import DataFrame
        df = DataFrame(np.hstack((self.features_,self.target_.reshape(-1,1))))
        pairplot(df)
        plt.show()
    
    def plot_fitted(self,reference_line=False):
        """
        Plots fitted values against the true output values from the data
        
        Arguments:
        reference_line: A Boolean switch to draw a 45-degree reference line on the plot
        """
        plt.title("True vs. fitted values",fontsize=14)
        plt.scatter(y,self.fitted_,s=100,alpha=0.75,color='red',edgecolor='k')
        if reference_line:
            plt.plot(y,y,c='k',linestyle='dotted')
        plt.xlabel("True values")
        plt.ylabel("Fitted values")
        plt.grid(True)
        plt.show()


class Outliers:
    
    def __init__():
        pass
    
    def cook_distance(self):
        '''Computes and plots Cook\'s distance'''
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence as influence
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        inf=influence(lm)
        (c, p) = inf.cooks_distance
        plt.figure(figsize=(8,5))
        plt.title("Cook's distance plot for the residuals",fontsize=14)
        plt.stem(np.arange(len(c)), c, markerfmt=",", use_line_collection=True)
        plt.grid(True)
        plt.show()
    
    def influence_plot(self):
        '''Creates the influence plot'''
        import statsmodels.api as sm
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        fig, ax = plt.subplots(figsize=(10,8))
        fig = sm.graphics.influence_plot(lm, ax= ax, criterion="cooks")
        plt.show()
    
    def leverage_resid_plot(self):
        '''Plots leverage vs normalized residuals' square'''
        import statsmodels.api as sm
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        fig, ax = plt.subplots(figsize=(10,8))
        fig = sm.graphics.plot_leverage_resid2(lm, ax= ax)
        plt.show()


class Multicollinearity:
    
    def __init__():
        pass
    
    def vif(self):
        '''Computes variance influence factors for each feature variable'''
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        for i in range(self.features_.shape[1]):
            v=vif(np.matrix(self.features_),i)
            print("Variance inflation factor for feature {}: {}".format(i,round(v,2)))
        
class MyLinearRegression(Metrics, Diagnostics_plots,Data_plots,Outliers,Multicollinearity):
    
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept
    
    def __repr__(self):
        return "I am a Linear Regression model!"
    
    def fit(self, X, y):
        """
        Fit model coefficients.

        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """
        
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # features and data
        self.features_ = X
        self.target_ = y
        
        # degrees of freedom of population dependent variable variance
        self.dft_ = X.shape[0] - 1   
        # degrees of freedom of population error variance
        self.dfe_ = X.shape[0] - X.shape[1] - 1
            
        # add bias if fit_intercept is True
        if self._fit_intercept:
            X_biased = np.c_[np.ones(X.shape[0]), X]
        else:
            X_biased = X
        
        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)
        
        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef
            
        # Predicted/fitted y
        self.fitted_ = np.dot(X,self.coef_) + self.intercept_
        
        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals
    
    def predict(self, X):
        """Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        self.predicted_ = self.intercept_ + np.dot(X, self.coef_)
        return self.predicted_
    
    def run_diagnostics(self):
        '''Runs diagnostics tests and plots'''
        Diagnostics_plots.fitted_vs_residual(self)
        Diagnostics_plots.histogram_resid(self)
        Diagnostics_plots.qqplot_resid(self)
        print()
        Diagnostics_plots.shapiro_test(self)
    
    def outlier_plots(self):
        '''Creates various outlier plots'''
        Outliers.cook_distance(self)
        Outliers.influence_plot(self)
        Outliers.leverage_resid_plot(self)