'''linear_regression.py
Subclass of Analysis that performs linear regression on data
YOUR NAME HERE
CS251 Data Analysis Visualization
Spring 2022
'''
from os import X_OK
import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None
        
        self.yOrig = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.m_sse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        
        # intercept: float. Regression intercept
        self.intercept = None
        
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None
        
        self.c = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        
        ind = self.data.get_header_indices(ind_vars)
        self.A = self.data.get_all_data()[:,ind]
        self.A = np.array(self.A,dtype=np.float32)
        
        shape = np.shape(self.A)
        Ahat = np.hstack(( np.ones((shape[0],1)), self.A))
        
        ind = self.data.get_ind(self.dep_var)
        y = self.data.get_all_data()[:,ind]
        shape = np.shape(y)
        y = y.reshape(shape[0],1)
        self.yOrig = y
        
        c, _, _, _ = sc.lstsq( Ahat, y )
        self.c = c
        self.intercept = c[0,0]
        self.slope = c[1:]
        
        self.y = self.predict()
        self.residuals = self.compute_residuals(self.y)
        self.R2 = self.r_squared(self.y)
        self.m_sse = self.mean_sse()   

    def predict(self, X=[]):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X==[]:
            X = self.A   
        
        pred = self.slope.T * X
        pred = np.sum (pred, axis = 1)
        pred += self.intercept
        shape = np.shape(pred)
        pred = pred.reshape(shape[0],1)
        
        return pred
            
    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        resid = self.compute_residuals(y_pred)
        
        ind = self.data.get_ind(self.dep_var)
        y = self.data.get_all_data()[:,ind]
        sh = np.shape(y)
        y = y.reshape(sh[0],1)
        
        R2 = 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)
        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        ind = self.data.get_ind(self.dep_var)
        y = self.data.get_all_data()[:,ind]
        sh = np.shape(y)
        y = y.reshape(sh[0],1)
        
        return y - y_pred

    def mean_sse(self):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        shape = np.shape(self.A)
        return np.sum(self.residuals**2)/shape[0]

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        
        x = self.A
        
        xline = np.linspace(x.min(),x.max(),100).reshape((100,1))
        Bhat =  np.hstack( (np.ones((100,1)), xline))
        
        #print(np.shape(Bhat))
        
        rline = Bhat @ self.c
        
        plt.plot( xline, rline, 'r')
        
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.scatter(x,self.yOrig)
        plt.title( title + f" R^2 = {self.R2:0.2f}" ) 

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        dataSet = self.data.select_data(data_vars)
        
        fig, axs = plt.subplots(len(data_vars), len(data_vars),figsize=fig_sz)
        fig.tight_layout()
        
        if hists_on_diag == True:

            for xInd in range(len(data_vars)):
                x = dataSet[:,xInd]
                xlabel = data_vars[xInd]
                
                for yInd in range(len(data_vars)):
                    
                    if yInd == xInd:
                        
                        axs[xInd, yInd].hist( x)
                        
                    else:
                        y = dataSet[:,yInd]
                        
                        ylabel = data_vars[yInd]
                        axs[xInd, yInd].scatter(x, y)
                        
                        shape = np.shape(y)
                        y = y.reshape(shape[0],1)
                        x = x.reshape(shape[0],1)
                        Ahat = np.hstack(( np.ones((shape[0],1)), x))
                        c, _, _, _ = sc.lstsq( Ahat, y )
                        xline = np.linspace(x.min(),x.max(),100).reshape((100,1))
                        Bhat =  np.hstack( (np.ones((100,1)), xline))
                        rline = Bhat @ c
                        axs[xInd, yInd].plot( xline, rline, 'r')
                        
                        resid = y - (Ahat @ c)
                        R2 = 1 - np.sum(resid**2) / np.sum(((y) - np.mean(y))**2)
                        
                        axs[xInd, yInd].title.set_text(round(R2, 2))
                        
        else:
            for xInd in range(len(data_vars)):
                x = dataSet[:,xInd]
                xlabel = data_vars[xInd]
                
                for yInd in range(len(data_vars)):
                        
                    
                    y = dataSet[:,yInd]
                        
                    ylabel = data_vars[yInd]
                    axs[xInd, yInd].scatter(x, y)
                        
                    shape = np.shape(y)
                    y = y.reshape(shape[0],1)
                    x = x.reshape(shape[0],1)
                    Ahat = np.hstack(( np.ones((shape[0],1)), x))
                    c, _, _, _ = sc.lstsq( Ahat, y )
                    xline = np.linspace(x.min(),x.max(),100).reshape((100,1))
                    Bhat =  np.hstack( (np.ones((100,1)), xline))
                    rline = Bhat @ c
                    axs[xInd, yInd].plot( xline, rline, 'r')
                        
                    resid = y - (Ahat @ c)
                    R2 = 1 - np.sum(resid**2) / np.sum(((y) - np.mean(y))**2)
                        
                    axs[xInd, yInd].title.set_text(round(R2, 2))
                
    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        self.p = p
        shape = np.shape(A)
        one = np.ones((shape[0],1))
        
        A = A.reshape(shape[0],1)
    
        Ahat = np.hstack(( one, A))
        
        for i in range (2, p+1):
            Ahat = np.hstack(( Ahat, A**i))
            
        return(Ahat)

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.ind_vars = ind_var
        self.dep_var = dep_var
        
        ind = self.data.get_ind(self.ind_vars)
        self.A = self.data.get_all_data()[:,ind]
        self.A = np.array(self.A,dtype=np.float32)
        
        Ahat = self.make_polynomial_matrix(self.A,p)
        
        ind = self.data.get_ind(self.dep_var)
        y = self.data.get_all_data()[:,ind]
        shape = np.shape(y)
        y = y.reshape(shape[0],1)
        self.yOrig = y
        
        c, _, _, _ = sc.lstsq( Ahat, y )
        self.c = c
        self.intercept = c[0,0]
        self.slope = c[1:]

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor.
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        
        ind = self.data.get_header_indices(self.ind_vars)
        self.A = self.data.get_all_data()[:,ind]
        self.A = np.array(self.A,dtype=np.float32)
        
        Ahat = self.make_polynomial_matrix(self.A,p)
        
        ind = self.data.get_ind(self.dep_var)
        y = self.data.get_all_data()[:,ind]
        shape = np.shape(y)
        y = y.reshape(shape[0],1)
        self.yOrig = y
        
        c, _, _, _ = sc.lstsq( Ahat, y )
        self.c = c
        self.intercept = c[0,0]
        self.slope = c[1:]
        
        self.y = self.predict()
        self.residuals = self.compute_residuals(self.y)
        self.R2 = self.r_squared(self.y)
        self.m_sse = self.mean_sse()   
        
    def plot_pol_reg(self, ind_var, dep_var, p):
        
        self.ind_vars = ind_var
        self.dep_var = dep_var
        
        ind = self.data.get_ind(self.ind_vars)
        self.A = self.data.get_all_data()[:,ind]
        self.A = np.array(self.A,dtype=np.float32)
        
        Ahat = self.make_polynomial_matrix(self.A,p)
        
        ind = self.data.get_ind(self.dep_var)
        y = self.data.get_all_data()[:,ind]
        shape = np.shape(y)
        y = y.reshape(shape[0],1)
        self.yOrig = y
        
        c, _, _, _ = sc.lstsq( Ahat, y )
        self.c = c
        self.intercept = c[0,0]
        self.slope = c[1:]
        
        xline = np.linspace(self.A.min(),self.A.max(),100)
        
        rline = c[0]*np.ones(xline.shape)
        for i in range(1,p+1):
            rline += c[i]*xline**i
        
        plt.figure()
        plt.scatter( self.A, y )
        plt.plot( xline ,rline, 'r')
        plt.xlabel( ind_var )
        plt.ylabel( dep_var )
        