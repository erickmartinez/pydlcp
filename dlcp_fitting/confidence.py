#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:13:55 2019

Confidence intervals and confidence bands for the fittings


@author: Erick R Martinez Loran
"""

import numpy as np

def confint(n,pars,pcov,confidence=0.95):
    """
    This function returns the confidence interval for each parameter
    Parameters
    ----------
    n : int
        The number of data points
    pars : [double]
        The array with the fitted parameters
    pcov : [double]
        The covariance matrix
    confidence : float
        The confidence interval
    
    Returns
    -------
    ci : [double]
        The matrix with the confindence intervals for the parameters
        
    Note:
        Adapted from http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
        Copyright (C) 2013 by John Kitchin.
        https://kite.com/python/examples/702/scipy-compute-a-confidence-interval-from-a-dataset
    
    """
    from scipy.stats.distributions import  t
    
    p       = len(pars)         # number of data points
    dof     = max(0,n - p)      # number of degrees of freedom
    
    # Quantile of Student's t distribution for p=(1 - alpha/2)
    # tval = t.ppf((1.0 + confidence)/2.0, dof) 
    alpha = 1.0 - confidence
    tval = t.ppf(1.0 - alpha/2.0, dof) 
    
    ci = np.zeros((p,2),dtype=np.float64)
    
    for i, p,var in zip(range(n), pars, np.diag(pcov)):
        sigma = var**0.5
        ci[i,:] = [p-sigma*tval, p+sigma*tval]
        
    return ci

def confidence_interval(x,func,res,**kwargs):
    """
    This function estimates the confidence interval for the optimized parameters
    from the fit.
    
    Parameters
    ----------
    x: [double]
        The observed x points
    func: callback
        The function
    res: OptimizeResult
        The optimzied result from least_squares minimization
    **kwargs
        confidence: float
            The confidence level (default 0.95)
    Returns
    -------
    ci: [double]
        The confidence interval
    """    
    from scipy.optimize import optimize
    if not isinstance(res,optimize.OptimizeResult):
        raise ValueError('Argument \'res\' should be an instance of \'scipy.optimize.OptimizeResult\'')
    
    import scipy.linalg as LA 
    from scipy.stats.distributions import  t
    
    confidence      = kwargs.get('confidence',0.95)
    
    # The residual
    resid   = res.fun
    n       = len(resid)
    p       = len(res.x)
    dfe     = n - p
    # Get MSE. The degrees of freedom when J is full rank is v = n-p and n-rank(J) otherwise
    mse     = (LA.norm(resid))**2 / (dfe)
    
    # Needs to estimate the jacobian at the predictor point!!!
    # From MATLAB toolbox/stats/stats/nlpredci
#    ypred = func(x,res.x)
#    delta = np.zeros((len(ypred),p));
#    fdiffstep       = np.amax(np.spacing(res.x)**(1/3));
#    for i in range(p):
#        change = np.zeros(p)
#        if res.x[i] == 0:
#            nb = np.sqrt(LA.norm(res.x))
#            change[i] = fdiffstep * (nb + (nb == 0))
#        else:
#            change[i] = fdiffstep * res.x[i]
#            
#        predplus    = func(x,res.x+change)
#        delta[:,i]  = (predplus - ypred)/change[i]
    # Find R to get the variance
    _, R = LA.qr(res.jac)
    # Get the rank of jac
    Rinv    = LA.pinv(R)
    
    v = np.sum(Rinv**2,axis=1) * mse
    alpha = 1.0 - confidence 
    tval = t.ppf(1.0 - alpha/2.0, dfe) 
    delta = np.sqrt(v)*tval
    ci = np.zeros((p,2),dtype=np.float64)
    
    for i, p,d in zip(range(n), res.x,delta):
        ci[i,:] = [p-d, p+d]
        
    return ci

def getRSquared(x,y,popt,func):
    """
    This function estimates R^2 for the fitting
    Parameters
    ----------
    x : [double]
        The experimetnal x points
    y : [double]
        The experimental y points
    popt : [double]
        The best fit parameters
    func : function(x,*popt)
        The fitted function
    
    Returns
    -------
    rsquared : float
        The value of R^2
        
    Reference:
        http://bagrow.info/dsv/LEC10_notes_2014-02-13.html
    """
    
    # Get the sum of the residuals from the linear function
    SLIN = np.sum((y - func(x,*popt))**2)
    # Get the sum of the residuals from the constant function
    SCON = np.sum((y - y.mean())**2)
    # Get r-squared
    rsquared = 1.0 - SLIN/SCON
    return rsquared

def predband(x, xd, yd, p, func, conf=0.95):
    """
    This function estimates the prediction bands for the specified function
    https://codereview.stackexchange.com/questions/84414/obtaining-prediction-bands-for-regression-model
    """
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = len(xd)          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    from scipy.stats.distributions import  t
    q = t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


def mean_squared_error(yd,ym):
    """
    This function estimates the mean squared error of a fitting
    Parameters
    ----------
    yd: [double]
        The observed data points
    ym: [double]
        The datapoints from the model
    Returns
    -------
    mse: double
        The mean squared error
    """
    if len(yd) != len(ym):
        raise ValueError('The length of the observations should be the same '+\
                         'as the length of the predictions.')
    if len(yd) <=1:
        raise ValueError('Too few datapoints')
    N = len(yd)
    mse = np.sum((yd-ym)**2)/N
    return mse

def predint(x,xd,yd,func,res,**kwargs):
    """
    This function estimates the prediction bands for the fit
    (see: https://www.mathworks.com/help/curvefit/confidence-and-prediction-bounds.html)
    Parameters 
    ----------
    x: [double]
        The requested x points for the bands
    xd: [double]
        The x datapoints
    yd: [double]
        The y datapoints
    func: obj
        The fitted function
    res: OptimizeResult
        The optimzied result from least_squares minimization
    **kwargs
        confidence: float
            The confidence level (default 0.95)
        simulateneous: bool
            True if the bound type is simultaneous, false otherwise
        mode: [functional, observation]
            Default observation        
    """
    
    if len(yd) != len(xd):
        raise ValueError('The length of the observations should be the same '+\
                         'as the length of the predictions.')
    if len(yd) <=1:
        raise ValueError('Too few datapoints')
    from scipy.optimize import optimize
    
    if not isinstance(res,optimize.OptimizeResult):
        raise ValueError('Argument \'res\' should be an instance of \'scipy.optimize.OptimizeResult\'')
    
    import scipy.linalg as LA 
    simultaneous    = kwargs.get('simultaneous',True)
    mode            = kwargs.get('mode','observation')
    confidence      = kwargs.get('confidence',0.95)
    
    
    p   = len(res.x)
    
    
    
    # Needs to estimate the jacobian at the predictor point!!!
    # From MATLAB toolbox/stats/stats/nlpredci
    ypred = func(x,res.x)
    if callable(res.jac):
        delta = res.jac(x)
    else:
        delta = np.zeros((len(ypred),p));
        fdiffstep = np.spacing(np.abs(res.x))**(1/3);
    #    print('diff_step = {0}'.format(fdiffstep))
    #    print('popt = {0}'.format(res.x))
        for i in range(p):
            change = np.zeros(p)
            if res.x[i] == 0:
                nb = np.sqrt(LA.norm(res.x))
                change[i] = fdiffstep[i] * (nb + (nb == 0))
            else:
                change[i] = fdiffstep[i] * res.x[i]
                
            predplus    = func(x,res.x+change)
            delta[:,i]  = (predplus - ypred)/change[i]
#    print('delta = {0}'.format(delta))
            
    # Find R to get the variance
    _, R = LA.qr(res.jac)
    # Get the rank of jac
    rankJ   = res.jac.shape[1]
    Rinv    = LA.pinv(R)
    pinvJTJ = np.dot(Rinv,Rinv.T)
    
    # The residual
    resid   = res.fun
    n       = len(resid)
    # Get MSE. The degrees of freedom when J is full rank is v = n-p and n-rank(J) otherwise
    mse     = (LA.norm(resid))**2 / (n-rankJ)
    # Calculate Sigma if usingJ 
    Sigma   = mse*pinvJTJ
    
    # Compute varpred
    varpred = np.sum(np.dot(delta,Sigma) * delta,axis=1)
#    print('varpred = {0}, len: '.format(varpred,len(varpred)))
    alpha = 1.0 - confidence 
    if mode == 'observation':
        # Assume a constant variance model if errorModelInfo and weights are 
        # not supplied.
        errorVar    = mse * np.ones(delta.shape[0])
#        print('errorVar = {0}, len: '.format(errorVar,len(errorVar)))
        varpred     += errorVar
    # The significance
    if simultaneous:
        from scipy.stats.distributions import  f
        sch     = [rankJ + 1]
        crit    = f.ppf(1.0-alpha,sch,n-rankJ)
    else:
        from scipy.stats.distributions import  t
        crit    = t.ppf(1.0-alpha/2.0,n-rankJ)
        
    
    delta = np.sqrt(varpred) * crit
    
    
    lpb = ypred - delta
    upb = ypred + delta
    
    return ypred,lpb,upb
    
# References:
# - Statistics in Geography by David Ebdon (ISBN: 978-0631136880)
# - Reliability Engineering Resource Website:
# - http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
# - http://reliawiki.com/index.php/Simple_Linear_Regression_Analysis#Confidence_Intervals_in_Simple_Linear_Regression
# - University of Glascow, Department of Statistics:
# - http://www.stats.gla.ac.uk/steps/glossary/confidence_intervals.html#conflim
#def predint(x, xd, yd, p, func, conf=0.95):
#    """
#    This function estimates the prediction bands for the speciied function
#    Parameters
#    ----------
#    xd : [double]
#        The experimental x
#    yd : [double]
#        The experimental y
#    p : [double]
#        An array with the best fit parameters
#    func : function
#        The fitted function
#    conf : float
#        The confidence
#    
#    Adapted from:
#        http://bagrow.info/dsv/LEC10_notes_2014-02-13.html
#    """
#    
#    alpha   = 1.0 - conf      # significance
#    m       = xd.size         # data sample size
#    var_n   = len(p)          # number of parameters
#    
#    # Quantile of Student's t distribution for p=(1-alpha/2)
#    from scipy.stats.distributions import  t
#    dof = m - var_n
#    q = t.ppf(1.0 - alpha / 2.0, dof)
#    # Predicted values (best-fit model)
#    yp = func(xd, *p)
#    # Errors per point
#    y_err = yd - yp
#    # The sum of the error
#    s_err = np.sum(np.power(y_err, 2))
#    
#    sd = 1./(m-var_n)*s_err
#    sd = np.sqrt(sd)
#    sxd = np.sum((xd-xd.mean())**2)
#    sx  = (x-xd.mean())**2
#    
#    
#    dy = q * sd*(1.0/m + sx/sxd)
#    
# 
#    
#    return func(x,*p), abs(dy)




# TEST the functions
    
#from scipy.optimize import curve_fit


#x = np.array([0.5, 0.387, 0.24, 0.136, 0.04, 0.011])
#y = np.array([1.255, 1.25, 1.189, 1.124, 0.783, 0.402])
#
## this is the function we want to fit to our data
#def f(x, a, b):
#    'nonlinear function in a and b to fit to data'
#    return a * x / (b + x)
#
#initial_guess = [1.2, 0.03]
#popt, pcov = curve_fit(f, x, y, p0=initial_guess)
#ci = confint(len(x),popt,pcov)
#xp  = np.linspace(x.min(),x.max(),100)
#yp, dy = predint(xp,x,y,popt,f)
##yp = f(x,*popt)
#lower = yp - dy
#upper = yp + dy
#
#import matplotlib.pyplot as plt
#
## set-up the plot
##plt.axes().set_aspect('equal')
#plt.xlabel('X values')
#plt.ylabel('Y values')
#plt.title('non linear fit and confidence limits')
# 
## plot sample data
#plt.plot(x,y,'bo',label='Sample observations')
## plot line of best fit
#plt.plot(xp,yp,'r-',label='Fit')
#
# # plot confidence limits
#plt.fill_between(xp,lower,upper, color='#888888', alpha=0.4,label='Lower confidence limit (95%)')
##plt.plot(xx,upper,'b--',label='Upper confidence limit (95%)')
