"""
Dirt simple binary logistic regression routines written with numpy/scipy.
Uses scipy.optimize.fmin_cg for the nonlinear optimization. 

TODO: 
    * Object-orient the whole thing (maybe; I like the simplicity of it now)
    * Fallback to gradient descent if scipy isn't available.
    * Handle multinomial regression (maybe?)
    * Come up with a test scenario to run if called at the command line.

By David Warde-Farley, April 2008

$Id$
"""

import numpy as N
import scipy as S
from scipy import optimize as opt

def lr_probs(coeffs, data):
    """
    Estimate p(target = 1) using the given coefficients.
     * coeffs is a 1-dimensional (D+1)-length array.
     * data is assumed to be N x D design matrix (no column of 1's).
    """
    bias = coeffs[0]
    coeffs = coeffs[1:]
    eta = bias + N.sum(coeffs * data, axis=1)
    probs = 1. / (1. + N.exp(-eta))
    return probs
    
def lr_nloglik(coeffs, data, target, decay=0):
    """
    Calculate the negative log likelihood of a dataset with given targets.
     * coeffs is a 1-dimensional (D+1)-length array.
     * data is assumed to be N x D design matrix (no column of 1's).
     * target is a binary 1D array of length N.
     * decay is an optional parameter controlling the amount of weight decay
       (regularization).
    """
    probs = lr_probs(coeffs, data)
    bias = coeffs[0]
    coeffs = coeffs[1:]
    L = N.sum(target * N.log(probs)) + N.sum((1 - target) * N.log(1 - probs))
    if decay:
        L = L - decay * (bias**2 + N.sum(coeffs**2))
    return -L

def lr_nloglik_grad(coeffs, data, target, decay=None):
    """
    Calculate the gradient of the negative log likelihood.
    * coeffs is a 1-dimensional (D+1)-length array.
    * data is assumed to be N x D design matrix (no column of 1's).
    * target is a binary 1D array of length N.
    * decay is an optional parameter controlling the amount of weight decay
      (regularization).
    """
    probs = lr_probs(coeffs, data)
    augdata = N.concatenate((N.ones((data.shape[0],1)),data),axis=1)
    grads = -N.sum((target - probs)[:,N.newaxis] * augdata, axis=0)
    if decay:
        grads = grads + 2*decay*coeffs
    return grads

def lr_fit(data, target, decay=None, maxiter=2000, callback=None, randinit=False):
    """
    Fit a binary logistic regression model to the given dataset and targets.
    * data is assumed to be N x D design matrix (no column of 1's).
    * target is a binary 1D array of length N.
    * decay is an optional parameter controlling the amount of weight decay
      (regularization). Defaults to none/zero.
    * maxiter is an optional parameter passed along to scipy.optimize.fmin_cg.
      Defaults to 20000.
    """
    #random initialization? 
    if randinit:
        initial = S.randn(data.shape[1] + 1) * 0.01
    else:
        initial = N.zeros(data.shape[1] + 1)
    beta = opt.fmin_cg(lr_nloglik, initial, lr_nloglik_grad, \
        args=(data,target,decay), maxiter=maxiter,callback=callback)
    return beta

def lr_eval(coeffs, data, target, thresh=0.5):
    probs = lr_probs(coeffs, data)
    predictions = probs > thresh
    disagreements = N.where(predictions != target)[0]
    falsenegs = len(N.where(target[disagreements] == True)[0])
    falsepos = len(N.where(target[disagreements] == False)[0])
    trueneg = len(N.where(target[predictions == target] == False)[0])
    truepos = len(N.where(target[predictions == target] == True)[0])
    actualpos = len(N.where(target == True)[0])
    actualneg = len(N.where(target == False)[0])
    predpos = len(N.where(predictions == True)[0])
    predneg = len(N.where(predictions == False)[0])
    
    precision = float(truepos) / predpos
    recall = float(truepos) / actualpos
    specificity = float(trueneg) / actualneg
    accuracy = (target.shape[0] - float(len(disagreements))) / target.shape[0]
    print float(len(disagreements))
    
    return dict(precision=precision, recall=recall, 
        specificity=specificity, accuracy=accuracy)
    
    
if __name__ == "__main__":
    print "Punt!"
    pass
