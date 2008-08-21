# $Id$ 

import numpy as N
import scipy as S

class PreTrainedAutoencoder:
	def __init__(self, trainedRBMs):
	    params = []
	    for rbm in trainedRBMs:
	        params.append(N.array(rbm._vishid.flat))
	        params.append(N.array(rbm._hidbiases.flat))
	    for rbm in trainedRBMs[::-1]:
	        params.append(N.array(rbm._vishid.T.flat))
	        params.append(N.array(rbm._visbiases.flat))
	    
	    self._allparams = N.concatenate(params)
	    self._weights = []
	    self._biases = []
	    offset = 0
	    for rbm in trainedRBMs:
	        top = offset + rbm._numhid * rbm._numvis
	        weights = N.asarray(N.reshape(self._allparams[offset:top], 
	            (rbm._numvis, rbm._numhid)))
	        offset = top
	        top = offset + rbm._numhid
	        biases = N.asarray(self._allparams[offset:top])
	        self._weights.append(weights)
	        self._biases.append(biases)
	        offset = top
	    for rbm in trainedRBMs[::-1]:
	        top = offset + rbm._numhid * rbm._numvis
	        weights = N.asarray(N.reshape(self._allparams[offset:top], 
	            (rbm._numhid, rbm._numvis)))
	        offset = top
	        top = offset + rbm._numvis
	        biases = N.asarray(self._allparams[offset:top])
	        offset = top
	        self._weights.append(weights)
	        self._biases.append(biases)
	        
	def eval(self, data, returninputoutput=False):
	    states = data
	    layerinputs = []
	    layeroutputs = []
	    for i in xrange(len(self._weights) / 2):
	        if i > 0:
	            #print "sigmoid on " + str(states.shape)
	            states = 1. / (1 + N.exp(-states))
	            layeroutputs.append(states)
	        #print "passing through linearity %s * %s (states * weights)" % (str(states.shape), str(self._weights[i].shape))
	        states = N.dot(states,self._weights[i]) + self._biases[i][N.newaxis,:]
	        layerinputs.append(states)
	    layeroutputs.append(states)
	    for i in xrange(len(self._weights) / 2, len(self._weights)):
	        states = N.dot(states,self._weights[i]) + self._biases[i][N.newaxis,:]
	        layerinputs.append(states)
	        #print "passing through linearity %s * %s (states * weights)" % (str(states.shape), str(self._weights[i].shape))
	        states = 1. / (1 + N.exp(-states))
	        layeroutputs.append(states)
	        
	        #print "sigmoid on " + str(states.shape)
	    if returninputoutput:
	        return layerinputs, layeroutputs
	    else:
	        return states
	
	def _error(self, params, data):
	    params_save = N.copy(self._allparams)
	    self._allparams.flat = params.flat
	    m = self.eval(data)
	    if N.any(m.flat == 0):
	        m.flat[m.flat == 0] = 1.0e-8
	    if N.any(m.flat == 1):
	        m.flat[m.flat == 1] = 1 - 1.0e-8
	    E = -N.sum(data * N.log(m) + (1 - data)*N.log(1 - m))
	    self._allparams.flat = params_save.flat
	    del params_save
	    return E
	    	
	def _grad(self, params, data):
	    params_save = N.copy(self._allparams)
	    self._allparams.flat = params.flat
	    layerinputs,layeroutputs = self.eval(data,True)
	    delta = [None] * len(self._weights)
	    grads = [None] * len(self._weights)
	    grads_biases = [None] * len(self._weights)
	    all_grads = [None] * 2 * len(self._weights)
	    for i in xrange(len(self._weights)-1,len(self._weights)/2-1,-1):
	        if i == len(self._weights) - 1:
	            delta[i] = layeroutputs[i] - data
	        else:
	            sig = (1 + N.exp(-layerinputs[i]))**-1
	            hprime = sig * (1 - sig)
	            delta[i] = hprime * N.dot(delta[i+1], self._weights[i+1].T)
	    for i in xrange(len(self._weights)/2 - 1, -1, -1):
	        if i == len(self._weights)/2 - 1:
	            delta[i] = N.dot(delta[i+1], self._weights[i+1].T)
	        else:
	            sig = (1 + N.exp(-layerinputs[i]))**-1
	            hprime = sig * (1 - sig)
	            delta[i] = hprime * N.dot(delta[i+1], self._weights[i+1].T)
	    for i in xrange(len(self._weights)):
	        if i == 0:
	            grads[i] =  N.dot(data.T,delta[i])
	        else:
	            grads[i] = N.dot(layeroutputs[i-1].T,delta[i])
	        grads_biases[i] = N.sum(delta[i],axis=0)
	        
	    for i in xrange(len(grads)):
	        all_grads[2*i] = grads[i]
	        all_grads[2*i+1] = grads_biases[i]
	    
	    self._allparams.flat = params_save.flat
	    del params_save
	    return N.concatenate([x.flat for x in all_grads])
	
	def tune(self, batchdata, valid, numruns, maxiter):
	    validerrors = []
	    for run in xrange(numruns):
	        for batch in xrange(batchdata.shape[2]):
	            print "Run %d, batch %d... " % (run, batch)
	            self.all_params = S.optimize.fmin_cg(self._error, self._allparams, 
	                self._grad, (batchdata[:,:,batch],), maxiter=maxiter)
	        validerrors.append(self._error(self._allparams, valid))
	        print "Validation error: " + str(validerrors[-1])
	        if run > 0 and validerrors[-1] > validerrors[-2]:
	            print "Validation error increased!"
	            break
	
	def encode(self, data):
	    states = data
	    for i in xrange(len(self._weights) / 2):
	        if i > 0:
	            #print "sigmoid on " + str(states.shape)
	            states = 1. / (1 + N.exp(-states))
	        #print "passing through linearity %s * %s (states * weights)" % (str(states.shape), str(self._weights[i].shape))
	        states = N.dot(states,self._weights[i]) + self._biases[i][N.newaxis,:]
	    return states
