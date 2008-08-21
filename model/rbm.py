# $Id$

import numpy as N
import scipy as S

class RBM:
    def __init__(self, numvis, numhid, hidactivate=None, visactivate=None,
        mfvis=True, mfhid=False):
        """Document me, please."""
        self._trainedfor = 0
        self._numhid = numhid
        self._numvis = numvis
        self._epsilonw = 0.001
        self._epsilonvb = 0.001
        self._epsilonhb = 0.001
        self._weightcost = 0.0002
        self._initialmomentum = 0.5
        self._finalmomentum = 0.9
        
        
        self._allparams = N.empty(numhid*numvis+numhid+numvis,dtype=float)
        self._allparams[:(numhid*numvis)] = 0.1 * S.randn(numhid*numvis)
        self._allparams[(numhid*numvis):] = 0
        
        self.set_params()
        
        if not hidactivate:
            self._hid_activate = RBM.sigmoid
        else:
            self._hid_activate = hidactivate
        if not visactivate:
            self._vis_activate = RBM.sigmoid
        else:
            self._vis_activate = visactivate
        self._mfhid = False
        self._mfvis = True
    
    def train(self,batchdata,nepochs,cdsteps=1,passthroughs=None):
        errors = N.empty(nepochs)
        numcases, numvis, numbatches = batchdata.shape
        for epoch in xrange(nepochs):
            vishidinc = N.zeros((self._numvis,self._numhid))
            hidbiasinc = N.zeros((self._numhid))
            visbiasinc = N.zeros((self._numvis))
            errsum = 0
            for batch in xrange(batchdata.shape[2]):
                data = batchdata[:,:,batch]
                if passthroughs:
                    for rbm in passthroughs:
                        data = rbm.passthrough(data)
                # Start of positive phase
                poshidprobs, poshidstates = self._hid_activate(self._vishid, 
                    self._hidbiases, data, meanfield=False)
                
                posprods = N.dot(data.T, poshidprobs)
                poshidact = N.sum(poshidprobs,axis=0)
                posvisact = N.sum(data,axis=0)
                
                # Start negative phase
                if cdsteps == 1:
                    negdata = self._vis_activate(self._vishid.T,
                        self._visbiases,poshidstates, meanfield=True)
                    neghidprobs = self._hid_activate(self._vishid,
                        self._hidbiases,negdata)[1] 
                else:
                    neghidstates = poshidstates
                    for i in xrange(cdsteps):
                        negdata, negvisstates = self._vis_activate(self._vishid.T,
                            self._visbiases,neghidstates)
                        
                        neghidstates, neghidprobs = self._hid_activate(
                            self._vishid, self._hidbiases,negvisstates)
                        
                        
                negprods = N.dot(negdata.T, neghidprobs)
                neghidact = N.sum(neghidprobs,axis=0)
                negvisact = N.sum(negdata,axis=0)
                
                # End of negative phase
                err = N.sum((data - negdata)**2)
                errsum = errsum + err
                if epoch+self._trainedfor > 5:
                    momentum = self._finalmomentum
                else:
                    momentum = self._initialmomentum
                
                # Updates
                
                vishidinc = (momentum * vishidinc + 
                    self._epsilonw*((posprods - negprods)/numcases - 
                    self._weightcost*self._vishid))
                visbiasinc = (momentum*visbiasinc +
                    (self._epsilonvb/numcases)*(posvisact-negvisact))
                hidbiasinc = (momentum*hidbiasinc +
                    (self._epsilonhb/numcases)*(poshidact-neghidact))
                
                self._vishid +=  vishidinc
                self._visbiases += visbiasinc
                self._hidbiases +=  hidbiasinc
                if passthroughs:
                    del data
            print "epoch %5d, error = %10.5f" % (epoch+1+self._trainedfor,
                errsum)
            errors[epoch] = errsum
        self._trainedfor += nepochs
        return errors
        
    def passthrough(self, data):
        return self._hid_activate(self._vishid, self._hidbiases,
            data, meanfield=True)
        
    def reconstruct_error(self, data, passthroughs=None):
        if passthroughs:
            for rbm in passthroughs:
                data = rbm.passthrough(data)
        hiddens = self._hid_activate(self._vishid, self._hidbiases, data,
            meanfield=False)
        junk, visibles = self._vis_activate(self._vishid.T, self._visbiases, hiddens,
            meanfield=True)
        return N.sum((visibles - data)**2)
    
    def set_params(self,newparams=None):
        if newparams != None:
            self._allparams = newparams
        self._vishid = N.asarray(N.reshape(self._allparams[
            :(self._numhid*self._numvis)],(self._numvis,self._numhid)))
        self._hidbiases = N.asarray(self._allparams[
            (self._numhid*self._numvis):(self._numhid*
            self._numvis+self._numhid)])
        self._visbiases = N.asarray(self._allparams[(self._numhid*
            self._numvis+self._numhid):])
        
        
    @staticmethod
    def sigmoid(weights, biases, states, meanfield=False):
        upd = 1./(1.+N.exp(-N.dot(states, weights)-biases[N.newaxis,:]))
        if not meanfield:
            return upd, N.float64(upd > N.random.uniform(size=upd.shape))
        else:
            return upd
    
    @staticmethod
    def linear(weights, biases, states, meanfield=False):
        upd = N.dot(states, weights) + biases[N.newaxis,:]
        if not meanfield:
            noise = N.random.normal(size=upd.shape)
            return (upd, upd + noise)
        else:
            return upd
    
if __name__ == '__main__':
    r = RBM(10,3)
    data = N.random.uniform(size=(10,10,100))
    r.train(data, 10)
