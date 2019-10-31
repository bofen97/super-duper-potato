from paddle import fluid
import numpy as np


class DiagGauss(object):
    def __init__(self,mean,std):
        assert len(mean.shape) == 2
        assert len(std.shape) == 2
        self.std = std
        self.mean = mean



    def loglikelihoold(self,a):

        return fluid.layers.reduce_sum(-0.5 * fluid.layers.square((a-self.mean)/self.std) - 0.5 * np.log(2.0*np.pi) - \
             fluid.layers.log(self.std),dim=1)
    

    def likelihoold(self,a):
        return fluid.layers.exp(self.loglikelihoold(a))
    
    def entropy(self):
        return fluid.layers.reduce_sum( fluid.layers.log(self.std) + 0.5 * np.log(2*np.pi*np.e),dim=1)
    def kl(self,mean1,std1):
        return fluid.layers.reduce_sum(fluid.layers.log(std1/self.std) + \
        (fluid.layers.square(self.std)+\
        fluid.layers.square(self.mean - mean1))/(2.0*fluid.layers.square(std1)) \
        - 0.5,dim=1)
        
        
        
    