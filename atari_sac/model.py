import parl
from paddle import fluid
import numpy as np
LOG_STD_CLIP_MAX = 2.0
LOG_STD_CLIP_MIN = -20.0
class policy_model(parl.Model):
    def __init__(self,actdim):
        self.fc = parl.layers.fc(size=256,act='relu')
        self.fc2 = parl.layers.fc(size=256,act='relu')
        self.mean = parl.layers.fc(size=actdim,act=None)
        self.logstd  = parl.layers.fc(size=actdim,act=None)
    def __call__(self,x):
        x  = self.fc(x)
        x= self.fc2(x)
        mean = self.mean(x)
        logstd = self.logstd(x)
        clip_logstd = fluid.layers.clip(logstd,LOG_STD_CLIP_MIN,LOG_STD_CLIP_MAX)


        return mean,clip_logstd


class q_model(parl.Model):
    def __init__(self,actdim):
        self.fc = parl.layers.fc(size=256,act='relu')
        self.fc2 = parl.layers.fc(size=256,act='relu')
        self.q = parl.layers.fc(size=1,act=None)

    def __call__(self,x):
        x = self.fc(x)
        x =self.fc2(x)
        x = self.q(x)
        x = fluid.layers.squeeze(x,axes=[1])
        return x

class v_model(parl.Model):
    def __init__(self):
        self.fc = parl.layers.fc(size=256,act='relu')
        self.fc2 = parl.layers.fc(size=256,act='relu')

        self.v = parl.layers.fc(size=1,act=None)

    def __call__(self,x):
        x = self.fc(x)
        x = self.fc2(x)
        x = self.v(x)
        x = fluid.layers.squeeze(x,axes=[1])
        return x
