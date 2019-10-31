import parl
from paddle import fluid
from parl.core.fluid.policy_distribution import CategoricalDistribution
from copy import deepcopy
from DiagGauss import DiagGauss


class sac(parl.Algorithm):
    def __init__(self,q1,q2,pi,v,hyperparam):
        self.q1 = q1
        self.q2 = q2
        self.pi = pi
        self.v = v
        self.target_v = deepcopy(v)    
        self.hyperparam = hyperparam
        
        self.gamma = hyperparam['gamma']
        self.temperature = hyperparam['temperature']
        
        
    def q1_value(self,state,action):
        assert len(state.shape) == 2 
        assert len(action.shape) == 2

        obsact = fluid.layers.concat([state,action],axis=1)
        q = self.q1(obsact)
        return q
        
        
        
    def q2_value(self,state,action):
        assert len(state.shape) == 2 
        assert len(action.shape) == 2
        obsact = fluid.layers.concat([state,action],axis=1)
        q = self.q2(obsact)
        return q
        
        
    def v_value(self,state):
        value = self.v(state)
        return value


    
    def sample(self,state,noise):
        """

        
        """
        mean,logstd = self.pi(state)
        std = fluid.layers.exp(logstd)

        diag = DiagGauss(mean,std)
        
        assert noise.shape == mean.shape
        noise.stop_gradient = True
        
        
        u = mean + std * noise
        a = fluid.layers.tanh(u)

        logu = diag.loglikelihoold(u)
        
        eab = fluid.layers.log(1.0 - fluid.layers.square(a)+ 1e-6)

        logp = logu - fluid.layers.reduce_sum(input=eab,dim=1)


        entropy = diag.entropy()
        entropy = fluid.layers.reduce_sum(entropy)
        mean = fluid.layers.tanh(mean)
        return a,logp,mean,entropy


    def sync_target_v(self,smooth_weight):
        self.v.sync_weights_to(self.target_v,decay=smooth_weight)
        
        
    def learn(self,state,noise,reward,next_state,done,actions,lr):
        
        sample_actions ,logp ,mean,entropy= self.sample(state,noise)


        q1 = self.q1_value(state,sample_actions)
        q2 = self.q2_value(state,sample_actions)

        
        min_q1q2 = fluid.layers.elementwise_min(q1,q2)
        vtarget = min_q1q2 - self.temperature * logp
    
        vtarget.stop_gradient = True


        next_value = self.target_v(next_state)
        qtarget = reward + self.gamma *(1. - done) * next_value
        qtarget.stop_gradient = True

        q1_ = self.q1_value(state,actions)
        q1_delta = q1_ - qtarget
        q1_loss = 0.5 * fluid.layers.reduce_sum(fluid.layers.square(q1_delta))

        q2_ = self.q2_value(state,actions)
        q2_delta = q2_ - qtarget
        q2_loss = 0.5 * fluid.layers.reduce_sum(fluid.layers.square(q2_delta))


        v = self.v_value(state)
        delta = v - vtarget
        v_loss = 0.5 * fluid.layers.reduce_sum(fluid.layers.square(delta))


        pi_loss = -1.0* fluid.layers.reduce_sum(min_q1q2 - self.temperature * logp)


        
        total_loss = pi_loss + q1_loss + q2_loss + v_loss
        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=40.0))
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr)
        optimizer.minimize(total_loss)

        return total_loss,pi_loss,q1_loss,q2_loss,v_loss,entropy

        
        
        
        
        
        
    
    
    
    
    
    
    
        
        
        
        