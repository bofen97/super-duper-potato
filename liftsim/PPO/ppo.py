#ppo.py
import parl
from paddle import fluid
from parl.core.fluid.policy_distribution import CategoricalDistribution
from copy import deepcopy
class PPO(parl.Algorithm):
    def __init__(self,model,config):
        self.model = model
        self.oldmodel = deepcopy(self.model)


        self.v_coeff = config['v_coeff']
        self.clip_value = config['clip_value']
    
    def sample(self,obs):
        logits,_ = self.model(obs)
        
        probs = fluid.layers.softmax(logits,axis=1)
        
        sample_act = fluid.layers.sampling_id(probs,seed=2)
        return sample_act
    
    def predict(self,obs):
        logits,_ = self.model(obs)
        
        probs = fluid.layers.softmax(logits,axis=1)
        
        predict_act = fluid.layers.argmax(probs,axis=1)
        
        return predict_act
    
    def value(self,obs):
        _,v = self.model(obs)
        return v
    def sysnc_old_policy(self):
        self.model.sync_weights_to(self.oldmodel)
        
    def learn(self,obs,act,advantage,vtarget,lr,entropy_coeff):
        assert len(act.shape) == 1
        
        
        logits,values = self.model(obs)
        old_logits,_ = self.oldmodel(obs)

        
        
        old_policy_distribution = CategoricalDistribution(old_logits)
        policy_distribution = CategoricalDistribution(logits)
        
        entropy = policy_distribution.entropy()
        entropy = fluid.layers.reduce_mean(entropy)
        
        
        logp    = policy_distribution.logp(act)
        logp_old = old_policy_distribution.logp(act)

        logp_old.stop_gradient = True
        
        kl =old_policy_distribution.kl(policy_distribution)

        kl_mean = fluid.layers.reduce_mean(kl)
        kl_mean.stop_gradient = True
        

        ratio = fluid.layers.exp(logp-logp_old)
        ratio_adv = ratio * advantage
        
        clip_    = fluid.layers.clip(ratio,1. - self.clip_value , 1. + self.clip_value)
        clip_adv  = clip_ * advantage
        
        pi_cost = fluid.layers.elementwise_min(ratio_adv,clip_adv,axis=-1)
        pi_cost =  -1.0* fluid.layers.reduce_mean(pi_cost)
        

        delta = values - vtarget
        v_loss = 0.5 * fluid.layers.reduce_mean(fluid.layers.square(delta))


        
        
        total_loss = pi_cost + entropy * entropy_coeff + v_loss * self.v_coeff       
        
        #clip normal gradint ..
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr)
        optimizer.minimize(total_loss)
        
        
        
        return total_loss,pi_cost,entropy,v_loss,kl_mean
        
        
        
