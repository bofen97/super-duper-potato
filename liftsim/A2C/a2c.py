import parl
from paddle import fluid
from parl.core.fluid.policy_distribution import CategoricalDistribution
class A2C(parl.Algorithm):
    def __init__(self,model,config):
        self.model = model
        self.v_coeff = config['v_coeff']
    
    def sample(self,obs):
        logits,_ = self.model(obs)
        
        probs = fluid.layers.softmax(logits,axis=1)
        
        sample_act = fluid.layers.sampling_id(probs)
        
        return sample_act
    
    def predict(self,obs):
        logits,_ = self.model(obs)
        
        probs = fluid.layers.softmax(logits,axis=1)
        
        predict_act = fluid.layers.argmax(probs,axis=1)
        
        return predict_act
    
    def value(self,obs):
        _,v = self.model(obs)
        return v
    
    def learn(self,obs,act,advantage,vtarget,lr,entropy_coeff):
        assert len(act.shape) == 1
        
        
        logits,values = self.model(obs)
        
        policy_distribution = CategoricalDistribution(logits)
        
        action_log_probs    = policy_distribution.logp(act)
        
        pi_loss = -1.0 * fluid.layers.reduce_sum(action_log_probs * advantage)
        

        

        values = fluid.layers.squeeze(values,axes=[1])
        delta = values - vtarget
        v_loss = 0.5 * fluid.layers.reduce_sum(fluid.layers.square(delta))
        
        entropy = policy_distribution.entropy()
        entropy = fluid.layers.reduce_sum(entropy)
        
        total_loss = (pi_loss + entropy * entropy_coeff + v_loss * self.v_coeff  )     
        
        fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=40.0))
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr)
        optimizer.minimize(total_loss)
        
        
        
        return total_loss,pi_loss,entropy,v_loss
        
        
        
