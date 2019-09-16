import parl
from paddle import fluid
from parl.core.fluid.policy_distribution import CategoricalDistribution
class a2c(parl.Algorithm):
    def __init__(self,model,config):
        
        self.model = model
        self.vf_coeff = config['vf_loss_coeff']
    def value(self,obs):
        value = self.model.value(obs)
        return value
    def predict(self,obs):
        logits= self.model.policy(obs)
        probs = fluid.layers.softmax(logits,axis=1)
        predict_acts = fluid.layers.argmax(probs,axis=1)
        return predict_acts
        
        
    def sample(self,obs):
        logits ,value= self.model.policy_and_value(obs)
        probs = fluid.layers.softmax(logits,axis=1)
        sample_acts = fluid.layers.sampling_id(probs)
        return sample_acts,value
    def learn(self,obs,act,adv,vtag,lr,ent_coeff):
        logits =self.model.policy(obs)
        
        policy_distributions = CategoricalDistribution(logits)
        
        action_log_probs = policy_distributions.logp(act)
        
        pi_loss = -1.0  * fluid.layers.reduce_sum(action_log_probs * adv)
        
        values = self.model.value(obs)
        
        delta = values - vtag
        vloss = 0.5*fluid.layers.reduce_sum(fluid.layers.square(delta))
        
        policy_entropy = policy_distributions.entropy()
        
        entropy = fluid.layers.reduce_sum(policy_entropy)
        
        total_loss = (pi_loss + vloss * self.vf_coeff + ent_coeff * entropy )
        
        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=40.0))

        optimizer = fluid.optimizer.AdamOptimizer(lr)
        optimizer.minimize(total_loss)
        
        return total_loss,pi_loss,vloss,entropy
    
        
    