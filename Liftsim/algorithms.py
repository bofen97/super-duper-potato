import parl
from paddle import fluid
from copy import deepcopy
from parl.core.fluid.policy_distribution import CategoricalDistribution
single_elevator_dim = 22
class a2c(parl.Algorithm):
    def __init__(self,model,config):      
        self.model = model
        self.l2reg = config['l2_reg']
        self.vf_coeff = config['vf_loss_coeff']

    def value(self,obs):
        value = self.model.value(obs)
        return value
    def predict(self,obs):
        predict_logits= self.model.policy(obs)
        predict_logits = fluid.layers.reshape(predict_logits,[-1,single_elevator_dim]) #only use for liftsim..
        predict_probs = fluid.layers.softmax(predict_logits,axis=1)
        predict_acts = fluid.layers.argmax(predict_probs,axis=1)
        return predict_acts
        
        
    def sample(self,obs):
        sample_logits ,value= self.model.policy_and_value(obs)
        sample_logits = fluid.layers.reshape(sample_logits,[-1,single_elevator_dim]) #act dim
        sample_probs = fluid.layers.softmax(sample_logits,axis=1)
        sample_acts = fluid.layers.sampling_id(sample_probs)
        return sample_acts,value


    def learn(self,obs,act,adv,vtag,lr,ent_coeff):
        logits =self.model.policy(obs)
        logits = fluid.layers.reshape(logits,[-1,single_elevator_dim]) 
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

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr,regularization=fluid.regularizer.L2DecayRegularizer(
        regularization_coeff=self.l2reg))

        optimizer.minimize(total_loss)
        
        return total_loss,pi_loss,vloss,entropy
    



class ppo(parl.Algorithm):
    def __init__(self,model,config):
        self.model = model
        self.vf_coeff = config['vf_loss_coeff']
        self.l2reg = config['l2_reg']
        self.old_model = deepcopy(self.model)
    def sync_old_policy(self):

        self.model.sync_weights_to(self.old_model)

    def value(self,obs):
        value = self.model.value(obs)
        return value
    def predict(self,obs):
        predict_logits= self.model.policy(obs)
        predict_logits = fluid.layers.reshape(predict_logits,[-1,single_elevator_dim]) #only use for liftsim..
        predict_probs = fluid.layers.softmax(predict_logits,axis=1)
        predict_acts = fluid.layers.argmax(predict_probs,axis=1)
        return predict_acts
        
        
    def sample(self,obs):
        sample_logits ,value= self.model.policy_and_value(obs)
        sample_logits = fluid.layers.reshape(sample_logits,[-1,single_elevator_dim]) #act dim
        sample_probs = fluid.layers.softmax(sample_logits,axis=1)
        sample_acts = fluid.layers.sampling_id(sample_probs)
        return sample_acts,value

    def learn(self,obs,act,adv,vtag,lr,ent_coeff):
        
        
        # calc policy loss 
        logits = self.model.policy(obs)
        logits = fluid.layers.reshape(logits,[-1,single_elevator_dim]) 
        
        old_logits = self.old_model.policy(obs)
        old_logits = fluid.layers.reshape(old_logits,[-1,single_elevator_dim]) 
        
        old_logits.stop_gradient = True
        policy_distributions = CategoricalDistribution(logits)
        oldpolicy_distributions = CategoricalDistribution(old_logits)

        action_log_probs = policy_distributions.logp(act)
        old_action_log_probs = oldpolicy_distributions.logp(act)
        old_action_log_probs.stop_gradient = True
        
        ratio = fluid.layers.exp(action_log_probs - old_action_log_probs)
        
        Lcpi =ratio * adv
        
        Lclip = fluid.layers.clip(ratio,1. - 0.2,1. + 0.2) * adv
        
        policy_loss = -1.0 * fluid.layers.reduce_sum(fluid.layers.elementwise_min(Lcpi,Lclip))
        
        # calc value loss
        values = self.model.value(obs)
        delta  = values - vtag
        
        value_loss = 0.5 * fluid.layers.reduce_sum(fluid.layers.square(delta))
        
        # calc entropy 
        
        policy_entropy = policy_distributions.entropy()
        policy_entropy = fluid.layers.reduce_sum(policy_entropy)
        
        
        # def total loss ,  using minimize optimizer 
        
        # we want maxmize policy entropy and policy loss ,so using entropy coeff < 0  and poliy loss * -1.0
        
        
        total_loss = policy_loss + self.vf_coeff * value_loss  + policy_entropy * ent_coeff 
        
        # define optimizer
        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=40.0))

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr,regularization=fluid.regularizer.L2DecayRegularizer(
        regularization_coeff=self.l2reg))
        optimizer.minimize(total_loss)
        return total_loss,policy_loss,value_loss,policy_entropy
    