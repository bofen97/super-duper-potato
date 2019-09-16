import parl
from paddle import fluid
from utils import entropy,action_score

class ACER_unTRoptimize(parl.Algorithm):
    def __init__(self,model,config=None):
        self.model = model
        self.qcoeff = config['qcoeff']
        self.eps = 1e-6
        
    def qret_params(self,obs,act,u):
        logit , q  = self.model.policy_and_q(obs)
        f = fluid.layers.softmax(logit,axis=1)        
        f_i = action_score(f,act)
        q_i = action_score(q,act)
        v = fluid.layers.reduce_sum(f * q ,dim=1)
        rho = f / (self.eps + u)
        rho_i = action_score(rho,act)
        return v,q_i,rho_i,f,q,rho


    
    
    def sample(self,obs):
        logits= self.model.policy(obs)

        probs = fluid.layers.softmax(logits)
        sample_actions = fluid.layers.sampling_id(probs)
        return sample_actions,probs

    def predict(self,obs):
        logits  = self.model.policy(obs)
        probs = fluid.layers.softmax(logits)
        predict_act = fluid.layers.argmax(probs,axis = 1)
        return predict_act

    
    
    def learn(self,obs,act,constant_f,constant_q,constant_q_i,constant_rho,constant_rho_i,constant_v,
              constant_qret,c,lr,entropy_coeff):
        
        logit , q  = self.model.policy_and_q(obs)
        f = fluid.layers.softmax(logit,axis=1)        
        f_i = action_score(f,act)
        q_i = action_score(q,act)
        entropy_ = entropy(logit)
        policy_entropy = fluid.layers.reduce_mean(entropy_)
        c_adv = constant_qret - constant_v
        c_advantage = c_adv * fluid.layers.elementwise_min(c,constant_rho_i)
        c_advantage.stop_gradient = True
        logf = fluid.layers.log(f_i + self.eps)
        gain_f = logf * c_advantage
        lossf = -1.0* fluid.layers.reduce_mean(gain_f)
        c_adv_bc = constant_q - constant_v
        c_advantage_bc = c_adv_bc * fluid.layers.relu(1.0 -  10.0/(constant_rho + self.eps)) * constant_f
        c_advantage_bc.stop_gradient = True
        logf_bc = fluid.layers.log(f + self.eps)
        gain_bc = fluid.layers.reduce_sum(logf_bc * c_advantage_bc,dim=1)
        loss_bc = -1.0* fluid.layers.reduce_mean(gain_bc)
        pi_loss = lossf + loss_bc
        constant_qret.stop_gradient = True

        qloss = fluid.layers.square_error_cost(q_i,constant_qret)
        qloss= fluid.layers.reduce_mean(qloss)


        total_loss = pi_loss + entropy_coeff*  policy_entropy +qloss * self.qcoeff
        
        
        fluid.clip.set_gradient_clip(

            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=40.0)
        )
        
        policy_optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate =lr)
        policy_optimizer.minimize(total_loss)
                
        return total_loss,pi_loss,policy_entropy,qloss
