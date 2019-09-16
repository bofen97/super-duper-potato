from paddle import fluid
import numpy as np

def clac_qret(v,r,q_i,rho_i,dones):
    rho_bar = np.array([c if c <1.0 else 1.0 for c in rho_i ])
    v_final = v[-1]
    qret = v_final
    qrets = np.zeros_like(v)
    for i in reversed(range(len(v))):
        qret = r[i] + 0.99 * qret
        qrets[i]=qret
        qret  = (qret -q_i[i]) *rho_bar[i]  + v[i]*(1.- dones[i])
    return qrets

def logp(logits,actions):
    assert len(actions.shape) == 1 

    logits = logits - fluid.layers.reduce_max(logits)
    e_logits = fluid.layers.exp(logits)
    z = fluid.layers.reduce_sum(e_logits,dim=1)
    prob = e_logits / z

    actions = fluid.layers.unsqueeze(actions,axes=[1])
    actions_onehot = fluid.layers.one_hot(actions,prob.shape[1])
    actions_onehot = fluid.layers.cast(actions_onehot,dtype='float32')
    actions_prob = fluid.layers.reduce_sum(prob * actions_onehot,dim =1)
    log_acp = fluid.layers.log(actions_prob + 1e-6)
    return log_acp

def entropy(logits):
    logits = logits - fluid.layers.reduce_max(logits, dim=1)
    e_logits = fluid.layers.exp(logits)
    z = fluid.layers.reduce_sum(e_logits, dim=1)
    prob = e_logits / z
    entropy = -1.0 * fluid.layers.reduce_sum(
        prob * (logits - fluid.layers.log(z)), dim=1)

    return entropy


def action_score(score,actions):
    
    assert len(actions.shape) == 1
    
    actions = fluid.layers.unsqueeze(actions, axes=[1])
    actions_onehot = fluid.layers.one_hot(actions, score.shape[1])
    actions_onehot = fluid.layers.cast(actions_onehot, dtype='float32')
    actions_score =   fluid.layers.reduce_sum(score * actions_onehot, dim=1)
    
    
    return actions_score


    
 


import numpy as np
class ReplayMemory(object):
    def __init__(self,max_size,obs_shape,act_dim):
        self.max_size = max_size
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        
        self.obs= np.zeros((max_size,obs_shape[0],obs_shape[1],obs_shape[2]),dtype = 'float32')
        self.action = np.zeros((max_size),dtype='int64')
        self.reward = np.zeros((max_size),dtype='float32')
        self.done = np.zeros((max_size),dtype='float32')
        self.U     = np.zeros((max_size,act_dim),dtype='float32')
        self._curr_size = 0
        self._curr_pos = 0
    def sample_batch(self,batch_size):
        batch_index = np.random.randint(self._curr_size -300 -1 ,size=batch_size)
        batch_index = (self._curr_pos + 300 + batch_index) % self._curr_size
        U  = self.U[batch_index]
        obs = self.obs[batch_index]
        rew = self.reward[batch_index]
        act = self.action[batch_index]
        done = self.done[batch_index]
        return obs,act,rew,U,done
    def append(self,obs,act,rew,probs,done):
        if self._curr_size < self.max_size:
            self._curr_size +=1
        self.obs[self._curr_pos] = obs
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = rew
        self.U[self._curr_pos] = probs
        self.done[self._curr_pos] = done
        self._curr_pos =  (self._curr_pos + 1 ) % self._curr_size
    def size(self):
        return self._curr_size