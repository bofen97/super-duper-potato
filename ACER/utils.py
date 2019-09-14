from paddle import fluid
import numpy as np

def gen_qret(v,r,q_i,rho_i):
    rho_bar = np.array([c if c <1.0 else 1.0 for c in rho_i ])

    v = v.reshape(-1,4)
    r = r.reshape(-1,4)
    q_i = q_i.reshape(-1,4)
    rho_i = rho_i.reshape(-1,4)
    
    v_final = v[-1]
    qret = v_final
    qrets = np.zeros_like(v)
    for i in reversed(range(len(v))):
        qret = r[i] + 0.99 * qret
        qrets[i]=qret
        qret  = (qret -q_i[i]) *rho_bar[i]  + v[i]
    qrets = qrets.reshape(-1)
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


def action_prob(prob,actions):
    
    assert len(actions.shape) == 1
    
    actions = fluid.layers.unsqueeze(actions, axes=[1])
    actions_onehot = fluid.layers.one_hot(actions, prob.shape[1])
    actions_onehot = fluid.layers.cast(actions_onehot, dtype='float32')
    actions_prob =   fluid.layers.reduce_sum(prob * actions_onehot, dim=1)
    
    
    return actions_prob


    
 


import numpy as np
class ReplayMemory(object):
    def __init__(self,max_size,obs_dim,act_dim,scarl_act):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.obs= np.zeros((max_size,obs_dim,),dtype = 'float32')
        self.action = np.zeros((max_size,scarl_act),dtype='int64')
        self.reward = np.zeros((max_size,scarl_act),dtype='float32')
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
        return obs,act,rew,U
    def append(self,obs,act,rew,probs):
        if self._curr_size < self.max_size:
            self._curr_size +=1
        self.obs[self._curr_pos] = obs
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = rew
        self.U[self._curr_pos] = probs
        self._curr_pos =  (self._curr_pos + 1 ) % self._curr_size
    def size(self):
        return self._curr_size






def reward_shaping(env,info,use_shaping = False):
    base_rew = -(info['time_consume'] + 0.01 * info['energy_consume'] + 100 * info['given_up_persons'])*1e-4 
    other_rews = []
    for state in env.env.state.ElevatorStates:
        if state.Velocity  >= state.MaximumSpeed * 0.29 - 5e-3 or \
            state.Velocity  <= state.MaximumSpeed * 0.29 + 5e-3:
            rew =5e-4
        else:
            rew = -5e-4

        
        other_rews.append(rew)
    
    ret = []
    for i in range(4):    
        if use_shaping:    
            ret.append(base_rew +other_rews[i] )
        else:
            ret.append(base_rew)
    return ret



from rlschool import LiftSim

from wrapper import  ObservationWrapper,ActionWrapper,Wrapper

env = ObservationWrapper(ActionWrapper(Wrapper(LiftSim())))

def test(agent,steps,render = False):
    c=0.
    obs = env.reset()
    for t in range(steps):
        if render:    
            env.render()
        obs = env.state
        act = agent.predict(obs)
        _,r,_,_ = env.step(act)
        c+= r
    return c




from matplotlib import pyplot as plt
def plot_results(results):
    mean_rews = results['meanr']
    mean_adv = results['meanadv']
    mean_v   = results['meanvalue']
    xs = range(len(mean_rews))
    plt.plot(xs, mean_rews, label='mean_r_per_batch')
    plt.plot(xs,mean_adv,label='mean_adv_per_batch')
    plt.plot(xs,mean_v,label='mean_value_per_batch')
    plt.title("results")
    plt.legend()
    plt.show()


def plot_scores(scores):
    plt.figure(2)
    plt.clf()
    plt.title('                               :)                         ')
    plt.xlabel('3600 steps per episode')
    plt.ylabel('score')
    plt.plot(scores)
    plt.pause(0.00001)

 