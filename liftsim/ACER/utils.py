from paddle import fluid
import numpy as np

def gen_qret(v,r,q_i,rho_i):
    rho_bar = np.array([c if c <1.0 else 1.0 for c in rho_i ])
    v_final = v[-1]
    qret = v_final
    qrets = np.zeros_like(v)
    for i in reversed(range(len(v))):
        qret = r[i] + 0.99 * qret
        qrets[i]=qret
        qret  = (qret -q_i[i]) *rho_bar[i]  + v[i]
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





import matplotlib.pyplot as plt
def plot_durations(episode_durations):
    
    plt.figure(2)
    plt.clf()
    plt.title('training              :)  ')
    plt.xlabel('steps')
    plt.ylabel('3600 steps acc reward')
    plt.plot(episode_durations)
    plt.pause(0.001)  # pause a bit so that plots are updated

from rlschool import LiftSim
from wrapper import Wrapper,ActionWrapper,ObservationWrapper


env = ObservationWrapper(ActionWrapper(Wrapper(LiftSim())))

def test(agent,steps,render):
    acc = 0.
    env.reset()
    for  _ in range(steps):
        if render:    
            env.render()
        ob  = env.state
        
        act = agent.predict(ob)
        
        ob,r,_,_ = env.step(act)
        acc +=r
    return acc







def reward_shape(info,env):
    rewards_base = -(info['time_consume']*10.0+  0.01 * info['energy_consume'] + \
                                     200.0 * info['given_up_persons'])*1e-4
    elevator_states = env.env.state.ElevatorStates
    elevator_weights= []
    elevator_speeds=[]
    elevator_numfloors=[]
    for state in elevator_states:
        elevator_weights.append(-(state.MaximumLoad - state.LoadWeight) *1e-6)
        elevator_speeds.append((state.MaximumSpeed - state.Velocity) *1e-3)
        elevator_numfloors.append(    -   np.abs(state.MaximumFloor/2.0  - state.Floor)*0 )
    
    ret =[]
    for i in range(4):
        ret.append(rewards_base + elevator_speeds[i] + elevator_weights[i]+ elevator_numfloors[i])

    return ret
