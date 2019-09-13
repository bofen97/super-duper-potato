import numpy as np

def clc_gae(rewards,values,values_next,gamma,lam,num_elev):
    reshape_values = values.squeeze(1).reshape(-1,num_elev)
    reshape_values_next = values_next.squeeze(1).reshape(-1,num_elev)
    reshape_rewards     = rewards.reshape(-1,num_elev)  
    tds = np.zeros_like(reshape_rewards)
    c=0.
    adv = np.zeros_like(tds)

    for i in reversed(range(len(tds))):

        tds[i] = reshape_rewards[i] + gamma *reshape_values_next[i] - reshape_values[i]
        c = c * gamma*lam +tds[i]
        adv[i]= c
    
    
    vtag = adv + reshape_values
    
    adv = adv.reshape(rewards.shape)
    vtag = vtag.reshape(rewards.shape)
    
    return adv,vtag

import six

class vector_env(object):
    def __init__(self,envs):
        assert isinstance(envs,list)
        
        self.envs = envs
        self.envs_num = len(envs)
        
    def reset(self):
        return [env.reset() for env in self.envs]
    def step(self,actions_list):
        observation_batch,reward_batch = [],[]
        
        for env_id in six.moves.range(self.envs_num):
            obs,_,_,info = self.envs[env_id].step(actions_list[env_id])
            observation_batch.append(obs)
            rew = reward_shaping(self.envs[env_id],info,True)
            reward_batch.append(rew)
        return observation_batch,reward_batch

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

def test(agent,steps,render = True):
    c=0.
    obs = env.reset()
    for t in range(steps):
        if render:    
            env.render()
        obs = env.state
        act = agent.predict(obs)
        _,r,_,_ = env.step(act)
        print(t,c)
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