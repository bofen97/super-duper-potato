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
            rew = reward_shape(info,self.envs[env_id])
            observation_batch.append(obs)
            reward_batch.append(rew)
        return observation_batch,reward_batch
    

def reward_shape(info,env):
    
    rewards_base = -(info['time_consume']+ 0.01* info['energy_consume'] + \
                                     100.0 * info['given_up_persons'])*1e-4
    ret =[]
    for _ in range(4):                                    
        ret.append(rewards_base)
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
        c+= r
        print(t,c)
    return c





from matplotlib import pyplot as plt
def plot_results(results):
    mean_rews = results['meanr']
    mean_adv = results['meanadv']
    xs = range(len(mean_rews))
    plt.plot(xs, mean_rews, label='mean_r_per_batch')
    plt.plot(xs,mean_adv,label='mean_adv_per_batch')
    plt.title("results")
    plt.legend()
    plt.show()
