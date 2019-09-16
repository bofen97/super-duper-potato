import gym
from parl.env.atari_wrappers import wrap_deepmind,get_wrapper_by_cls,MonitorEnv
from parl.env.vector_env import VectorEnv
from model import AtariModel
from alg import a2c
from agent import Agent
from collections import defaultdict
import numpy as np
from parl.utils import calc_gae
import parl
@parl.remote_class
class Actor(object):
    def __init__(self,config):
        self.config = config
        self.envs = []
        
        for _ in range(config['env_num']):
            env = gym.make(config['env_name'])
            env = wrap_deepmind(env,dim=config['env_dim'],obs_format='NCHW')
            self.envs.append(env)
        self.config['obs_shape'] = env.observation_space.shape
        self.vec_env = VectorEnv(self.envs)
        self.obs_batch = self.vec_env.reset()
        model = AtariModel(env.action_space.n)
        alg = a2c(model,config)
        self.agent = Agent(alg,config)
    def get_metrics(self):
        metrics = defaultdict(list)
        for env in self.envs:
            monitor = get_wrapper_by_cls(env,MonitorEnv)
            if monitor is not None:
                
                for episode_rew,episode_step in monitor.next_episode_results():
                    metrics['episode_rewards'].append(episode_rew)
                    metrics['episode_steps'].append(episode_step)
        return metrics
    def set_weights(self,params):
        self.agent.set_weights(params)
        
        
    def sample(self):
        sample_data = defaultdict(list)
        env_sample_data = {}
        
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)
        
        
        for i in range(self.config['sample_batch_steps']):
            action_batch,value_batch = self.agent.sample(np.stack(self.obs_batch))
            next_obs_batch,reward_batch,done_batch,info_batch = \
                                self.vec_env.step(action_batch)
            
            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['act'].append(action_batch[env_id])
                env_sample_data[env_id]['rew'].append(reward_batch[env_id])
                env_sample_data[env_id]['value'].append(value_batch[env_id])
                env_sample_data[env_id]['done'].append(done_batch[env_id])
            
                if done_batch[env_id] or i == self.config['sample_batch_steps'] -1:
                    next_value = 0.
                    if not done_batch[env_id] :
                        next_obs = np.expand_dims(next_obs_batch[env_id],0)
                        next_value = self.agent.value(next_obs)
                        
                    value = env_sample_data[env_id]['value']
                    reward = env_sample_data[env_id]['rew']
                    advantage = calc_gae(reward,value,next_value,self.config['gamma'],self.config['lambda'])
                    vtag = advantage + value
                    
                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['act'].extend(env_sample_data[env_id]['act'])
                    sample_data['adv'].extend(advantage)
                    sample_data['vtag'].extend(vtag)
                    
                    env_sample_data[env_id] = defaultdict(list)
            self.obs_batch = next_obs_batch
        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])
        return sample_data

