from ppo import PPO
from model import LiftSimModel
from agent import LiftSimAgent
import numpy as np
from utils import clc_gae,vector_env,test
import parl
from collections import defaultdict
@parl.remote_class
class Actor(object):
    def __init__(self,config):
        self.config = config
        self.vec_env = vector_env([config['env']]*config['env_num'])
        model = LiftSimModel(config['actdim'])
        alg = PPO(model,config)
        self.agent = LiftSimAgent(alg,config)
        self.obs_batch = self.vec_env.reset()
  
    def reshape(self,acts_batch):
        
        act_array =  np.array(acts_batch).reshape(self.config['env_num'],self.config['env'].elevator_num)
        return [[int(a) for a in acts] for acts in act_array ]
    
    def sample(self):
        
        sample_data = defaultdict(list)
        env_sample_data = defaultdict(list)
        
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)
        for i in range(self.config['sample_batch_steps']):
            acts_batch  = self.agent.sample(np.concatenate(self.obs_batch))
            
            next_obs_batch,rewards_batch  =self.vec_env.step(self.reshape(acts_batch))
            
            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['act'].append(self.reshape(acts_batch)[env_id])
                env_sample_data[env_id]['rew'].append(rewards_batch[env_id])
                env_sample_data[env_id]['obs_next'].append(next_obs_batch[env_id])
            
            
                if i == self.config['sample_batch_steps'] -1:
                    
                    env_obs_batch = env_sample_data[env_id]['obs']
                    env_obs_next_batch = env_sample_data[env_id]['obs_next']
                    
                    env_values    = self.agent.value(np.concatenate(env_obs_batch))
                    env_values_next = self.agent.value(np.concatenate(env_obs_next_batch))
                    env_rewards   = np.concatenate(env_sample_data[env_id]['rew'])
                    
                    advantage,vtag = clc_gae(env_rewards,env_values,env_values_next,self.config['gamma'],
                                          self.config['lam'],self.config['env'].elevator_num)
                    
                    
                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['act'].extend(env_sample_data[env_id]['act'])
                    sample_data['adv'].extend(advantage)
                    sample_data['vtag'].extend(vtag)
                    
            self.obs_batch = next_obs_batch
        
        
        train_data = {
            'obs':np.concatenate(sample_data['obs']),
            'act':np.concatenate(sample_data['act']),
            'adv':np.array(sample_data['adv'],dtype='float32'),
            'vtag':np.array(sample_data['vtag'],dtype='float32'),
            
        }
        return train_data
    
    def set_weights(self,params):
        self.agent.set_weights(params)
