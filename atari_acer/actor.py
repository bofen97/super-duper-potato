import numpy as np
import parl
from paddle import fluid
from acer import ACER_unTRoptimize
from model import AtariModel
from agent import Agent
from utils import clac_qret,ReplayMemory
from parl.env.vector_env import VectorEnv
from parl.env.atari_wrappers import wrap_deepmind,MonitorEnv,get_wrapper_by_cls
import gym
from collections import defaultdict
@parl.remote_class
class Actor(object):
    def __init__(self,config):     
        self.config = config
        self.envs = []
        for  _ in range(self.config['env_num']):
            env = gym.make(self.config['env_name'])
            env = wrap_deepmind(env,dim=self.config['env_dim'],obs_format='NCHW')
            self.envs.append(env)
        self.env_vec = VectorEnv(self.envs)
        self.obs_batch = self.env_vec.reset()
        self.config['obs_shape'] = env.observation_space.shape
        self.config['act_dim'] = env.action_space.n
        model = AtariModel(self.config['act_dim'])
        acer = ACER_unTRoptimize(model,self.config)
        self.agent = Agent(acer,self.config)
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
        env_sample_data = {}
        sample_data=defaultdict(list)
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)
        
        for i in range(self.config['sample_batch_steps']):
            action_batch,probs_batch = self.agent.sample(np.stack(self.obs_batch))
            next_obs_batch,reward_batch,done_batch,_ = self.env_vec.step(action_batch)

            for env_id in range(self.config['env_num']):    
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['act'].append(action_batch[env_id])
                env_sample_data[env_id]['rew'].append(reward_batch[env_id])
                env_sample_data[env_id]['done'].append(done_batch[env_id])
                env_sample_data[env_id]['probs'].append(probs_batch[env_id])

                if done_batch[env_id] or i == self.config['sample_batch_steps'] - 1:
                    obs = np.stack(env_sample_data[env_id]['obs'])
                    act = np.stack(env_sample_data[env_id]['act'])
                    rew = np.stack(env_sample_data[env_id]['rew'])
                    done = np.stack(env_sample_data[env_id]['done'])
                    probs = np.stack(env_sample_data[env_id]['probs'])

                    v,qi,rhoi,f,q,rho=self.agent.qret_param(obs,act,probs)
                    qret = clac_qret(v,rew,qi,rhoi,done)
                    c = np.ones_like(rhoi) * 10.0

                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['act'].extend(env_sample_data[env_id]['act'])
                    sample_data['rew'].extend(env_sample_data[env_id]['rew'])
                    sample_data['probs'].extend(env_sample_data[env_id]['probs'])
                    sample_data['v'].extend(v)
                    sample_data['qi'].extend(qi)
                    sample_data['rhoi'].extend(rhoi)
                    sample_data['f'].extend(f)
                    sample_data['q'].extend(q)
                    sample_data['rho'].extend(rho)
                    sample_data['qret'].extend(qret)
                    sample_data['c'].extend(c)


                    env_sample_data[env_id] = defaultdict(list)

            self.obs_batch = next_obs_batch
        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])


        return sample_data



    
