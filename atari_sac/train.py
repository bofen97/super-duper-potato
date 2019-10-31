from model import q_model ,v_model,policy_model
from sac import sac
from config import hyperparam
from agent import Agent
import gym
from parl.env.atari_wrappers import wrap_deepmind
from parl.utils.replay_memory import ReplayMemory
from parl.env.vector_env import VectorEnv
from copy import deepcopy
import numpy as np
from parl.utils.window_stat import WindowStat
from tensorboardX import SummaryWriter
from collections import defaultdict

class Actor(object):
    def __init__(self,hyperparam):
        self.hyperparam = hyperparam
        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.v_loss_stat = WindowStat(100)
        self.q1_loss_stat = WindowStat(100)
        self.q2_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.writer = SummaryWriter()

        self.episode_rewards_stat = WindowStat(100)
        self.episode_rewards = []
        self.envs = []
        for env_id in range(self.hyperparam['env_num']):    
            env = gym.make(hyperparam['env_name'])        
            self.envs.append(env)
        self.vec_env = VectorEnv(self.envs)




        
        self.hyperparam['obs_dim'] = env.observation_space.shape[0]
        self.hyperparam['act_dim'] = env.action_space.shape[0]

        q1 = q_model(self.hyperparam['act_dim'])
        q2 = q_model(self.hyperparam['act_dim'])
        v = v_model()
        pi = policy_model(self.hyperparam['act_dim'])

        alg = sac(q1,q2,pi,v,hyperparam)

        self.agent = Agent(alg,hyperparam)
        self.obs_batch = self.vec_env.reset()

        self.mem = ReplayMemory(self.hyperparam['replay_buffer_size'],self.hyperparam['obs_dim'],
                                                               self.hyperparam['act_dim'] )
        self.global_steps = 0
        self.mergic = None
    
    def run(self):
        while self.global_steps <= self.hyperparam['max_sample_steps']:
            self.sample()

            


    def learn_step(self,states,actions,rews,dones):
        self.global_steps += 1
        if self.global_steps >= 500:
            for i in range(self.hyperparam['env_num']):
                self.mem.append(self._last_observation_array[i],
                                                    self._last_actions[i],self._last_rewards[i],
                                                    deepcopy(states[i]),self._last_dones[i])
        
        self._last_observation_array = deepcopy(states)
        self._last_actions                   = deepcopy(actions)
        self._last_rewards               = deepcopy(rews)
        self._last_dones             = deepcopy(dones)
    
        if self.mem.size() > 500:
            obs,act,rew,next_obs,done = self.mem.sample_batch(self.hyperparam['batch_size'])
            total_loss,pi_loss,q1_loss,q2_loss,v_loss,entropy=self.agent.learn(obs,act,rew,next_obs,done)
            self.total_loss_stat.add(total_loss)
            self.q1_loss_stat.add(q1_loss)
            self.q2_loss_stat.add(q2_loss)
            self.v_loss_stat.add(v_loss)
            self.entropy_stat.add(entropy)
            self.pi_loss_stat.add(pi_loss)

            metrics = {
                'total loss':self.total_loss_stat.mean,
                'q1 loss':self.q1_loss_stat.mean,
                'q2 loss':self.q2_loss_stat.mean,
                'v loss':self.v_loss_stat.mean,
                'entropy':self.entropy_stat.mean,
                'pi loss':self.pi_loss_stat.mean,
                'episode rewards': self.episode_rewards_stat.mean 
                
            }
            for key,value in metrics.items():
                self.writer.add_scalar(key,value,self.global_steps)
        








    def sample(self):
        self.vec_env.envs[0].render()
        actions,_,_,_ = self.agent.sample(np.stack(self.obs_batch))
        next_obs,rewards,dones,_ = self.vec_env.step(actions)
        self.episode_rewards.append(rewards)
        if dones[0]:
            self.episode_rewards_stat.add(np.sum(self.episode_rewards))
            self.episode_rewards = []
    
        self.learn_step(self.obs_batch,actions,rewards,dones)
        self.agent.sync_weights()
        self.obs_batch = next_obs





if __name__ == "__main__":
    from config import hyperparam
    a  = Actor(hyperparam)
    a.run()

        