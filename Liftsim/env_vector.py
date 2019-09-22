
import six
from collections import defaultdict


class VectorEnv(object):
    

    def __init__(self, envs):
        
        self.envs = envs
        self.envs_num = len(envs)

    def reset(self):
        
        return [env.reset() for env in self.envs]

    def step(self, actions):
        
        obs_batch, reward_batch, done_batch, info_batch = [], [], [], []
        for env_id in six.moves.range(self.envs_num):
            obs, _, done, info = self.envs[env_id].step(actions[env_id])

            reward = - (
            info['time_consume'] + 
            5e-4* info['energy_consume'] +
            300 * info['given_up_persons']) * 1e-4
            
            if done:
                obs = self.envs[env_id].reset()

            obs_batch.append(obs)
            reward_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)
        return obs_batch, reward_batch, done_batch, info_batch