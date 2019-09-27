from model import MLP
from algorithms import a2c,ppo
from agent import Agent
import numpy as np
from wrapper import Wrapper,ActionWrapper,ObservationWrapper
from rlschool import LiftSim
from env_vector import VectorEnv
from collections import defaultdict
from parl.utils import calc_gae
import parl
@parl.remote_class
class Actor(object):
    def __init__(self,config):
        self.config = config
        envs = []
        for id_ in range(self.config['env_num']):
            mansion_env = LiftSim()
            mansion_env = Wrapper(mansion_env)
            mansion_env = ActionWrapper(mansion_env)
            mansion_env = ObservationWrapper(mansion_env)
            envs.append(mansion_env)
        self.vec_env = VectorEnv(envs)
        self.elev_num = mansion_env.elevator_num
        self.config['act_dim'] = mansion_env.action_space *self.elev_num
        self.config['obs_shape'] = (mansion_env.observation_space*self.elev_num,)
        model=MLP(self.config['act_dim'])
        if self.config['algorithm'] =='a2c':
            print("algorithm is a2c .  ") 
            algorithm= a2c(model,self.config)
        elif self.config['algorithm'] =='ppo':
            print("algorithm is ppo . ")
            algorithm= ppo(model,self.config)
        else:
            algorithm = None

        self.agent = Agent(algorithm,self.config)
        self.obs_batch = self.vec_env.reset()
    def sample(self):

        sample_data = defaultdict(list)
        env_sample_data = {}
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)

        for sample_step in range(self.config['sample_batch_steps']):

            sample_actions,value_batch= self.agent.sample(np.stack(self.obs_batch))
            sample_actions = np.reshape(sample_actions,(-1,self.elev_num))
            sample_actions_batch = [[int(action) for action in actions] for actions in sample_actions]
            next_obs_batch,reward_batch,_,_ = self.vec_env.step(sample_actions_batch)
            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['rew'].append(reward_batch[env_id])
                env_sample_data[env_id]['act'].append(sample_actions_batch[env_id])
                env_sample_data[env_id]['value'].append(value_batch[env_id])

                if sample_step  == self.config['sample_batch_steps'] -1:
                    reward = env_sample_data[env_id]['rew']
                    value = env_sample_data[env_id]['value']
                    next_value = self.agent.value(np.expand_dims(next_obs_batch[env_id],axis=0) )

                    advantage = calc_gae(reward,value,next_value,self.config['gamma'],self.config['lambda'])
                    value_target = advantage + value
        
                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['act'].extend(env_sample_data[env_id]['act'])
                    sample_data['adv'].extend(advantage)
                    sample_data['vtag'].extend(value_target)
                    sample_data['rew'].extend(reward)
                    sample_data['value'].extend(value)
            self.obs_batch = next_obs_batch
        
        advs = np.array(sample_data['adv'])
        advs= np.concatenate([[adv]*self.elev_num for adv in advs])
        train_data={
            'obs':np.stack( sample_data['obs']),
            'act':np.concatenate(sample_data['act']),
            'rew': np.array(sample_data['rew']),
            'value':np.array(sample_data['value']),
            'adv': advs,
            'vtag':np.array(sample_data['vtag']) }

        return train_data

    def set_weights(self,params):
        self.agent.set_weights(params)

