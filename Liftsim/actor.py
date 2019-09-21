from model import MLP
from algorithms import a2c,ppo
from agent import Agent
import numpy as np
from wrapper import Wrapper,ActionWrapper,ObservationWrapper
from rlschool import LiftSim
from env_vector import VectorEnv
from collections import defaultdict
from utils import calc_gae_for_multi_agent
from utils import calc_gae_for_multi_agent_2
import parl
@parl.remote_class
class Actor(object):
    def __init__(self,config,hour):

        self.config = config
        envs = []
        for i in range(self.config['env_num']):
            mansion_env = LiftSim()
            mansion_env._config._current_time = hour * 60 * 2  *2 

            mansion_env = Wrapper(mansion_env)
            mansion_env = ActionWrapper(mansion_env)
            mansion_env = ObservationWrapper(mansion_env)
            envs.append(mansion_env)
        self.vec_env = VectorEnv(envs)

        self.config['act_dim'] = mansion_env.action_space
        self.config['obs_shape'] = (mansion_env.observation_space,)
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
            actions_batch,value_batch = self.agent.sample(np.concatenate(self.obs_batch))
            # reshape value batch
            value_batch_reshape = np.reshape(value_batch,(self.config['env_num'],-1))
            #  use for get next step data .
            actions_batch_reshape=np.reshape(actions_batch,(self.config['env_num'],-1))
            actions_batch_reshape_int=[[int(action) for action in actions] for actions in actions_batch_reshape]
            # and stor to sample data
            next_obs_batch,rewards,_,_ =self.vec_env.step(actions_batch_reshape_int)

            rewards_batch = [ [reward]*4 for reward in rewards]

            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['act'].append(actions_batch_reshape[env_id])
                env_sample_data[env_id]['rew'].append(rewards_batch[env_id])
                env_sample_data[env_id]['value'].append(value_batch_reshape[env_id])

                if sample_step == self.config['sample_batch_steps'] -1:
                    #calc advantage and value target
                    next_obs_per_env =   next_obs_batch[env_id]
                    next_values              = self.agent.value(next_obs_per_env)


                    rews                = env_sample_data[env_id]['rew']
                    values             = env_sample_data[env_id]['value']
                    concat_values = np.concatenate(values)
                    concat_rews = np.concatenate(rews)

                    """advantages = calc_gae_for_multi_agent(concat_rews,concat_values,next_values,self.config['gamma'],\
                                                                                                self.config['lambda'])
                    """
                    advantages = calc_gae_for_multi_agent_2(concat_rews,concat_values,self.config['gamma'],\
                                                                                             self.config['lambda'])
                    
                    value_targets = advantages + concat_values
                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['act'].extend(env_sample_data[env_id]['act'])
                    sample_data['adv'].extend(advantages)
                    sample_data['vtag'].extend(value_targets)
                    sample_data['rews'].extend(concat_rews)
            self.obs_batch = next_obs_batch
        train_data={
            'obs':np.concatenate(sample_data['obs']),
            'act':np.concatenate(sample_data['act']),
            'adv':np.stack(sample_data['adv']),
            'vtag':np.stack(sample_data['vtag']),
            'rews':np.stack(sample_data['rews'])
        }
        


        return train_data

    def set_weights(self,params):
        self.agent.set_weights(params)


