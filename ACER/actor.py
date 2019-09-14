from model import PolicyModel
from acer import ACER_unTRoptimize
from agent import Agent
from utils import gen_qret
import parl
from utils import ReplayMemory
from copy import deepcopy
from utils import reward_shaping

@parl.remote_class
class Actor(object):

    def __init__(self,config):
        self.obs_dim = config['obsdim']
        self.act_dim = config['actdim']
        self.env     = config['env']
        self.sample_batch = config['sample_batch']
        
        model = PolicyModel(self.act_dim)
        alg   = ACER_unTRoptimize(model,config)
        self.agent = Agent(alg,config)
        
        self.obs = self.env.reset()
        self._mem = ReplayMemory(config['rmsize'],self.obs_dim*self.env.elevator_num,\
                                                          self.act_dim*self.env.elevator_num,self.env.elevator_num)
        
        self.global_step = 0
        
    def set_weights(self,params):
        self.agent.set_weights(params)
    def sample(self):

        _act_,_u_ = self.agent.sample(self.obs)
        
        mem_obs = self.obs.reshape(self.env.elevator_num*self.env.observation_space)
        mem_u = _u_.reshape(self.env.elevator_num*self.env.action_space)
        
        _obs_next_,_,_,info = self.env.step(_act_)


        mem_rew =reward_shaping(self.env,info,False)
        mem_act = deepcopy(_act_)
        
        
        
        
        
        self.global_step += 1
        if self.global_step > 500:
            
            self._mem.append(self._last_observations_arry,self._last_actions_array,
                                self._last_r,self._last_u_array)
        
        self._last_observations_arry = deepcopy(mem_obs)
        self._last_actions_array     = deepcopy(mem_act)
        self._last_r                 = mem_rew
        self._last_u_array           = deepcopy(mem_u)
        
        self.obs = _obs_next_
        
        if self._mem.size() > 500:

            obs_batch,act_batch,rew_batch,u_batch = self._mem.sample_batch(self.sample_batch)
            
            obs_batch = obs_batch.reshape(-1,self.env.observation_space)
            act_batch = act_batch.reshape(-1)
            rew_batch = rew_batch.reshape(-1)
            u_batch   = u_batch.reshape(-1,self.env.action_space)

            v_batch,qi_batch,rhoi_batch,f_batch,q_batch,rho_batch = self.agent.qret_param(obs_batch,act_batch,u_batch)
        
        
            qret = gen_qret(v_batch,rew_batch,qi_batch,rhoi_batch)
        
            ret_dict = {
                'obs':obs_batch,
                'act':act_batch,
                'rew':rew_batch,
                'u':u_batch,
                'v':v_batch,
                'qi':qi_batch,
                'rhoi':rhoi_batch,
                'f':f_batch,
                'q':q_batch,
                'rho':rho_batch,
                'qret':qret

            }

            return ret_dict
        else: return -1
