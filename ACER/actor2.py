from wrapper import ActionWrapper,ObservationWrapper,Wrapper

from model import PolicyModel
from acer import ACER_unTRoptimize

from agent import Agent


import numpy as np

from collections import defaultdict
from utils import gen_qret

import parl
import queue
import threading

from utils import ReplayMemory
from copy import deepcopy
from utils import test,reward_shape

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


        mem_rew =reward_shape(info,self.env)
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
class learner(object):
    def __init__(self,config):
        self.merage = defaultdict(list)
        model = PolicyModel(config['actdim'])
        self.master_addr = config['master_addr']
        self.num_actor  = config['num_actor']
        self.config=config
        self.max_step = config['max_sample_steps']
        alg =ACER_unTRoptimize(model,config)
        self.agent = Agent(alg,config)

        self.sample_data_queue = queue.Queue()
        self.params_queues = []
        self.sample_total_steps = 0

        self.create_actor() 
    
    def create_actor(self):
        parl.connect(self.master_addr)
        for i in range(self.num_actor):
            params_queue = queue.Queue()
            self.params_queues.append(params_queue)

            remote_thread = threading.Thread(
                target=self.run_remote_episode,args=(params_queue,))

            remote_thread.setDaemon(True)
            remote_thread.start()







    def run_remote_episode(self,params_queue):
        remote_actor = Actor(self.config)

        while True:
            latest_params = params_queue.get()

            remote_actor.set_weights(latest_params)

            # the batch is -1  or train data .. will be check
            batch = remote_actor.sample()
            self.sample_data_queue.put(batch)

    def step(self):
        latest_params = self.agent.get_weights()
        for params_queue in self.params_queues:
            
            params_queue.put(latest_params)
            
        train_batch = defaultdict(list)
        
        for i in range(self.num_actor):
            
            sample_data = self.sample_data_queue.get()
            if sample_data  == -1:
                continue
            else:
                for key,value in sample_data.items():
                    train_batch[key].append(value)
            
        if train_batch:    
            for key,value in train_batch.items():

                train_batch[key] = np.concatenate(value)

            obs = train_batch['obs']
            act = train_batch['act']
            v = train_batch['v']
            qi = train_batch['qi']
            rhoi = train_batch['rhoi']
            rho = train_batch['rho']
            f    = train_batch['f']
            q = train_batch['q']
            qret = train_batch['qret']
            

            self.sample_total_steps += obs.shape[0]
            c=np.ones_like(rhoi) * 10.0
            total_loss,pi_loss,ent,qloss = self.agent.learn(obs,act,v,qi,rhoi,f,rho,q,qret,c)
            print('global step: {} total loss:{} ,pi loss : {}, entropy :{}  , qloss: {} ' \
                                                                        .format(self.sample_total_steps, \
                                                                        total_loss,pi_loss,ent,qloss))
        else:pass
        
                                
    def should_done(self):
        return self.sample_total_steps >= self.max_step
    

if __name__ == "__main__":
    from config import acer_config
    from utils import plot_durations
    l = learner(acer_config)
    c = 0
    plots= []
    while not l.should_done():
        c += 1
        if c % 500 ==0:
            acc = test(l.agent,3600,True)
            plots.append(acc)
            plot_durations(plots)
            l.agent.save('./model.ckpt')
            
        l.step()
