import numpy as np
from collections import defaultdict
from model import PolicyModel
from acer import ACER_unTRoptimize
from agent import Agent
from actor import Actor
import queue
import threading
import parl
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
    l = learner(acer_config)
    c = 0
    while not l.should_done():
        c += 1
        if c % 100 ==0:
            l.agent.save('./model.ckpt')
        l.step()