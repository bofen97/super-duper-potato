from a2c import A2C
from model import LiftSimModel
from agent import LiftSimAgent
import numpy as np
from collections import defaultdict
import queue
import parl
import threading
from actor import Actor

class learner(object):
    def __init__(self,config):
        self. config = config
        self.merage = defaultdict(list)
        model = LiftSimModel(config['actdim'])
        alg =A2C(model,config)
        self.agent = LiftSimAgent(alg,config)

        self.sample_data_queue = queue.Queue()
        self.params_queues = []
        self.sample_total_steps = 0
        self.rew_mean_per_batch =[]
        self.mean_adv_per_batch=[]
        self.value_mean_per_batch=[]
        self.create_actor() 
    
    def create_actor(self):
        parl.connect(self.config['master_addr'])
        for i in range(self.config['num_actor']):
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

            batch = remote_actor.sample()
            self.sample_data_queue.put(batch)

    def step(self):
        latest_params = self.agent.get_weights()
        for params_queue in self.params_queues:
            
            params_queue.put(latest_params)
            
        train_batch = defaultdict(list)
        
        for i in range(self.config['num_actor']):
            
            sample_data = self.sample_data_queue.get()
            for key,value in sample_data.items():
                train_batch[key].append(value)
            
        for key,value in train_batch.items():
            train_batch[key] = np.concatenate(value)

        obs = train_batch['obs']
        act = train_batch['act']
        adv = train_batch['adv']
        vtag = train_batch['vtag']
        rews = train_batch['rew']
        value=train_batch['value']
        mean_rew  =np.sum(rews) 
        mean_adv  = np.sum(adv)
        mean_value = np.sum(value)/ (self.config['num_actor']*self.config['env_num'])
        self.mean_adv_per_batch.append(mean_adv)
        self.rew_mean_per_batch.append(mean_rew)
        self.value_mean_per_batch.append(mean_value)
        
        self.sample_total_steps += obs.shape[0]
        total_loss,piloss,_,ent,_,_ = self.agent.learn(obs,act,adv,vtag)
        if self.sample_total_steps % 50000 ==0:    
            print('global step: {} total loss:{} ,pi loss : {}, entropy :{}' \
                                                                            .format(self.sample_total_steps, \
                                                                            total_loss,piloss,ent))
                                    
    def should_done(self):
        return self.sample_total_steps >= self.config['max_sample_steps']
    
if __name__ == "__main__":
    from config import config
    from utils import test,plot_results
    train = True
    l = learner(config)
    if not train:    
        l.agent.restore('./model.ckpt')
        print('=========================restored==============================')
        print(test(l.agent,28800,True))
    else:
        print("=========================train=======================")
        c = 0
        while not l.should_done():
            c+=1
            if c%100 == 0:
                l.agent.save('./model.ckpt')
            l.step()
        
        results = {
            'meanr':l.rew_mean_per_batch,
            'meanadv':l.mean_adv_per_batch,
            'meanvalue':l.value_mean_per_batch
        }
        plot_results(results)
