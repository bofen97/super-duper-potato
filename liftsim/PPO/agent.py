#agent.py
from paddle import fluid
import parl
from parl.utils.scheduler import PiecewiseScheduler,LinearDecayScheduler
import numpy as np
import random

class LiftSimAgent(parl.Agent):
    
    def __init__(self,alg,config):
        
        self.config = config
        super(LiftSimAgent,self).__init__(alg)
        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],config['max_sample_steps'])
        self.entropy_coeff_scheduler = PiecewiseScheduler(config['entropy_coeff_scheduler'])
        
        
        
    
    def learn(self,obs,act,adv,vtag):
        
        lr = self.lr_scheduler.step(step_num=obs.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()
        feed = {
            'obs':obs,
            'act':act,
            'adv':adv,
            'tag':vtag,
            'lr':np.array([lr],dtype='float32'),
            'entcoeff':np.array([entropy_coeff],dtype='float32')
            
        }
        
        total_loss,piloss,ent,vloss,kl=self.fluid_executor.run(self.learn_program,feed=feed,fetch_list =self.loss_array)
        return total_loss[0],piloss[0],vloss[0],ent[0],lr,entropy_coeff,kl[0]
    
    
    def predict(self,obs):
        feed={
            'obs':obs
        }
        acts = self.fluid_executor.run(self.predict_program,feed=feed,fetch_list=[self.predict_acts])[0]
        
        
        return [int(a) for a in acts ]
    
    
    def sample(self,obs):
        feed={
            'obs':obs
        }
        
        acts = self.fluid_executor.run(self.sample_program,feed=feed,fetch_list=[self.sample_acts])[0]
        
        

        return [int(a) for a in acts]
    def value(self,obs):
        feed={
            'obs':obs
        }
        
        values = self.fluid_executor.run(self.value_program,feed=feed,fetch_list=[self.values])[0]
        
        
        return values
    
    
        
    
    def build_program(self):
        self.predict_program = fluid.Program()
        self.sample_program = fluid.Program()
        self.value_program = fluid.Program()
        self.learn_program = fluid.Program()
        
        with fluid.program_guard(self.learn_program):
            obs=fluid.layers.data('obs',[self.config['obsdim']],'float32')
            act = fluid.layers.data('act',[],'int64')
            adv = fluid.layers.data('adv',[],'float32')
            vtag = fluid.layers.data('tag',[],'float32')
            lr = fluid.layers.data(name = 'lr',shape = [1], dtype='float32', append_batch_size=False)
            ent_coeff = fluid.layers.data('entcoeff',[],'float32')
            self.loss_array = self.algorithm.learn(obs,act,adv,vtag,lr,ent_coeff)
        with fluid.program_guard(self.predict_program):
            obs =fluid.layers.data('obs',[self.config['obsdim']],'float32')
            self.predict_acts = self.algorithm.predict(obs)
            
        with fluid.program_guard(self.sample_program):
            
            obs =fluid.layers.data('obs',[self.config['obsdim']],'float32')
            self.sample_acts = self.algorithm.sample(obs)
        with fluid.program_guard(self.value_program):
            obs =fluid.layers.data('obs',[self.config['obsdim']],'float32')
            self.values = self.algorithm.value(obs)
        
            
            
            
            
        
        