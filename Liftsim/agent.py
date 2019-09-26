from parl.utils.scheduler import PiecewiseScheduler,LinearDecayScheduler
import parl
import numpy as np
from paddle import fluid

class Agent(parl.Agent):
    
    def __init__(self,alg,config):
        assert isinstance(alg,parl.Algorithm),"Not define algorithm "
        self.obs_shape = config['obs_shape']
        
        super(Agent, self).__init__(alg)

        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_sample_steps'])

        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])
    def sample(self,obs):
        
        feed = {'obs' : obs.astype('float32')}
        
        sample_acts,values = self.fluid_executor.run(self.sample_program,feed=feed,fetch_list = [self.sample_acts, \
                                                                           self.sample_values])

    
        return sample_acts,values
    def predict(self,obs):
        
        feed = {'obs' : obs.astype('float32')}
        
        predict_acts = self.fluid_executor.run(self.predict_program,feed=feed,fetch_list = [self.predict_acts])[0]
        return predict_acts
    def value(self,obs):
        
        feed = {'obs' : obs.astype('float32')}
        
        values = self.fluid_executor.run(self.value_program,feed=feed,fetch_list = [self.values])[0]
        return values
    def learn(self,obs,act,adv,vtag):
        
        
        lr = self.lr_scheduler.step(step_num=obs.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()
        feed={
            'obs':obs.astype('float32'),
            'act':act.astype('int64'),
            'adv':adv.astype('float32'),
            'vtag':vtag.astype('float32'),
            'lr' : np.array([lr],dtype='float32'),
            'entropy_coeff':np.array([entropy_coeff],dtype='float32')
            
        }
        
        total_loss, pi_loss, vf_loss, entropy=self.fluid_executor.run(self.learn_program,feed=feed,fetch_list=self.learn_outputs)
        return total_loss,pi_loss,vf_loss,entropy,lr,entropy_coeff
        
    def build_program(self):
        
        self.sample_program = fluid.Program()
        self.predict_program = fluid.Program()
        self.value_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.sample_program):
            
            obs = fluid.layers.data('obs',self.obs_shape,'float32')
            self.sample_acts, self.sample_values = self.algorithm.sample(obs)
            
        with fluid.program_guard(self.predict_program):
            obs = fluid.layers.data('obs',self.obs_shape,'float32')
            self.predict_acts = self.algorithm.predict(obs)

        with fluid.program_guard(self.value_program):
            obs = fluid.layers.data('obs',self.obs_shape,'float32')
            self.values = self.algorithm.value(obs)

        with fluid.program_guard(self.learn_program):
            obs = fluid.layers.data('obs',self.obs_shape,'float32')
            act = fluid.layers.data('act',[],'int64')
            adv = fluid.layers.data('adv',[],'float32')
            vtag = fluid.layers.data('vtag',[],'float32')
            
            lr = fluid.layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = fluid.layers.data(
                name='entropy_coeff', shape=[], dtype='float32')

            total_loss, pi_loss, vf_loss, entropy = self.algorithm.learn(
                obs, act, adv, vtag, lr, entropy_coeff)
            self.learn_outputs = [total_loss, pi_loss, vf_loss, entropy]