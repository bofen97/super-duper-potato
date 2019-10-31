from parl.utils.scheduler import PiecewiseScheduler,LinearDecayScheduler
import parl
import numpy as np
from paddle import fluid

class Agent(parl.Agent):
    
    def __init__(self,alg,hyperparam):
        self.obs_dim = hyperparam['obs_dim']
        self.act_dim = hyperparam['act_dim']
        self.smooth_weight = hyperparam['smooth_weight']
        self.lr = hyperparam['lr']
        super(Agent, self).__init__(alg)
        

    def sync_weights(self):
        self.algorithm.sync_target_v(self.smooth_weight)

    def sample(self,obs):
        noise = np.random.normal(loc=0.0,scale=1.0,size=(obs.shape[0],self.act_dim))
        feed = {'obs' : obs.astype('float32'),
          'noise':noise.astype("float32")}
        
        action,logp ,mean,entropy= self.fluid_executor.run(self.sample_program,feed=feed,fetch_list = self.sample_outputs)
        return action,logp,mean,entropy
    def q_value(self,obs,act):
        
        feed = {'obs' : obs.astype('float32'),
                        'act':act.astype('float32')}
        
        q1,q2 = self.fluid_executor.run(self.q_program,feed=feed,fetch_list = self.q_outputs)
        return q1,q2

    def v_value(self,obs):
        
        feed = {'obs' : obs.astype('float32')}
        
        values = self.fluid_executor.run(self.value_program,feed=feed,fetch_list = [self.v_output])[0]
        return values
    def learn(self,obs,act,rew,next_obs,done):
        
        batch_sz = obs.shape[0]
        noise = np.random.normal(loc=0.0,scale=1.0,size=(batch_sz,self.act_dim))
        lr = self.lr
        feed={
            'obs':obs.astype('float32'),
            'act':act.astype('float32'),
            'rew':rew.astype('float32'),
            'next_obs':next_obs.astype('float32'),
            'done':done.astype("float32"),
            'noise':noise.astype("float32"),
            'lr' : np.array([lr],dtype='float32'),
            
        }
        
        
        total_loss,pi_loss,q1_loss,q2_loss,v_loss,entropy=self.fluid_executor.run(self.learn_program,feed=feed,fetch_list=self.learn_outputs)
        return total_loss,pi_loss,q1_loss,q2_loss,v_loss,entropy
        
    def build_program(self):
        
        self.sample_program = fluid.Program()
        self.value_program = fluid.Program()
        self.q_program = fluid.Program()


        self.learn_program = fluid.Program()

        with fluid.program_guard(self.sample_program):
            
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            noise = fluid.layers.data('noise',[self.act_dim],'float32')
            action,logp,mean,entropy = self.algorithm.sample(obs,noise)
            self.sample_outputs = [action,logp,mean,entropy]

            
        with fluid.program_guard(self.q_program):
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            actions = fluid.layers.data('act',[self.act_dim],"float32")
            q1= self.algorithm.q1_value(obs,actions)
            q2 = self.algorithm.q2_value(obs,actions)
            self.q_outputs = [q1,q2]


            
        with fluid.program_guard(self.value_program):
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            self.v_output = self.algorithm.v_value(obs)


        with fluid.program_guard(self.learn_program):
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            act = fluid.layers.data('act',[self.act_dim],"float32")
            rew = fluid.layers.data('rew',[],'float32')
            next_obs = fluid.layers.data('next_obs',[self.obs_dim],'float32')
            done = fluid.layers.data('done',[],'float32')
            noise = fluid.layers.data('noise',[self.act_dim],'float32')

            lr = fluid.layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)


            total_loss,pi_loss,q1_loss,q2_loss,v_loss,entropy= self.algorithm.learn(
                obs,noise,rew,next_obs,done,act,lr)
            self.learn_outputs = [ total_loss,pi_loss,q1_loss,q2_loss,v_loss,entropy]