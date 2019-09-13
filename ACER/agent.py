import parl
from paddle import fluid
import numpy as np
from parl.utils.scheduler import PiecewiseScheduler,LinearDecayScheduler

class Agent(parl.Agent):
    def __init__(self,algorithm,config):
        self.obs_dim = config['obsdim']
        self.act_dim = config['actdim']
        super(Agent,self).__init__(algorithm)
        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],config['max_sample_steps'])
        self.entropy_coeff_scheduler = PiecewiseScheduler(config['entropy_coeff_scheduler'])
       
        
    
    def learn(self,obs,act,cv,cqi,crhoi,cf,crho,cq,cqret,c):
        assert len(obs.shape) ==2, "[batch_size,obs_dim]"
        assert obs.shape[1] == self.obs_dim
        assert len(act.shape) ==1  , "[batch_size]"
        assert len(cv.shape) == 1 , " [batch_size]"
        assert len(cqi.shape) == 1 , " [batch_size]"
        assert len(crhoi.shape) == 1 , " [batch_size]"
        assert len(cf.shape) ==2 , " [ batch_size,act_dim]"
        assert cf.shape[1] == self.act_dim
        assert len(cq.shape) ==2 , " [ batch_size,act_dim]"
        assert cq.shape[1] == self.act_dim
        assert len(crho.shape) ==2 , " [ batch_size,act_dim]"
        assert crho.shape[1] == self.act_dim
        assert len(cqret.shape) == 1 , "[batch size]"
        assert len(c.shape) == 1 ,"[batch_size]"

        lr = self.lr_scheduler.step(step_num=obs.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()
        
        
        feed = {
            'obs':obs,
            'act':act,
            'cv':cv,
            'cqi':cqi,
            'rhoi':crhoi,
            'cf':cf,
            'crho':crho,
            'cq':cq,
            'qret':cqret,
            'c':c,
            'lr':np.array([lr],dtype='float32'),
            'entcoeff':np.array([entropy_coeff],dtype='float32')
            
        }
        
        total_loss,pi_loss,ent,qloss= self.fluid_executor.run(self.learn_program,feed=feed,fetch_list=[self.total_loss,self.pi_loss,self.ent,self.qloss])
        return total_loss,pi_loss,ent,qloss
    
    
    def predict(self,obs):
        feed = {'obs':obs}
        
        acts = self.fluid_executor.run(self.predict_program,feed=feed,fetch_list=[self.predict_acts])[0]
        return [int(a) for a in acts]

    
    
    def sample(self,obs):
        feed = {'obs':obs}
        acts,u = self.fluid_executor.run(self.sample_program,feed=feed,fetch_list=[self.sample_acts,
                                                                                                                                                                self.sample_probs])
        
        return [int(a) for a in acts ],u
    

    
        
        
    def build_program(self):
        self.predict_program = fluid.Program()
        self.learn_program = fluid.Program()
        self.qret_program = fluid.Program()
        self.sample_program = fluid.Program()
        
        
        with fluid.program_guard(self.predict_program):
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            self.predict_acts = self.algorithm.predict(obs)
        
        with fluid.program_guard(self.sample_program):
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            self.sample_acts,self.sample_probs= self.algorithm.sample(obs)

        
            
            
        with fluid.program_guard(self.learn_program):
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            act = fluid.layers.data('act',[],'int64')
            constant_v = fluid.layers.data('cv',[],'float32')
            constant_q_i = fluid.layers.data('cqi',[],'float32')
            constant_rho_i = fluid.layers.data('rhoi',[],'float32')
            constant_f = fluid.layers.data('cf',[self.act_dim],'float32')
            constant_rho = fluid.layers.data('crho',[self.act_dim],'float32')
            constant_q = fluid.layers.data('cq',[self.act_dim],'float32')
            constant_qret = fluid.layers.data('qret',[],'float32')
            c = fluid.layers.data('c',[],'float32')
            lr = fluid.layers.data(name = 'lr',shape = [1], dtype='float32', append_batch_size=False)
            ent_coeff = fluid.layers.data(name='entcoeff',shape=[],dtype='float32')
            self.total_loss,self.pi_loss,self.ent,self.qloss = self.algorithm.learn(obs,act,constant_f,constant_q,constant_q_i,constant_rho,
                               constant_rho_i,constant_v,constant_qret,c,lr,ent_coeff)
    


            
            

        with fluid.program_guard(self.qret_program):
            obs = fluid.layers.data('obs',[self.obs_dim],'float32')
            act = fluid.layers.data('act',[],'int64')
            u   = fluid.layers.data('u',[self.act_dim],'float32')
            self.c_v,self.c_q_i,self.c_rho_i,self.c_f,self.c_q,self.c_rho=self.algorithm.qret_params(obs,act,u)
            
           

    def qret_param(self,obs,act,u):
        feed = {
            'obs':obs,
            'act':act,
            'u':u
            
        }
        
       
        v,qi,rhoi,f,q,rho = self.fluid_executor.run(self.qret_program,feed=feed,
            fetch_list=[self.c_v,self.c_q_i,self.c_rho_i,self.c_f,self.c_q,self.c_rho])
        
        return v,qi,rhoi,f,q,rho
    

