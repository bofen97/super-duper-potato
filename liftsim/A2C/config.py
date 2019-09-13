
from rlschool import LiftSim

from wrapper import  ObservationWrapper,ActionWrapper,Wrapper

env = ObservationWrapper(ActionWrapper(Wrapper(LiftSim())))

config = {}
config['v_coeff'] =0.5
config['gamma'] = 0.9
config['lam'] =1.0
config['max_sample_steps'] = int(2e7)
config['entropy_coeff_scheduler']=[(0.0,-0.01)]
config['obsdim'] = env.observation_space
config['actdim']=env.action_space
config['env'] = env
config['master_addr'] = "localhost:6006"
config['num_actor'] = 10
config['env_num']=5
config['sample_batch_steps'] = 5
config['start_lr'] =5e-6
