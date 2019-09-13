
from rlschool import LiftSim

from wrapper import  ObservationWrapper,ActionWrapper,Wrapper

env = ObservationWrapper(ActionWrapper(Wrapper(LiftSim())))

config = {}
config['clip_value'] = 0.2
config['v_coeff'] =0.5
config['max_sample_steps'] = 5e8
config['entropy_coeff_scheduler']=[(0.0,-0.01)]
config['obsdim'] = env.observation_space
config['actdim']=env.action_space
config['env'] = env
config['gamma'] = 0.995
config['lam'] =1.0
config['master_addr'] = "localhost:6006"
config['num_actor'] = 10
config['env_num']=5
config['sample_batch_steps'] = 5
config['start_lr'] =1e-5
