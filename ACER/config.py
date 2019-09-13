
from rlschool import LiftSim

from wrapper import  ObservationWrapper,ActionWrapper,Wrapper

env = ObservationWrapper(ActionWrapper(Wrapper(LiftSim())))


acer_config = {}
acer_config['master_addr'] = "localhost:6006"
acer_config['rmsize'] = 50000
acer_config['obsdim'] = env.observation_space
acer_config['actdim'] = env.action_space
acer_config['env'] = env
acer_config['start_lr'] =1e-5
acer_config['max_sample_steps']=int(3e7)
acer_config['num_actor'] = 5
acer_config['qcoeff'] = 0.5
acer_config['sample_batch'] = 8
acer_config['entropy_coeff_scheduler']=[(0.0,-0.01)]
