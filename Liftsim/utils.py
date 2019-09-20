import numpy as np
import scipy

def test(agent):
    from wrapper import Wrapper,ActionWrapper,ObservationWrapper
    from rlschool import LiftSim
    mansion_env = LiftSim()
    mansion_env = Wrapper(mansion_env)
    mansion_env = ActionWrapper(mansion_env)
    mansion_env = ObservationWrapper(mansion_env)

    obs = mansion_env.reset()
    rs=0.0
    for i in range(28800*6*3):
        mansion_env.render()
        acts = agent.predict(obs)
        obs,r,_,_ = mansion_env.step([int(act) for act in acts])
        rs+= r
    print('test  28800 * 6 * 3  steps reward mean {}'.format(rs/3.))

        


def calc_gae_for_multi_agent(concat_rews,concat_values,next_values,gamma,lam):
    baseline  = np.append(concat_values,next_values)
    tds  = concat_rews + gamma*baseline[4:] - baseline[:-4]
    
    tds_reshape = np.reshape(tds,(-1,4))
    advantages = np.zeros_like(tds_reshape)
    advantage =0.
    for i in reversed(range(len(tds_reshape))):
        advantage = gamma * lam*tds_reshape[i] + advantage
        advantages[i] = advantage    
    advantages = np.reshape(advantages,tds.shape)
    return advantages
def calc_gae_for_multi_agent_2(concat_rews,concat_values,gamma,lam):
    baseline  = np.append(concat_values,concat_values[-1])
    tds  = concat_rews + gamma*baseline[1:] - baseline[:-1]
    
    advantages =  scipy.signal.lfilter([1],[1,-gamma*lam],tds[::-1], axis=0)[::-1]

    return advantages