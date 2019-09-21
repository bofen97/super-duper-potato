import numpy as np
import scipy
def calc_gae(concat_rews,concat_values,gamma,lam):
    baseline  = np.append(concat_values,concat_values[-1])
    tds  = concat_rews + gamma*baseline[1:] - baseline[:-1]
    
    advantages =  scipy.signal.lfilter([1],[1,-gamma*lam],tds[::-1], axis=0)[::-1]

    return advantages