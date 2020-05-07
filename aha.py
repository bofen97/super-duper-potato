import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
def truth_state(X_k):
    "just using for cache error"
    var = 0.1
    noise = np.random.normal(loc = 0.,scale=var)
    
    return X_k + 2.*noise

def measurements(X_k):
    var = 0.5
    noise = np.random.normal(loc = 0.,scale=var)
    
    return X_k + 4.*noise
    
    

    
def initial_condition():
    return np.random.normal(loc = 0.,scale=0.05),np.random.normal(loc = 0.,scale=0.2)


def clac_Kk_posteriori(Priori_Sk,initial_Sk):
    Kk = Priori_Sk/(Priori_Sk+initial_Sk+1e-4)
    return Kk


def clac_Sk_posteriori(Priori_Sk,Kk):
    return Priori_Sk*(1. - Kk)



def clac_Xk_posteriori(Xk_priori,Kk,Zk):
    return Xk_priori+Kk*(Zk-Xk_priori)



def clac_Xk_priori(Xk_posteriori):
    "No noise"
    return Xk_posteriori



def clac_Sk_priori(Sk_posteriori):
    var = 0.1
    noise = noise = np.random.normal(loc = 0.,scale=var)
    return Sk_posteriori + var



def main():
    k=0
    catches = {}
    catches['truth_xk'] = []
    catches['posteriori_xk'] = []
    catches['measurements'] = []
    
    initial_X,initial_S = initial_condition()
    priori_Sk = np.random.normal(loc = 0.,scale=1.)
    priori_Xk = np.random.normal(loc = 0.,scale=1.)
    truth_Xk = np.copy(initial_X)
    while k<200:
        catches['truth_xk'].append(truth_Xk)
        
        Zk = measurements(truth_Xk)
        
        catches['measurements'].append(Zk)
                
        Kk = clac_Kk_posteriori(priori_Sk,initial_S)
        posteriori_Sk = clac_Sk_posteriori(priori_Sk,Kk)
        posteriori_Xk = clac_Xk_posteriori(priori_Xk,Kk,Zk)
        catches['posteriori_xk'].append(posteriori_Xk)
        priori_Xk = clac_Xk_priori(posteriori_Xk)
        priori_Sk = clac_Sk_priori(posteriori_Sk)
        
        truth_Xk = truth_state(truth_Xk)
        k=k+1
    return catches
    
    
dic = main()
plt.legend(loc = 'upper right')
plt.plot(dic['truth_xk'],color = 'red' ,label= "truth_xk")
plt.plot(dic['posteriori_xk'],color = 'green',linestyle = '-', label ="predict ")
plt.plot(dic['measurements'],color = 'pink', linestyle = '--',label = "measurements")
plt.show()
    
    
    
    


