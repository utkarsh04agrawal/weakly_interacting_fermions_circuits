import os
import pickle
import numpy as np
import time
from evolution_utils import get_proj_data, get_weak_data
from analysis_utils import get_filename

def run_proj(g,p,L,L_A,shots,T,t_scr,initial_state:np.ndarray,PH=False,BC='PBC',seed_outcome=None,seed_unitary=None,seed_scrambling=None):
    
    p = np.round(p,3)
        
    start = time.time()
    entropy_data = []
    correlation_data = []

    seed_o = seed_outcome
    seed_u = seed_unitary
    for _ in range(shots):
        
        state = initial_state.copy()
    
        # get data
        if seed_outcome is None:
            seed_o = np.random.randint(0,1000000000,1)
        if seed_unitary is None:
            seed_u = np.random.randint(0,1000000000,1)

        state,entropy,correlation,_ = get_proj_data(state,L,L_A,T,t_scr,g,p,seed_unitary=seed_u,seed_scram=seed_scrambling,seed_outcomes=seed_o,BC=BC)
        
        entropy_data.append(entropy)
        correlation_data.append(correlation)
        
    data = {'entropy':entropy_data,
            'correlation':correlation_data,
            'seed_outcome': seed_outcome,
            'seed_unitary':seed_unitary,
            'seed_scrambling':seed_scrambling}
    print("L={}, g={}, p={}, time={}".format(L,g,p,time.time()-start))

    return data


def run_weak(g,theta,L,L_A,shots,T,t_scr,initial_state:np.ndarray,PH=False,BC='PBC',seed_outcome=None,seed_unitary=None,seed_scrambling=None):
    
    # theta = np.round(theta,3)
        
    start = time.time()
    entropy_data = []
    correlation_data = []

    seed_o = seed_outcome
    seed_u = seed_unitary

    for _ in range(shots):
        
        state = initial_state.copy()
    
        # get data
        if seed_outcome is None:
            seed_o = np.random.randint(0,1000000000,1)
        if seed_unitary is None:
            seed_u = np.random.randint(0,1000000000,1)

        state,entropy,correlation,_ = get_weak_data(state,L,L_A,T,t_scr,g,theta,seed_unitary=seed_u,seed_scram=seed_scrambling,seed_outcomes=seed_o,BC=BC)

        entropy_data.append(entropy)
        correlation_data.append(correlation)
        
    data = {'entropy':entropy_data,
            'correlation':correlation_data,
            'seed_outcome': seed_outcome,
            'seed_unitary':seed_unitary,
            'seed_scrambling':seed_scrambling}
    print("L={}, p={}, time={}".format(L,theta*2/np.pi,time.time()-start))

    return data
    
    


def save_data(data,L,T,t_scr,root_direc,g,p,BC,shots):
    file_dir = get_filename(L,T,t_scr,p,g,BC,root_direc)
    # if os.path.isdir(file_dir):
    #     shutil.rmtree(file_dir)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    filename = file_dir+'/shots='+str(shots)+'_'+str(np.random.randint(0,1000000000,1))
    
    with open(filename,'wb') as f:
        pickle.dump(data,f)