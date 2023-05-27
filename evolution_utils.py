import time
from Circuits.circuit_evolution import evenlayer, oddlayer, weak_measurement_layer, measurement_layer
import Circuits.entanglement as entanglement
import unitary_sampler
import numpy as np
L_max = 60
T_max = 300


def scramble(state,L,t_scr,g,seed_scram):
    assert t_scr < T_max
    if t_scr == 0:
        return state

    unitary_gates = unitary_sampler.weak_interacting_gates(g,seed_scram,L//2*t_scr)

    for t in range(t_scr):
        # even layer
        if t%2 == 0:
            state = evenlayer(state,unitary_gates[L//2*t: L//2*t + L//2],L)
        else:
            state = oddlayer(state,unitary_gates[L//2*t: L//2*t + L//2],L,BC='PBC')

    return state


def do_unitary_layer(state,t,U_list,L,BC):
    if t%2 == 0:
        state = evenlayer(state,U_list,L=L)
    else:
        state = oddlayer(state,U_list,L=L,BC=BC)
    return state


def get_weak_data(state,L,L_A,T,t_scr,g,theta,seed_scram,seed_unitary,seed_outcomes,BC='PBC'):


    GATES_NEEDED = L//2 * T
    state = scramble(state,L,t_scr,g,seed_scram=seed_scram)

    unitary_gates = unitary_sampler.weak_interacting_gates(g,seed_unitary,GATES_NEEDED)

    cache = {}

    outcome_rng = np.random.default_rng(seed=seed_outcomes)
    total_N_m = 0

    entropy_data = []
    correlation_data = []
    outcomes = []
    for t in range(T):
        # do unitary layer
        U_list = unitary_gates[L//2 * t: L//2 * t + L//2]
        state = do_unitary_layer(state,t,U_list,L,BC)
        
        # do measurement
        m_locations = list(range(L)) # locations of weak measurement
        total_N_m += L
        state, outcome = weak_measurement_layer(state,theta,L,rng_outcome=outcome_rng,m_locations=m_locations)
        outcomes.append(outcome)

        #collect_data
        entropy = get_entropy(L,L_A,state)
        entropy_data.append(entropy)
        correlation_data.append(state_correlation(state,L))
    
    cache['total_N_m'] = total_N_m
    cache['outcomes'] = outcomes
    
    return state, np.array(entropy_data), correlation_data, cache


def get_proj_data(state,L,L_A,T,t_scr,g,p,seed_scram,seed_unitary,seed_outcomes,BC='PBC'):
    

    GATES_NEEDED = L//2 * T
    state = scramble(state,L,t_scr,g,seed_scram=seed_scram)

    unitary_gates = unitary_sampler.weak_interacting_gates(g,seed_unitary,GATES_NEEDED)
    
    cache = {}

    outcome_rng = np.random.default_rng(seed=seed_outcomes)
    total_N_m = 0

    locations = []
    outcomes = []
    entropy_data = []
    correlation_data = []
    for t in range(T):
         # do unitary layer
        U_list = unitary_gates[L//2 * t: L//2 * t + L//2]
        state = do_unitary_layer(state,t,U_list,L,BC)
        
        # do measurement
        m_locations = np.where(outcome_rng.uniform(0,1,L)<p)[0] # locations of measurement
        total_N_m += len(m_locations)
        state, outcome = measurement_layer(state,rng_outcome=outcome_rng,m_locations=m_locations)
        locations.append(m_locations)
        outcomes.append(outcome)
        
        if np.any(np.isnan(state)):
            print(seed_outcomes,t,L)

        #collect data
        try: entropy = get_entropy(L,L_A,state)
        except:
            print('Oh oh!',state.shape,state,np.any(np.isnan(state.flatten())),np.any(np.isinf(state.flatten())))
        entropy_data.append(np.array(entropy))
        correlation_data.append(state_correlation(state,L))
    
    cache['total_N_m'] = total_N_m
    cache['m_locations'] = locations
    cache['outcomes'] = outcomes
    
    return state, np.array(entropy_data), correlation_data, cache


def state_entropy(state,interval,renyi_index):
    # start = time.time()
        
    B = list(interval)
    entropy_data = entanglement.renyi_entropy(state,B,renyi_index=renyi_index)
    

    # print("end_time=", time.time() - start)
    return entropy_data


def get_entropy(L,L_A,state):
    entropy_list = []
    # for A in range(0,L//2+1,1):
    for A in range(0,1,1):
        interval = list(range(L-A,L+L_A)) # Interval to calculate entropy of
        # A = 0 correponds to entropy of ancilla 
        entropy_list.append(state_entropy(state,interval=interval,renyi_index=1))
    return entropy_list

def state_correlation(state,L):
    
    C = np.zeros((L,L))
  
    for i in range(L):
        for j in range(i,L,1):
            if i==j:
                C[i,i] = 1
                continue
            x = i
            y = j
            state = np.swapaxes(state,x,0)
            state = np.swapaxes(state,y,1)
            C[i,j] = -np.sum(2*(np.abs(state[0,1,:])**2 + np.abs(state[1,0,:])**2))
            C[j,i] = C[i,j]
            state = np.swapaxes(state,x,0)
            state = np.swapaxes(state,y,1)
    return C
