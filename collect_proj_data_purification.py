import sys
sys.path.append('/Users/utkarshagrawal/Documents/Postdoc/U_1_haar')
import numpy as np
from evolution_utils import scramble
from collect_utils import run_proj, save_data
from analysis_utils import save_S_vs_t

# Don't change these
L_max = 60
T_max = 300

PH = True

# Fixed parameters
L_A = 1
L_list = [8,10,12,14][:]
# L_list = [16]
p_list = np.linspace(0.01,0.16,8)[7:8]
p_list = [0.05,0.15,0.2,0.25,0.3,0.35,0.4]
p_list = [0.2]
g_list = [0.1,0.3,0.5,0.6,0.7,0.8]
g_list = [0.2,0.3,0.4,0.6,0.7,0.8]
initial_state = False
scram_depth = 1
depth = 4
root_direc = 'data_proj_measurement/purification/'
seed_scrambling = 1000
shots = 1000
BC = 'PBC'
to_save = True
collect = True

if collect:
    
    for L in L_list:
        for g in g_list:
            g = round(g,3)
            initial_state = np.zeros((2,)*(L+L_A))
            index_state_0 = (1,0)*(L//2) + (0,)*L_A 
            index_state_1 = (1,0)*(L//2 - 1) + (0,1) + (1,)*L_A
            initial_state[index_state_0] = 1/2**0.5
            initial_state[index_state_1] = 1/2**0.5 
            T = int(depth*L)
            t_scr = int(scram_depth*L)
            initial_state = scramble(initial_state,L,t_scr,g,seed_scrambling)

            for p in p_list:
                p = round(p,3)
                
                data = run_proj(g=g,p=p,L=L,L_A=L_A,shots=shots,T=T,t_scr=0,
                            initial_state=initial_state,BC=BC,PH=PH,
                            seed_scrambling=None,seed_outcome=None,seed_unitary=None
                            )
                
                if to_save:
                    save_data(data,L,T,t_scr,
                            root_direc,g,p,BC,shots)


for p in p_list:
    for g in g_list:
        p = round(p,3)
        g = round(g,3)
    # save_S_vs_t(p,L_list,BC=BC,evolution_type=evolution_type,scrambling_type=scrambling_type,root_direc=root_direc,A=0,depth=depth,scr_depth=scram_depth)

    #half_system_S_vs_t(p,L_list,BC=BC,evolution_type=evolution_type,scrambling_type=scrambling_type,root_direc=root_direc,depth=depth,scr_depth=scram_depth)