import os
import pickle
import numpy as np
import matplotlib.pyplot as pl

def get_filedir(t_scr,root_direc):
    file_dir = root_direc

    if t_scr==0:
        file_dir = file_dir + 'without_scrambling' 
    else:
        file_dir = file_dir + 'with_scrambling'

    return file_dir


def get_filename(L,T,t_scr,p,g,BC,root_direc):
    
    file_dir = get_filedir(t_scr=t_scr,root_direc=root_direc)
       
    file_dir = file_dir + '/L='+str(L)
    file_dir = file_dir+'/T='+str(T)+'_tscr='+str(t_scr)+'_g='+str(g)+'_p='+str(p)+'_BC='+BC

    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    return file_dir


    
    
def read_entropy(L,T,t_scr,g,theta,BC,root_direc):
    file_dir = get_filename(L,T,t_scr=t_scr,p=theta,g=g,BC=BC,root_direc=root_direc)
    file_dir = file_dir+'/T='+str(T)+'_tscr='+str(t_scr)+'_p='+str(theta)+'_BC='+BC+'/indiviudal_files/'
    entropy_list = []
    for filename in os.listdir(file_dir):
        with open(file_dir + '/' + filename,'rb') as f:
            data = pickle.load(f)
        entropy_list.extend(data['entropy'])
    entropy = np.array(entropy_list)
    return entropy


def get_slope(xdata,ydata):
    zz = np.polyfit(xdata,ydata,1,full=True)
    p = np.poly1d(zz[0])
    return zz[0], p

def save_S_vs_t(g,theta,L_list,BC,root_direc,A=0,depth=4,scr_depth=1):
    pl.figure(1)
    slope = []
    for L in L_list:
        T = int(depth*L)
        t_scr = int(scr_depth*L)
        entropy = read_entropy(L,T,t_scr,theta,BC=BC,root_direc=root_direc)
        ydata = np.average(entropy[:,:,A],axis=0)
        err = np.std(entropy[:,:,A],axis=0)/(entropy.shape[0]-1)**0.5
        xdata = np.arange(0,T,1)
        zz,p = get_slope(xdata[2*L:3*L],np.log(ydata[2*L:3*L]))
        slope.append(-zz[0])
        # print(entropy[-1])
        pl.errorbar(xdata[:]/L,ydata[:],yerr=err[:],ls='-',marker='o',label='L={}'.format(L))
        pl.plot(xdata[:]/L,np.exp(p(np.array(xdata))),'k',ls=':')

    pl.yscale('log')
    pl.ylabel(r'$S_A(t)$',fontsize=16)
    pl.xlabel(r'$t/L$',fontsize=16)
    pl.title(r'$p={}$'.format(theta),fontsize=16)
    pl.legend()
    save_file = get_filedir(t_scr,root_direc) + '/figures/S_vs_t_A='+str(A)+'_BC='+str(BC)+'/'
    if not os.path.isdir(save_file):
        os.makedirs(save_file)
    pl.savefig(save_file+'theta={}_g={}.pdf'.format(theta,g))
    pl.close(1)

    pl.figure(2)
    pl.plot(L_list[:],[1/i for i in slope],'-o')
    pl.ylabel(r'$\tau$',fontsize=16)
    pl.xlabel(r'$L$',fontsize=16)
    pl.yscale('log')
    pl.savefig(save_file+'decay_rate_theta={}.pdf'.format(theta))
    pl.close(2)


