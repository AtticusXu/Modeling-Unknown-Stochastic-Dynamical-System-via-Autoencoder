
import sys
import os
import json

import munch
import numpy as np
import scipy.io as sio
import pdb
from matplotlib import pyplot as plt

def std_normal(N_data, t_steps, dim):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]*dim])
    for i in range(t_steps.shape[0]-1):
        grow[:,(i+1)*dim:(i+2)*dim] = np.random.normal(0.0, np.sqrt(diff[i]), [N_data,dim])
    return grow

def EM_auto_md(drift,diffusion,dim,initial,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = Xt+drift(Xt)*diff[i]+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt)))[:,0,:]
    return data

def geneq(dim):
    if dim==2:
        mu = np.array(((0,1),(-1,0)))
        sigma = np.array(((0.0,0),(0,0.1)))
    else:
        raise AttributeError('no info for dim')
    
    def drift(x):
        return x@(mu.T)
    def diff(x):
        return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    return drift,diff

def Gendata2D(T,Nt,N_data,IC_,mean=False):
    #
    #
    # The Multi-dimension OU:
    # dX_t = mu X_t dt + sigma exp(-X_t^2) dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        xIC = np.array((np.random.uniform(-1.5,1.5,N_data),np.random.uniform(-1.5,1.5,N_data))).T
    elif IC_=='value':
        xIC = np.array((0.3*np.ones(N_data),0.4*np.ones(N_data))).T
    # function
    drift,diffu = geneq(2)
    # data
    data = np.zeros((2,Nt+1,N_data))
    datag = EM_auto_md(drift,diffu,2,xIC,t)
    data[0,:,:] = (datag[:,::2]).T
    data[1,:,:] = (datag[:,1::2]).T
    if mean:
        data = (np.mean(data,axis=2)).reshape([1,Nt+1])
    if N_data==1:
        data = data.reshape([1,Nt+1])
    return data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    data_train = Gendata2D(T=0.01, Nt=1, N_data=10000, IC_='uniform')
    data_test  = Gendata2D(T=6.5, Nt=650, N_data=10000,  IC_='value',mean=False)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
