
import sys
import os
import json

import munch
import numpy as np
import scipy.io as sio
import pdb
from matplotlib import pyplot as plt

def std_normal(N_data,t_steps, seeds):
    # x0 and t_steps should be 1d array
    np.random.seed(seeds)
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]])
    noise = np.random.normal(0.0, np.sqrt(diff[0]), [t_steps.shape[0]-1, N_data])
    for i in range(t_steps.shape[0]-1):
        grow[:,i+1] = noise[i]
    return grow

def Gendata(T,Nt,N_data,IC_,seeds,mean=False,steady=False):
    #
    #
    # The Ornstein-Uhlenbeck process:
    # dX_t = th(mu-X_t) dt + sig dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    th = 1.0
    mu = 1.2
    sig = 0.3
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        np.random.seed(2)
        xIC = np.random.uniform(0,2.5,N_data)
    elif IC_=='value':
        xIC = 1.5*np.ones(N_data)
    else:
        xIC = IC_
    # data
    data = np.zeros((1,Nt+1,N_data))
    brownian = std_normal(N_data, t, seeds)
    Ext = np.exp(-th*t)
    data[0,:,:] = (xIC[:,None]*Ext+mu*(1-Ext)+\
                   sig*Ext*np.cumsum(np.exp(th*t)*brownian, axis=-1)).T
    if steady:
        Nt = 1
        data = data[:,[0,-1],:]
        data[0][1] -= np.mean(data[0][1])
        # data = np.tile(data,(10,1,1))
    if mean:
        data = (np.mean(data,axis=2)).reshape([1,Nt+1])
    if N_data==1:
        data = data.reshape([1,Nt+1])
    return data

if __name__ == '__main__':

    json_dir = 'jsons/Ex3OU.json'
    with open(json_dir) as json_data_file:
        config = json.load(json_data_file)
    
    config = munch.munchify(config)
    DC = config.data_config
    
    locals().update(DC)
    
    # generate training data
    # for i in range(2):
    #     data_train = np.zeros((N_train[i],n_sample[i],L_train+1))
    #     beta_coef = np.random.uniform(beta_range_1[0],beta_range_1[1],(N_train[i],2))
    #     beta_rescale = np.random.uniform(beta_range_2[0],beta_range_2[1],(N_train[i],2))
    #     for j in range(N_train[i]):
    #         IC = np.random.beta(beta_coef[j,0], beta_coef[j,1], size=n_sample[i])
    #         IC = IC*(beta_rescale[j,1]-beta_rescale[j,0])+beta_rescale[j,0]
    #         data_train[j] = Gendata(T=L_train*dt, Nt=L_train, N_data=n_sample[i],
    #                                 IC_=IC, seeds=j ,steady=False)[0].T
    #     np.save('data/'+eqn_name+'/train_{}.npy'.format(i+1),data_train)
        
    # generate training data
    data_ori = Gendata(T=L_train*dt, Nt=L_train, N_data=N_traj,
                                IC_='uniform', seeds=114514 ,steady=False)[0].T
    
    
    data_re = np.reshape(data_ori[:,:-1], (N_traj*L_train,))
    data_sort_ind = np.argsort(data_re)
    re_indices = np.unravel_index(data_sort_ind, (N_traj,L_train))
    data_sort_0 = data_ori[re_indices]
    data_sort_1 = data_ori[(re_indices[0],re_indices[1]+1)]
    data_sort = np.concatenate((data_sort_0[:, np.newaxis], data_sort_1[:, np.newaxis]),axis = -1)
    
    plt.hist(data_sort_0, histtype = 'step')
    plt.title(eqn_name+' raw data histogram')
    plt.savefig('data/'+eqn_name+'/raw data.jpg')
    plt.show()
    plt.close('all')
    
    # beta
    # for i in range(2):
    #     data_train = np.zeros((N_train[i],n_sample[i],2))
    #     beta_coef = np.random.uniform(beta_range_1[0],beta_range_1[1],(N_train[i],2))
    #     beta_mid = np.random.randint(0,N_traj*L_train,(N_train[i],1))
    #     beta_length = np.random.randint(0,N_traj*L_train/3,(N_train[i],1))
    #     beta_left = np.max(np.concatenate((np.zeros((N_train[i],1)),
    #                                        beta_mid-beta_length),1),1,keepdims=True)
    #     beta_right = np.min(np.concatenate((np.ones((N_train[i],1))*(N_traj*L_train-1),
    #                                         beta_mid+beta_length),1),1,keepdims=True)
    #     beta_rescale = np.concatenate((beta_left,beta_right), axis = 1)
    #     for j in range(N_train[i]):
    #         IC_normal = np.random.beta(beta_coef[j,0], beta_coef[j,1], size=n_sample[i])
    #         IC = IC_normal*(beta_rescale[j,1]-beta_rescale[j,0])+beta_rescale[j,0]
    #         IC_int = IC.astype(np.int32)
    #         data_train[j] = data_sort[IC_int]
    #         if j <20:
    #             plt.hist(data_train[j,:,0], histtype = 'step')
    #     plt.title(eqn_name+'beta {} histogram'.format(n_sample[i]))
    #     plt.savefig('data/'+eqn_name+'/beta_train_{}.jpg'.format(i+1))
    #     plt.show()
    #     plt.close('all')
    #     np.save('data/'+eqn_name+'/beta_train_{}.npy'.format(i+1),data_train)
    
    # uniform
    # for i in range(2):
    #     data_train = np.zeros((N_train[i],n_sample[i],2))
        
    #     for j in range(N_train[i]):
    #         IC_int = np.random.randint(0,N_traj*L_train,(n_sample[i],))
    #         data_train[j] = data_sort[IC_int]
    #         if j <20:
    #             plt.hist(data_train[j,:,0], histtype = 'step')
    #     plt.title(eqn_name+' uniform {} histogram'.format(n_sample[i]))
    #     plt.savefig('data/'+eqn_name+'/uni_train_{}.jpg'.format(i+1))
    #     plt.show()
    #     plt.close('all')
    #     np.save('data/'+eqn_name+'/uni_train_{}.npy'.format(i+1),data_train)
    
    # range
    for i in range(2):
        data_train = np.zeros((N_train[i],n_sample[i],1,2))

        beta_left = np.random.randint(0,N_traj*L_train-n_sample[i],(N_train[i]))
        
        beta_right = beta_left + n_sample[i]

        for j in range(N_train[i]):
            data_train[j,:,0] = data_sort[beta_left[j]:beta_right[j]]
            if j <20:
                plt.hist(data_train[j,:,0,0], histtype = 'step')
        plt.title(eqn_name+'range {} histogram'.format(n_sample[i]))
        plt.savefig('data/'+eqn_name+'/range_train_{}.jpg'.format(i+1))
        plt.show()
        plt.close('all')
        np.save('data/'+eqn_name+'/range_train_{}.npy'.format(i+1),data_train)

    # generate validation data
    # for i in range(2):
    #     data_val = np.zeros((N_val[i],n_sample[i],L_train+1))
    #     beta_coef = np.random.uniform(beta_range_1[0],beta_range_1[1],(N_train[i],2))
    #     beta_rescale = np.random.uniform(beta_range_2[0],beta_range_2[1],(N_train[i],2))
    #     for j in range(N_val[i]):
    #         IC = np.random.beta(beta_coef[j,0], beta_coef[j,1], size=n_sample[i])
    #         IC = IC*(beta_rescale[j,1]-beta_rescale[j,0])+beta_rescale[j,0]
    #         data_val[j]  = Gendata(T=L_train*dt, Nt=L_train, N_data=n_sample[i],
    #                                IC_=IC, seeds=14514+j,mean=False,steady=False)[0].T
    #     np.save('data/'+eqn_name+'/val_{}.npy'.format(i+1),data_val)
    
    # generate test data
    data_test = np.zeros((N_test,n_sample[1],1,L_test+1))
    for j in range(N_test):
        data_test[j,:,0]  = Gendata(T=L_test*dt, Nt=L_test, N_data=n_sample[1],
                                IC_='value', seeds=114514+j, mean=False,steady=False)[0].T
    np.save('data/'+eqn_name+'/test.npy',data_test)

