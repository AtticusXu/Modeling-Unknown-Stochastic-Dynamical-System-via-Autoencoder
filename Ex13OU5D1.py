
import sys
import os
import json

import munch
import numpy as np
import scipy.io as sio
import pdb
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

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
    if dim==5:
        mu  = 2* np.array([
    [0.1, 0.5, 0.1, 0.2, 0.1],
    [-0.5, 0.0, 0.1, 0.4, -0.5],
    [0.1, 0.1, -0.4, -0.6, 0.1],
    [-0.3, 0.0, 0.6, -0.1, 0.3],
    [0.1, 0.1, 0.3, 0.2, 0.0],
])
        
        sigma = np.zeros((5,5))
        sigma[3,3] = 1.0
        # sigma = np.zeros((5,5))
    else:
        raise AttributeError('no info for dim')
    
    def drift(x):
        return x@(mu.T)
    def diff(x):
        return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    return drift,diff

def Gendata5D(T,Nt,N_data,IC_,steady = False, mean=False):
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
    dim = 5
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        xIC = np.array(np.random.uniform(-4,4,(N_data,5)))
    elif IC_=='value':
        xIC = np.array((0.3*np.ones(N_data),
                        -0.2*np.ones(N_data),
                        -1.7*np.ones(N_data),
                        2.5*np.ones(N_data),
                        1.4*np.ones(N_data))).T
    # function
    drift,diffu = geneq(dim)
    # data
    data = np.zeros((dim,Nt+1,N_data))
    datag = EM_auto_md(drift,diffu,dim,xIC,t)
    for i in range(dim):
        data[i,:,:] = (datag[:,i::dim]).T
    if steady:
        Nt = 1
        data = data[:,[0,-1],:]
        # data[0][1] -= np.mean(data[0][1])
        # data = np.tile(data,(10,1,1))
    if mean:
        data = (np.mean(data,axis=2)).reshape([1,Nt+1])
    if N_data==1:
        data = data.reshape([1,Nt+1])
    return data

if __name__ == '__main__':
    
    # neighbors = NearestNeighbors(n_neighbors=10000, algorithm='auto').fit(A)

    # # Find the 10,000 nearest neighbors for each point in B
    # distances, indices = neighbors.kneighbors(B)
    
    json_dir = 'jsons/Ex13OU5D1.json'
    with open(json_dir) as json_data_file:
        config = json.load(json_data_file)
    
    config = munch.munchify(config)
    DC = config.data_config
    
    locals().update(DC)
    save_dir = 'data/'+eqn_name+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # generate training data
    data_ori = Gendata5D(T=L_train*dt, Nt=L_train, N_data=N_traj,
                                IC_='uniform')
    t = np.linspace(0,L_train*dt,L_train+1)
    
    for i in range(5):
        plt.plot(t,data_ori[i,:,0])
        plt.title(eqn_name+' raw data histogram')
        # plt.savefig('data/'+eqn_name+'/raw data.jpg')
        plt.show()
        plt.close('all')
        
        
    data_re = np.reshape(data_ori[:,:-1,:], (5,N_traj*L_train)).T
    
    for i in range(5):
        plt.hist(data_re[:,i],bins = 100)
        plt.title(eqn_name+' raw data histogram')
        # plt.savefig('data/'+eqn_name+'/raw data.jpg')
        plt.show()
        plt.close('all')
#%%
    for i in range(2):
        
        data_c = np.array(np.random.uniform(-4,4,(N_train[i],5)))
        # index_c = np.random.randint(0,N_traj*L_train,(N_train[i]//2))
        # data_c[::2] = data_re[index_c]
        
        neighbors = NearestNeighbors(n_neighbors=n_sample[i], algorithm='auto').fit(data_re)
        distances, indices = neighbors.kneighbors(data_c)
        re_indices = np.unravel_index(indices, (L_train,N_traj))
        data_sort_0 = np.empty(( N_train[i], n_sample[i],5))
        data_sort_1 = np.empty(( N_train[i], n_sample[i],5))
        for s in range(5):
            data_sort_0[:,:,s] = data_ori[s][re_indices]
            data_sort_1[:,:,s] = data_ori[s][(re_indices[0]+1,re_indices[1])]
    
        data_sort = np.concatenate((data_sort_0[:,:,:, np.newaxis], data_sort_1[:,:,:, np.newaxis]),axis = -1)
        np.save('data/'+eqn_name+'/neighbor_space_train_{}.npy'.format(i+1),data_sort)
        for j in range(100):
            plt.scatter(data_sort[j,:,0,0], data_sort[j,:,4,0], s=0.1)
        plt.title(eqn_name+' neighbor {} scatter'.format(n_sample[i]))
        plt.savefig('data/'+eqn_name+'/neighbor_space_train_{}.jpg'.format(i+1),dpi=600)
        plt.show()
        plt.close('all')

        for t in range(n_sample[i]):
            plt.plot(data_sort[0,t,0,:], data_sort[0,t,1,:], c = 'yellow', linewidth=0.3)
        plt.scatter(data_sort[0,:,0,0], data_sort[0,:,1,0], c = 'red', s=0.5,label = 'x(0)')
        plt.scatter(data_sort[0,:,0,1], data_sort[0,:,1,1], c = 'blue', s=0.5,label = 'x(dt)')
        plt.title(eqn_name+' neighbor {} 1 step trajectories'.format(n_sample[i]))
        plt.legend()
        plt.savefig('data/'+eqn_name+'/neighbor_train_traj_{}.jpg'.format(i+1),dpi=600)
        plt.show()
        plt.close('all')
#%%   
    # generate test data
    data_test = np.zeros((N_test,n_sample[1],5,L_test+1))
    for j in range(N_test):
        tmp = Gendata5D(T=L_test*dt, Nt=L_test, N_data=n_sample[1],
                                IC_='value')
        for s in range(n_sample[1]):
            data_test[j,s] = tmp[:,:,s]
    
    np.save('data/'+eqn_name+'/test.npy',data_test)
        
#%%
testdata =data_test.reshape((-1,5,501))
testdata = testdata[:,5]
xt_test = np.arange(testdata.shape[-1])*dt
xmean_test = np.mean(testdata,axis=0)
xstde_test = np.std(testdata,axis=0,ddof=1)
test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test

from matplotlib import pyplot as plt
fig1, ax1 = plt.subplots(figsize=[10,7])
ax1.plot(xt_test, xmean_test, color='#4169E1', label='Ground Truth')
ax1.fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
ax1.legend()
plt.show()