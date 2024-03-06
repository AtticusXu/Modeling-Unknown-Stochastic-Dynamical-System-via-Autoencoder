# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:45:31 2023

@author: xzs16
"""
import numpy as np
import matplotlib.pyplot as plt
import array_to_latex as a2l
# mse_loss = np.array([[1.5e-7, 2.6e-7, 4.4e-7, 3.7e-7, 2.8e-7],
#                      [3.1e-4, 4.9e-7, 2.5e-7, 3.0e-7, 2.4e-7,],
#                      [1.8e-3, 2.7e-4, 4.2e-7, 4.0e-7, 3.4e-7],
#                      [1.5e-3, 7.6e-4, 2.4e-4, 2.7e-7, 4.8e-7],
#                      [3.3e-3, 1.2e-3, 7.2e-4, 4.7e-4, 1.8e-7]])
# z_dim = np.linspace(1, 5,5)
# m = ['o', 'v', '^', 's', 'x']
# for i in range(5):
#     plt.semilogy(z_dim, mse_loss[i,:], marker =m[i], label = f'noise dimention = {i+1}')

# plt.ylabel('mse loss at the last epoch')
# plt.xlabel('dimention of z')
# plt.xticks(z_dim)
# plt.legend(fontsize = 8 )

relative_error = np.zeros((5,5,501,2))
for i in range(5):
    print(f"N{i+1}_testmean.npy")
    test_mean = np.load(f"N{i+1}_testmean.npy")
    test_std = np.load(f"N{i+1}_testdstd.npy")
    for j in range(5):
        pred_mean = np.load(f"N{i+1}Z{j+1}_predmean.npy")
        pred_std = np.load(f"N{i+1}Z{j+1}_predstd.npy")
        
        relative_error[i,j,:,0] = np.linalg.norm(pred_mean-test_mean,2,0)/np.linalg.norm(test_mean,2,0)
        relative_error[i,j,:,1] = np.linalg.norm(pred_std-test_std,2,0)/np.linalg.norm(test_std,2,0)
    
#%%
max_relative_error = np.max(relative_error,2)

# print(max_relative_error[:,:,0].T)
a2l.to_ltx(max_relative_error[:,:,0].T, frmt = '{:.2E}', arraytype = 'tabular') 

# print(max_relative_error[:,:,1].T)
a2l.to_ltx(max_relative_error[:,:,1].T, frmt = '{:.2E}', arraytype = 'tabular')
#%%


a2l.to_ltx(max_relative_error[:,:,0].T, frmt = '{:6.3e}', arraytype = 'array')

mu  = 2* np.array([
    [0.1, 0.5, 0.1, 0.2, 0.1],
    [-0.5, 0.0, 0.1, 0.4, -0.5],
    [0.1, 0.1, -0.4, -0.6, 0.1],
    [-0.3, 0.0, 0.6, -0.1, 0.3],
    [0.1, 0.1, 0.3, 0.2, 0.0],
])
a2l.to_ltx(mu, frmt = '{:6.1f}', arraytype = 'matrix')

b3 =  np.array([
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [-0.4, 0.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
a2l.to_ltx(b3, frmt = '{:.1f}', arraytype = 'matrix')

b4 = np.array([
            [0.7, 0.0, -0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.6, 0.2, -0.1],
            [0.0, 0.0, 0.1, -0.6, 0.2],
            [0.0, 0.0, 0.0, 0.3, 0.8],
        ])
a2l.to_ltx(b4, frmt = '{:.1f}', arraytype = 'matrix')

b5 = np.array([
            [0.8, 0.2, 0.1, -0.3, 0.1],
            [-0.3, 0.6, 0.1, 0.0, -0.1],
            [0.2, -0.1, 0.9, 0.1, 0.2],
            [0.1, 0.1, -0.2, 0.7, 0.0],
            [-0.1, 0.1, 0.1, -0.1, 0.5],
        ])
a2l.to_ltx(b5, frmt = '{:.1f}', arraytype = 'matrix')
#%%
import json

examples = ['Ex13OU5D1', 'Ex14OU5D2', 'Ex11OU5D', 'Ex15OU5D4', 'Ex12OU5D5']
mse_loss_end_avg = np.zeros((5,5))
for k in range(5):
    example = examples[k]
    for i in range(5):
        for j in range(10):
            json_dir = example+f'/models/z{i+1}_best/RNN1/sample10000/test{j+1}loss history.json'
            with open(json_dir) as json_data_file:
                loss = json.load(json_data_file)
            mse_loss_end_avg[i,k] += loss["val_MSE_s0"][-1]/10
    
a2l.to_ltx(mse_loss_end_avg, frmt = '{:.2E}', arraytype = 'matrix') 
print(mse_loss_end_avg)