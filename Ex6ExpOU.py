import sys
import os
import json

import munch
import numpy as np
import scipy.io as sio
import pdb
from matplotlib import pyplot as plt


def std_normal(N_data, t_steps):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:] - t_steps[:-1]
    grow = np.zeros([N_data, t_steps.shape[0]])
    for i in range(t_steps.shape[0] - 1):
        grow[:, i + 1] = np.random.normal(0.0, np.sqrt(diff[i]), N_data)
    return grow


def EM_auto_1d(condprob, initial, t_steps):
    data = np.zeros([initial.shape[0], t_steps.shape[0]])
    data[:, 0] = initial
    noise = std_normal(initial.shape[0], t_steps - 1)
    diff = t_steps[1:] - t_steps[:-1]
    for i in range(t_steps.shape[0] - 1):
        Xt = data[:, i]
        data[:, i + 1] = condprob(Xt, diff[i], noise[:, i + 1])
    return data


def geneq(th, mu, sig):
    def condprob(x, dt, dw):
        return x ** (1 - th * dt) * np.exp(th * mu * dt + sig * dw)

    return condprob


def Gendata(T, Nt, N_data, IC_, mean=False):
    #
    #
    # The Geometric Brownian Motion:
    # dX_t = mu X_t dt + sigma exp(-X_t^2) dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    #
    th = 1.0
    mu = -0.5
    sig = 0.3
    t = np.linspace(0, T, Nt + 1)
    # initial condition - can be changed
    if IC_ == "uniform":
        xIC = np.random.uniform(0.1, 2.0, N_data)
    elif IC_ == "value":
        xIC = 0.4 * np.ones(N_data)
    # function
    condprob = geneq(th, mu, sig)
    # data
    data = np.zeros((1, Nt + 1, N_data))
    data[0, :, :] = (EM_auto_1d(condprob, xIC, t)).T
    if mean:
        data = (np.mean(data, axis=2)).reshape([1, Nt + 1])
    if N_data == 1:
        data = data.reshape([1, Nt + 1])
    return data


if __name__ == "__main__":

    seeds = 114514
    np.random.seed(seeds)

    json_dir = "configs/Ex4-3-2.json"
    with open(json_dir) as json_data_file:
        config = json.load(json_data_file)

    config = munch.munchify(config)
    DC = config.data_config

    locals().update(DC)
    save_dir = "data/" + eqn_name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # generate training data
    data_ori = Gendata(T=L_train * dt, Nt=L_train, N_data=N_traj, IC_="uniform")[0].T
    print(data_ori.shape)
    data_re = np.reshape(data_ori[:, :-1], (N_traj * L_train,))
    data_sort_ind = np.argsort(data_re)
    re_indices = np.unravel_index(data_sort_ind, (N_traj, L_train))
    data_sort_0 = data_ori[re_indices]
    data_sort_1 = data_ori[(re_indices[0], re_indices[1] + 1)]
    data_sort = np.concatenate(
        (data_sort_0[:, np.newaxis], data_sort_1[:, np.newaxis]), axis=-1
    )

    plt.hist(data_sort_0, histtype="step")
    plt.title(eqn_name + " raw data histogram")
    plt.show()
    plt.close("all")

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
        data_train = np.zeros((N_train[i], n_sample[i], 2))

        beta_left = np.random.randint(0, N_traj * L_train - n_sample[i], (N_train[i]))

        beta_right = beta_left + n_sample[i]

        for j in range(N_train[i]):
            data_train[j] = data_sort[beta_left[j] : beta_right[j]]
            if j < 20:
                plt.hist(data_train[j, :, 0], histtype="step")
        plt.title(eqn_name + "range {} histogram".format(n_sample[i]))
        plt.savefig("data/" + eqn_name + "/range_train_{}.jpg".format(i + 1))
        plt.show()
        plt.close("all")
        print(data_train)
        # np.save("data/" + eqn_name + "/range_train_{}.npy".format(i + 1), data_train)

    # generate validation data
    # for i in range(2):
    #     data_val = np.zeros((N_val[i],n_sample[i],L_train+1))
    #     beta_coef = np.random.uniform(beta_range_1[0],beta_range_1[1],(N_train[i],2))
    #     beta_rescale = np.random.uniform(beta_range_2[0],beta_range_2[1],(N_train[i],2))
    #     for j in range(N_val[i]):
    #         IC = np.random.beta(beta_coef[j,0], beta_coef[j,1], size=n_sample[i])
    #         IC = IC*(beta_rescale[j,1]-beta_rescale[j,0])+beta_rescale[j,0]
    #         data_val[j]  = Gendata(T=L_train*dt, Nt=L_train, N_data=n_sample[i],
    #                                 IC_=IC, seeds=14514+j,mean=False,steady=False)[0].T
    #     np.save('data/'+eqn_name+'/val_{}.npy'.format(i+1),data_val)

    # generate test data
    # data_test = np.zeros((N_test,n_sample[1],1,L_test+1))
    # for j in range(N_test):
    #     data_test[j,:,0]  = Gendata(T=L_test*dt, Nt=L_test, N_data=n_sample[1],
    #                             IC_='value', mean=False)[0].T
    # np.save('data/'+eqn_name+'/test.npy',data_test)

    data_test = np.zeros((N_test, n_sample[1], 1, 1 + 1))
    for j in range(N_test):
        data_test[j, :, 0] = Gendata(
            T=1 * dt, Nt=1, N_data=n_sample[1], IC_="value", mean=False
        )[0].T
    np.save("data/" + eqn_name + "/1step.npy", data_test)
