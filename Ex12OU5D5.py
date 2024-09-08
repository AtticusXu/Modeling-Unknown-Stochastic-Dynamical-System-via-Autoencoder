import os
import json
import munch
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


def std_normal(N_data, t_steps, dim):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:] - t_steps[:-1]
    grow = np.zeros([N_data, t_steps.shape[0] * dim])
    for i in range(t_steps.shape[0] - 1):
        grow[:, (i + 1) * dim : (i + 2) * dim] = np.random.normal(
            0.0, np.sqrt(diff[i]), [N_data, dim]
        )
    return grow


def EM_auto_md(drift, diffusion, dim, initial, t_steps):
    data = np.zeros([initial.shape[0], dim * t_steps.shape[0]])
    data[:, :dim] = initial
    noise = std_normal(initial.shape[0], t_steps, dim)
    diff = t_steps[1:] - t_steps[:-1]
    for i in range(t_steps.shape[0] - 1):
        Xt = data[:, i * dim : (i + 1) * dim]
        print(f"noise shape: {noise.shape}")
        print(f"i: {i}, dim: {dim}")
        print(f"diffusion(Xt) shape: {diffusion(Xt).shape}")
        data[:, (i + 1) * dim : (i + 2) * dim] = (
            Xt
            + drift(Xt) * diff[i]
            + ((noise[:, (i + 1) * dim : (i + 2) * dim][:, None, :]) @ (diffusion(Xt)))[
                :, 0, :
            ]
        )
    return data


def geneq(dim):
    if dim == 5:
        mu = 2 * np.array(
            [
                [0.1, 0.5, 0.1, 0.2, 0.1],
                [-0.5, 0.0, 0.1, 0.4, -0.5],
                [0.1, 0.1, -0.4, -0.6, 0.1],
                [-0.3, 0.0, 0.6, -0.1, 0.3],
                [0.1, 0.1, 0.3, 0.2, 0.0],
            ]
        )

        sigma = np.array(
            [
                [0.8, 0.2, 0.1, -0.3, 0.1],
                [-0.3, 0.6, 0.1, 0.0, -0.1],
                [0.2, -0.1, 0.9, 0.1, 0.2],
                [0.1, 0.1, -0.2, 0.7, 0.0],
                [-0.1, 0.1, 0.1, -0.1, 0.5],
            ]
        )
        # sigma = np.zeros((5,5))
    else:
        raise AttributeError("no info for dim")

    def drift(x):
        return x @ (mu.T)

    def diff(x):
        return np.repeat(sigma[None, :, :], x.shape[0], axis=0)

    return drift, diff


def Gendata5D(T, Nt, N_data, IC_, steady=False, mean=False):
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
    t = np.linspace(0, T, Nt + 1)
    # initial condition - can be changed
    if IC_ == "uniform":
        xIC = np.array(np.random.uniform(-4, 4, (N_data, 5)))
    elif IC_ == "value":
        xIC = np.array(
            (
                0.3 * np.ones(N_data),
                -0.2 * np.ones(N_data),
                -1.7 * np.ones(N_data),
                2.5 * np.ones(N_data),
                1.4 * np.ones(N_data),
            )
        ).T
    # function
    drift, diffu = geneq(dim)
    # data
    data = np.zeros((dim, Nt + 1, N_data))
    datag = EM_auto_md(drift, diffu, dim, xIC, t)
    for i in range(dim):
        data[i, :, :] = (datag[:, i::dim]).T
    if steady:
        Nt = 1
        data = data[:, [0, -1], :]
        # data[0][1] -= np.mean(data[0][1])
        # data = np.tile(data,(10,1,1))
    if mean:
        data = (np.mean(data, axis=2)).reshape([1, Nt + 1])
    if N_data == 1:
        data = data.reshape([1, Nt + 1])
    return data


if __name__ == "__main__":

    # neighbors = NearestNeighbors(n_neighbors=10000, algorithm='auto').fit(A)

    # # Find the 10,000 nearest neighbors for each point in B
    # distances, indices = neighbors.kneighbors(B)

    json_dir = "configs/Ex12OU5D5.json"
    with open(json_dir) as json_data_file:
        config = json.load(json_data_file)

    config = munch.munchify(config)
    DC = config.data_config

    save_dir = "data/" + DC.eqn_name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # generate training data
    data_ori = Gendata5D(
        T=DC.L_train * DC.dt, Nt=DC.L_train, N_data=DC.N_traj, IC_="uniform"
    )
    t = np.linspace(0, DC.L_train * DC.dt, DC.L_train + 1)

    data_re = np.reshape(data_ori[:, :-1, :], (5, DC.N_traj * DC.L_train)).T

    # %%
    for i in range(2):

        data_c = np.array(np.random.uniform(-4, 4, (DC.N_train[i], 5)))

        neighbors = NearestNeighbors(n_neighbors=DC.n_sample[i], algorithm="auto").fit(
            data_re
        )
        distances, indices = neighbors.kneighbors(data_c)
        re_indices = np.unravel_index(indices, (DC.L_train, DC.N_traj))
        data_sort_0 = np.empty((DC.N_train[i], DC.n_sample[i], 5))
        data_sort_1 = np.empty((DC.N_train[i], DC.n_sample[i], 5))
        for s in range(5):
            data_sort_0[:, :, s] = data_ori[s][re_indices]
            data_sort_1[:, :, s] = data_ori[s][(re_indices[0] + 1, re_indices[1])]

        data_sort = np.concatenate(
            (data_sort_0[:, :, :, np.newaxis], data_sort_1[:, :, :, np.newaxis]),
            axis=-1,
        )
        np.save(
            "data/" + DC.eqn_name + "/neighbor_space_train_{}.npy".format(i + 1),
            data_sort,
        )
    # %%
    # generate test data
    data_test = np.zeros((DC.N_test, DC.n_sample[1], 5, DC.L_test + 1))
    for j in range(DC.N_test):
        tmp = Gendata5D(
            T=DC.L_test * DC.dt, Nt=DC.L_test, N_data=DC.n_sample[1], IC_="value"
        )
        for s in range(DC.n_sample[1]):
            data_test[j, s] = tmp[:, :, s]

    np.save("data/" + DC.eqn_name + "/test.npy", data_test)
