import numpy as np
import argparse
import json
import munch
import os
from typing import Callable
from sklearn.neighbors import NearestNeighbors


def std_normal_1d(N_data: int, t_steps: np.ndarray) -> np.ndarray:
    diff = t_steps[1:] - t_steps[:-1]
    grow = np.zeros([N_data, t_steps.shape[0]])
    for i in range(t_steps.shape[0] - 1):
        grow[:, i + 1] = np.random.normal(0.0, np.sqrt(diff[i]), N_data)
    return grow


def std_exp_1d(N_data: int, t_steps: np.ndarray) -> np.ndarray:
    diff = t_steps[1:] - t_steps[:-1]
    grow = np.zeros([N_data, t_steps.shape[0]])
    for i in range(t_steps.shape[0] - 1):
        grow[:, i + 1] = np.random.exponential(1.0, N_data)
    return grow


def std_normal_nd(N_data: int, t_steps: np.ndarray, dim: int) -> np.ndarray:
    diff = t_steps[1:] - t_steps[:-1]
    grow = np.zeros([N_data, t_steps.shape[0] * dim])
    for i in range(t_steps.shape[0] - 1):
        grow[:, (i + 1) * dim : (i + 2) * dim] = np.random.normal(
            0.0, np.sqrt(diff[i]), [N_data, dim]
        )
    return grow


def EM_auto_1d(
    condprob: Callable, initial: np.ndarray, t_steps: np.ndarray, noise_func: Callable
) -> np.ndarray:
    data = np.zeros([initial.shape[0], t_steps.shape[0]])
    data[:, 0] = initial
    noise = noise_func(initial.shape[0], t_steps)
    diff = t_steps[1:] - t_steps[:-1]
    for i in range(t_steps.shape[0] - 1):
        Xt = data[:, i]
        data[:, i + 1] = condprob(Xt, diff[i], noise[:, i + 1])
    return data


def EM_auto_nd(
    drift: Callable,
    diffusion: Callable,
    dim: int,
    initial: np.ndarray,
    t_steps: np.ndarray,
    noise_func: Callable,
) -> np.ndarray:
    data = np.zeros([initial.shape[0], dim * t_steps.shape[0]])
    data[:, :dim] = initial
    noise = noise_func(initial.shape[0], t_steps, dim)
    diff = t_steps[1:] - t_steps[:-1]

    for i in range(t_steps.shape[0] - 1):
        Xt = data[:, i * dim : (i + 1) * dim]

        data[:, (i + 1) * dim : (i + 2) * dim] = (
            Xt
            + drift(Xt) * diff[i]
            + ((noise[:, (i + 1) * dim : (i + 2) * dim][:, None, :]) @ (diffusion(Xt)))[
                :, 0, :
            ]
        )
    return data


class SDE1DGenerator:
    def __init__(self, config: munch.Munch):
        self.config = config
        self.eqn_name = self.config.eqn_name

    def _geneq(self) -> Callable:
        if self.eqn_name == "4-3-2":
            th, mu, sig = 1.0, -0.5, 0.3
            return lambda x, dt, dw: x ** (1 - th * dt) * np.exp(
                th * mu * dt + sig * dw
            )
        elif self.eqn_name == "4-3-1":
            th, sig = -2.0, 0.1
            return lambda x, dt, dw: x + th * x * dt + sig * dw * np.sqrt(dt)
        elif self.eqn_name == "4-2-3":
            sigma = 0.5
            return lambda x, dt, dw: x + (x - x**3) * dt + sigma * dw
        elif self.eqn_name == "4-2-2":
            k, sigma = 1, 0.5
            return (
                lambda x, dt, dw: x
                + np.sin(2 * k * np.pi * x) * dt
                + sigma * np.cos(2 * k * np.pi * x) * dw
            )
        elif self.eqn_name == "4-2-1":
            mu, sigma = 5, 0.5
            return lambda x, dt, dw: x - mu * x * dt + sigma * np.exp(-(x**2)) * dw
        else:
            raise ValueError(f"Unknown equation type: {self.eqn_name}")

    def generate_data(self, T: float, Nt: int, N_data: int, mode: str) -> np.ndarray:
        t = np.linspace(0, T, Nt + 1)
        if mode == "train":
            train_IC_range = np.array(self.config.train_IC_range)
            xIC = np.random.uniform(train_IC_range[0], train_IC_range[1], N_data)
        elif mode == "test":
            xIC = self.config.test_IC * np.ones((N_data))
        else:
            raise ValueError(f"Unknown mode type: {mode}")

        data = np.zeros((1, N_data, Nt + 1))
        if self.eqn_name in ["4-3-2", "4-2-1", "4-2-2", "4-2-3"]:
            condprob = self._geneq()
            noise_func = std_normal_1d
            data[0, :, :] = EM_auto_1d(condprob, xIC, t, noise_func)
        elif self.eqn_name == "4-3-1":
            condprob = self._geneq()
            noise_func = std_exp_1d
            data[0, :, :] = EM_auto_1d(condprob, xIC, t, noise_func)
        elif self.eqn_name == "4-1-1":
            mu, sigma = 2.0, 1.0
            brownian = std_normal_1d(N_data, t)
            data[0, :, :] = xIC[:, None] * np.exp(
                (mu - sigma**2 / 2) * t + sigma * brownian
            )
        elif self.eqn_name == "4-1-2":
            th, mu, sig = 1.0, 1.2, 0.3
            Ext = np.exp(-th * t)
            brownian = std_normal_1d(N_data, t)
            data[0, :, :] = (
                xIC[:, None] * Ext
                + mu * (1 - Ext)
                + sig * Ext * np.cumsum(np.exp(th * t) * brownian, axis=-1)
            )

        else:
            raise ValueError(f"Unknown equation type: {self.eqn_name}")
        return data

    def generate_train_data(self) -> None:
        DC = self.config
        data_ori = self.generate_data(
            T=DC.L_train * DC.dt, Nt=DC.L_train, N_data=DC.N_traj, mode="train"
        )[0]
        data_re = np.reshape(data_ori[:, :-1], (DC.N_traj * DC.L_train,))
        data_sort_ind = np.argsort(data_re)
        re_indices = np.unravel_index(data_sort_ind, (DC.N_traj, DC.L_train))
        data_sort_0 = data_ori[re_indices]
        data_sort_1 = data_ori[(re_indices[0], re_indices[1] + 1)]
        data_sort = np.concatenate(
            (data_sort_0[:, np.newaxis], data_sort_1[:, np.newaxis]), axis=-1
        )
        for i in range(2):
            data_train = np.zeros((DC.N_train[i], DC.n_sample[i], 1, 2))

            beta_left = np.random.randint(
                0, DC.N_traj * DC.L_train - DC.n_sample[i], (DC.N_train[i])
            )

            beta_right = beta_left + DC.n_sample[i]

            for j in range(DC.N_train[i]):
                data_train[j, :, 0] = data_sort[beta_left[j] : beta_right[j]]

            os.makedirs(f"data/{self.eqn_name}", exist_ok=True)
            np.save(f"data/{DC.eqn_name}/train_{i + 1}.npy", data_train)

    def generate_test_data(self) -> None:
        DC = self.config
        data_test = np.zeros((DC.N_test, DC.n_sample[1], 1, DC.L_test + 1))
        for j in range(DC.N_test):
            tmp = self.generate_data(
                T=DC.L_test * DC.dt, Nt=DC.L_test, N_data=DC.n_sample[1], mode="test"
            )
            data_test[j, :, 0] = tmp[0]
        os.makedirs(f"data/{self.eqn_name}", exist_ok=True)
        np.save(f"data/{self.eqn_name}/test.npy", data_test)


class OUProcessMultiDGenerator:
    def __init__(self, config: munch.Munch, rank: int = None):
        self.config = config
        self.rank = rank
        if self.rank:
            self.eqn_name = self.config.eqn_name + f"-{self.rank}"
        else:
            self.eqn_name = self.config.eqn_name
        self.mu = self._get_mu()
        self.sigma = self._get_sigma()

    def _get_mu(self) -> np.ndarray:
        if self.config.eqn_name == "4-4-1":
            return np.array(((-1, -0.5), (-1, -1)))
        elif self.config.eqn_name[:5] == "4-4-2":
            return 2 * np.array(
                [
                    [0.1, 0.5, 0.1, 0.2, 0.1],
                    [-0.5, 0.0, 0.1, 0.4, -0.5],
                    [0.1, 0.1, -0.4, -0.6, 0.1],
                    [-0.3, 0.0, 0.6, -0.1, 0.3],
                    [0.1, 0.1, 0.3, 0.2, 0.0],
                ]
            )
        else:
            raise ValueError(f"Unknown example type: {self.config.eqn_name}")

    def _get_sigma(self) -> np.ndarray:

        if self.eqn_name == "4-4-1":
            return np.array(((1, 0), (0, 0.5)))

        elif self.eqn_name == "4-4-2-1":
            sigma = np.zeros((5, 5))
            sigma[3, 3] = 1.0
            return sigma
        elif self.eqn_name == "4-4-2-2":
            return np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.8, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -0.8],
                ]
            )
        elif self.eqn_name == "4-4-2-3":
            return np.array(
                [
                    [0.8, 0.2, 0.0, 0.0, 0.0],
                    [-0.4, 0.6, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.7, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        elif self.eqn_name == "4-4-2-4":
            return np.array(
                [
                    [0.7, 0.0, -0.4, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.6, 0.2, -0.1],
                    [0.0, 0.0, 0.1, -0.6, 0.2],
                    [0.0, 0.0, 0.0, 0.3, 0.8],
                ]
            )
        elif self.eqn_name == "4-4-2-5":
            return np.array(
                [
                    [0.8, 0.2, 0.1, -0.3, 0.1],
                    [-0.3, 0.6, 0.1, 0.0, -0.1],
                    [0.2, -0.1, 0.9, 0.1, 0.2],
                    [0.1, 0.1, -0.2, 0.7, 0.0],
                    [-0.1, 0.1, 0.1, -0.1, 0.5],
                ]
            )
        else:
            raise ValueError(f"Unknown example type: {self.config.eqn_name}")

    def drift(self, x: np.ndarray) -> np.ndarray:
        return x @ (self.mu.T)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.sigma[None, :, :], x.shape[0], axis=0)

    def generate_data(self, T: float, Nt: int, N_data: int, mode: str) -> np.ndarray:
        t = np.linspace(0, T, Nt + 1)
        if mode == "train":
            train_IC_range = np.array(self.config.train_IC_range)
            uniform_random = np.random.uniform(0, 1, (N_data, self.config.dim))
            train_IC_length = train_IC_range[:, 1] - train_IC_range[:, 0]
            xIC = train_IC_range[:, 0] + train_IC_length * uniform_random
        elif mode == "test":
            xIC = np.array(self.config.test_IC) * np.ones((N_data, self.config.dim))
        else:
            raise ValueError(f"Unknown mode type: {mode}")

        data = np.zeros((self.config.dim, Nt + 1, N_data))
        datag = EM_auto_nd(
            self.drift, self.diffusion, self.config.dim, xIC, t, std_normal_nd
        )
        for i in range(self.config.dim):
            data[i, :, :] = (datag[:, i :: self.config.dim]).T

        return data

    def generate_train_data(self) -> None:
        DC = self.config
        data_ori = self.generate_data(
            T=DC.L_train * DC.dt, Nt=DC.L_train, N_data=DC.N_traj, mode="train"
        )

        data_re = np.reshape(
            data_ori[:, :-1, :], (self.config.dim, DC.N_traj * DC.L_train)
        ).T

        for i in range(2):
            data_c = np.array(
                np.random.uniform(-4, 4, (DC.N_train[i], self.config.dim))
            )

            neighbors = NearestNeighbors(
                n_neighbors=DC.n_sample[i], algorithm="auto"
            ).fit(data_re)
            distances, indices = neighbors.kneighbors(data_c)
            re_indices = np.unravel_index(indices, (DC.L_train, DC.N_traj))
            data_sort_0 = np.empty((DC.N_train[i], DC.n_sample[i], self.config.dim))
            data_sort_1 = np.empty((DC.N_train[i], DC.n_sample[i], self.config.dim))
            for s in range(self.config.dim):
                data_sort_0[:, :, s] = data_ori[s][re_indices]
                data_sort_1[:, :, s] = data_ori[s][(re_indices[0] + 1, re_indices[1])]

            data_sort = np.concatenate(
                (data_sort_0[:, :, :, np.newaxis], data_sort_1[:, :, :, np.newaxis]),
                axis=-1,
            )
            # Ensure the data directory exists before saving
            os.makedirs(f"data/{self.eqn_name}", exist_ok=True)
            np.save(
                f"data/{self.eqn_name}/train_{i + 1}.npy",
                data_sort,
            )

    def generate_test_data(self) -> None:
        DC = self.config
        data_test = np.zeros(
            (DC.N_test, DC.n_sample[1], self.config.dim, DC.L_test + 1)
        )
        for j in range(DC.N_test):
            tmp = self.generate_data(
                T=DC.L_test * DC.dt, Nt=DC.L_test, N_data=DC.n_sample[1], mode="test"
            )
            for s in range(DC.n_sample[1]):
                data_test[j, s] = tmp[:, :, s]
        os.makedirs(f"data/{self.eqn_name}", exist_ok=True)
        np.save(f"data/{self.eqn_name}/test.npy", data_test)


def load_config(json_path: str) -> munch.Munch:
    with open(json_path, "r") as f:
        config = json.load(f)
    return munch.munchify(config)


def generate_data_for_example(Example):
    json_dir = f"configs/Ex{Example}.json"
    config = load_config(json_dir)

    if Example[:3] in ["4-3", "4-2", "4-1"]:
        generator = SDE1DGenerator(config.data_config)
        generator.generate_train_data()
        generator.generate_test_data()
    elif Example == "4-4-1":
        generator = OUProcessMultiDGenerator(config.data_config)
        generator.generate_train_data()
        generator.generate_test_data()
    elif Example == "4-4-2":
        for rank in range(1, 6):
            generator = OUProcessMultiDGenerator(config.data_config, rank=rank)
            generator.generate_train_data()
            generator.generate_test_data()


def main():
    parser = argparse.ArgumentParser(description="Generate data for SDE examples")
    parser.add_argument(
        "--example",
        type=str,
        default="all",
        help="Example to generate data for (e.g., 4-1-2, 4-3-1, 4-4-1, 4-4-2, or 'all' for all examples)",
    )
    args = parser.parse_args()

    seeds = 114514
    np.random.seed(seeds)

    all_examples = [
        "4-1-1",
        "4-1-2",
        "4-2-1",
        "4-2-2",
        "4-2-3",
        "4-3-1",
        "4-3-2",
        "4-4-1",
        "4-4-2",
    ]

    if args.example.lower() == "all":
        for Example in all_examples:
            print(f"Generating data for Example {Example}")
            generate_data_for_example(Example)
    else:
        generate_data_for_example(args.example)


if __name__ == "__main__":
    main()
