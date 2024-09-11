import numpy as np
import math
import DEAE_network
from scipy.stats import norm
import tensorflow as tf
import random as rn


randseedIDs = {
    "s1": [17, 11, 1263],
    "s2": [25, 2345, 2091],
    "s3": [3, 313, 342],
    "s4": [416, 4159, 444],
    "s5": [572, 5319, 580],
    "s6": [666, 66, 6],
    "s7": [3453, 27, 98],
    "s8": [3478, 687, 1987],
    "s9": [87, 68, 2],
    "s10": [10, 100, 1000],
}  # seedID: [np_seed, rn_seed, tf_seed]


def set_seed_sess(gpu, np_seed, rn_seed, tf_seed):
    """
    Set random seeds and configure GPU/CPU device for TensorFlow.

    This function sets random seeds for NumPy, Python's random module, and TensorFlow
    to ensure reproducibility. It also configures the appropriate device (GPU or CPU)
    for TensorFlow operations.

    Args:
        gpu (int): GPU device index to use. Use -1 for CPU.
        np_seed (int): Seed for NumPy's random number generator.
        rn_seed (int): Seed for Python's random module.
        tf_seed (int): Seed for TensorFlow's random number generator.

    Returns:
        str: Device string for TensorFlow operations (e.g., "/device:GPU:0" or "/CPU:0").

    The function performs the following steps:
    1. Sets random seeds for NumPy, Python's random module, and TensorFlow.
    2. Determines whether to use CPU or GPU based on the 'gpu' parameter.
    3. If using GPU, it configures the specified GPU device and sets memory growth.
    4. Returns the appropriate device string for TensorFlow operations.
    """

    np.random.seed(np_seed)
    rn.seed(rn_seed)
    tf.random.set_seed(tf_seed)

    if gpu == -1:
        device = "/CPU:0"

    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")

        if gpus:
            print("Available GPUs:")
            for gpu_device in gpus:
                print(gpu_device)
        else:
            print("No GPU devices found.")
            return "/CPU:0"

        if gpu < 0 or gpu >= len(gpus):
            print(f"Invalid GPU index {gpu}. Using CPU instead.")
            return "/CPU:0"

        tf.config.experimental.set_visible_devices(gpus[gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu], True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        device = f"/device:GPU:{gpu}"

    return device


def G_1step(x, z, example):
    """
    Perform one step of the stochastic process for a given example.

    Args:
        x (float): Current state of the system.
        z (float): Random normal variable.
        example (str): Name of the example to simulate.

    Returns:
        float: Next state of the system after one time step.

    The function uses a time step of 0.01 and supports various stochastic processes:
    - Ornstein-Uhlenbeck (OU) process
    - Geometric Brownian Motion
    - SDE with nonlinear diffusion
    - Trigonometric drift-diffusion
    - Double well potential
    - Exponential distribution
    - Exponential OU process
    """
    t = 0.01
    sqrt_t = np.sqrt(t)

    def ou_process():
        """Ornstein-Uhlenbeck process"""
        th, mu, sig = 1.0, 1.2, 0.3
        Ext = np.exp(-th * t)
        return x * Ext + mu * (1 - Ext) + sig * sqrt_t * z

    def nonlinear_diffusion():
        """SDE with nonlinear diffusion"""
        mu, sigma = 5, 0.5
        return x + (-mu * x) * t + sigma * np.exp(-(x**2)) * sqrt_t * z

    def geometric_brownian_motion():
        """Geometric Brownian Motion"""
        mu, sigma = 2.0, 1.0
        return x * np.exp((mu - sigma**2 / 2) * t + sigma * sqrt_t * z)

    def trigonometric_drift_diffusion():
        """Trigonometric drift-diffusion process"""
        k, sigma = 1, 0.5
        return (
            x
            + np.sin(2 * k * np.pi * x) * t
            + sigma * np.cos(2 * k * np.pi * x) * sqrt_t * z
        )

    def exponential_ou_process():
        """Exponential Ornstein-Uhlenbeck process"""
        th, mu, sig = 1.0, -0.5, 0.3
        return x ** (1 - th * t) * np.exp(th * mu * t + sig * sqrt_t * z)

    def double_well_potential():
        """Double well potential process"""
        sigma = 0.5
        return x + (x - x**3) * t + sigma * sqrt_t * z

    def exponential_distribution():
        """Exponential distribution process"""
        th, sig = -2.0, 0.1
        return x + th * x * t + sig * Normal2Exp(z) * sqrt_t

    example_functions = {
        "4-1-1": ou_process,
        "4-1-2": geometric_brownian_motion,
        "4-2-1": nonlinear_diffusion,
        "4-2-2": trigonometric_drift_diffusion,
        "4-2-3": double_well_potential,
        "4-3-1": exponential_distribution,
        "4-3-2": exponential_ou_process,
    }

    return example_functions.get(example, lambda: None)()


def HermiteF(x, degree):
    """
    Compute the Hermite polynomial of given degree at x.

    Args:
    x (array-like): Input values.
    degree (int): Degree of the Hermite polynomial.

    Returns:
    array-like: Hermite polynomial values of the specified degree at x.
    """
    if degree == 0:
        return np.ones_like(x)
    elif degree == 1:
        return x
    else:
        return x * HermiteF(x, degree - 1) - (degree - 1) * HermiteF(x, degree - 2)


def Normal2Exp(z):
    """
    Convert standard normal random variable to exponential random variable.

    Args:
    z (array-like): Standard normal random variable.

    Returns:
    array-like: Approximation of exponential random variable.
    """
    b_coef = [
        1.0,
        0.903249477665220,
        0.297985195834260,
        0.0335705586952089,
        -0.00228941798505679,
        -0.000388473483538765,
    ]

    b = 0
    for i in range(6):
        b += HermiteF(z, i) * b_coef[i]

    return b


def Normal2logN(z):
    """
    Convert standard normal random variable to log-normal random variable.

    Args:
    z (array-like): Standard normal random variable.

    Returns:
    array-like: Approximation of log-normal random variable.
    """
    b_coef = np.ones((6, 1)) * np.exp(0.5)

    b = 0
    for i in range(6):
        b += HermiteF(z, i) * b_coef[i] / math.factorial(i)

    return b


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


def sample_nd_ball(n, m):
    """
    Sample points uniformly from an n-dimensional ball with radius 3.

    This function generates a set of points that are uniformly distributed
    within an n-dimensional ball of radius 3, centered at the origin.

    Args:
    n (int): The dimension of the ball.
    m (int): The minimum number of points to generate.

    Returns:
    numpy.ndarray: An array of shape (num_points, n) containing the sampled points,
                   where num_points is at least m.
    """

    N_r = int(pow(m, 1 / n)) + 1
    num_points = 0
    while num_points < m:

        r = [np.linspace(-3, 3, N_r) for i in range(n)]

        grids = np.meshgrid(*r, indexing="ij")

        points = np.stack(grids, axis=-1).reshape(-1, n)

        # Compute the Euclidean distance for each point
        distances = np.linalg.norm(points, axis=1)

        # Select points that are within the n-dimensional ball of radius 3
        inside_ball = points[distances <= 3]
        num_points = inside_ball[:, 0].size
        N_r += 1

    return inside_ball


def moment_preparation(max_moment, moment_weights, batch_size, latent_dim):
    """
    Prepare moment calculations for the normal distribution.

    This function calculates the moments of the normal distribution and reshapes
    them for use in the DEAE model.

    Args:
    max_moment (int): The maximum order of moments to calculate.
    moment_weights (array-like): Weights for each moment.
    batch_size (int): The batch size for processing.
    latent_dim (int): The dimension of the latent space.

    Returns:
    numpy.ndarray: An array of shape (batch_size, max_moment * latent_dim) containing
                   the prepared moments for the normal distribution.
    """
    moment_normal_c = np.zeros((max_moment))
    for m in range(1, max_moment + 1):
        if m % 2 == 0:
            moment_normal_c[m - 1] = (
                2 ** (-m / 2) * np.math.factorial(m) / np.math.factorial(int(m / 2))
            )
    moment_normal_c = moment_normal_c / moment_weights
    moment_normal_c = np.tile(
        moment_normal_c.reshape(1, -1, 1), (batch_size, 1, latent_dim)
    )
    moment_normal_c = np.reshape(moment_normal_c, (batch_size, -1))

    moment_layer = DEAE_network.MomentLayer1D(max_moment, moment_weights)
    return moment_layer, moment_normal_c


def kde_preparation(example, latent_dim, kde_range, kde_num, n_sample, batch_size):
    """
    Prepare Kernel Density Estimation (KDE) layers and probability density functions for different examples.

    This function sets up KDE layers and calculates probability density functions
    for various stochastic processes, including 1D and multi-dimensional cases.

    Args:
    example (str): The example identifier (e.g., "4-1-1", "4-4-1", "4-4-2").
    latent_dim (int): The dimension of the latent space.
    kde_method (str): The KDE method to use (relevant for example "4-4-2").    kde_range (tuple): The range for KDE (min, max).
    kde_num (int): The number of points for KDE.
    n_sample (int): The number of samples.
    batch_size (int): The batch size for processing.

    Returns:
    tuple: Contains KDE layers and probability density functions for 1D and multi-dimensional cases.
           (kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd)
    """
    # Initialize variables
    kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd = (
        None,
        None,
        None,
        None,
    )

    # Common calculations for all examples
    x = np.linspace(kde_range[0], kde_range[1], kde_num)
    norm_pdf = norm.pdf(x)
    pdf_normal_c_1d = np.tile(
        norm_pdf.reshape(1, kde_num, 1), (batch_size, 1, latent_dim)
    )
    kde_layer_1d = DEAE_network.KDELayer1D(
        lower=kde_range[0], upper=kde_range[1], num=kde_num, n_sample=n_sample
    )

    if example[:3] in ["4-1", "4-2", "4-3"]:
        # 1D cases: Only 1D KDE is needed
        pass
    elif example in ["4-4-1", "4-4-2"]:
        # Multi-dimensional cases
        kde_num *= latent_dim
        points = sample_nd_ball(latent_dim, kde_num)
        kde_layer_nd = DEAE_network.KDELayer(points)

        z_ = (1 / (2 * np.pi) ** (latent_dim / 2)) * np.exp(
            -0.5 * np.linalg.norm(points, 2, 1) ** 2
        )

        pdf_normal_c_nd = np.tile(z_.reshape(1, -1), (batch_size, 1))
        pdf_normal_c_nd *= 5.0 ** (latent_dim - 2)

    return kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd
