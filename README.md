# Modeling Unknown Stochastic Dynamical System via Autoencoder

This project contains the code for the paper "Modeling Unknown Stochastic Dynamical System via Autoencoder" by Zhongshu Xu, Yuan Chen, Qifan Chen, and Dongbin Xiu. The paper is available as a preprint on arXiv: [arXiv:2312.10001](https://arxiv.org/abs/2312.10001).

If you use this code in your research, please cite our paper:
```
Xu, Z., Chen, Y., Chen, Q., & Xiu, D. (2023). Modeling unknown stochastic dynamical system via autoencoder. arXiv preprint arXiv:2312.10001.
```
## Installation

This project uses Poetry for dependency management. To set up the project, follow these steps:

1. Make sure you have Poetry installed. If not, install it by following the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).

2. Clone this repository:
   ```
   git clone https://github.com/AtticusXu/Modeling-Unknown-Stochastic-Dynamical-System-via-Autoencoder.git
   cd Modeling-Unknown-Stochastic-Dynamical-System-via-Autoencoder
   ```

3. Install the project dependencies using Poetry:
   ```
   poetry install
   ```

This will create a virtual environment and install all the necessary packages specified in the `pyproject.toml` file.

## Generating Data

To generate data for the stochastic dynamical systems, use the `data_generate.py` script in the `data_generation` folder. You can generate data for specific examples or for all examples at once.

1. Activate the Poetry virtual environment:
   ```
   poetry shell
   ```

2. To generate data for all examples:
   ```
   python data_generation/data_generate.py --example all
   ```

3. To generate data for a specific example (e.g., 4-1-2):
   ```
   python data_generation/data_generate.py --example 4-1-2
   ```

The generated data will be saved in the `data` folder, organized by example name.

## Available Examples

The following table lists the available examples and their corresponding names:

| Example | Name |
|---------|------|
| 4-1-1   | OU Process |
| 4-1-2   | Geometric Brownian Motion |
| 4-2-1   | SDE with nonlinear diffusion |
| 4-2-2   | Trig process |
| 4-2-3   | Double Well Potential |
| 4-3-1   | Noises with Exponential Distribution |
| 4-3-2   | Noise with Lognormal Distribution |
| 4-4-1   | Two-dimensional Ornstein-Uhlenbeck process |
| 4-4-2   | Five-dimensional Ornstein-Uhlenbeck process |


## Training the Model

To train the model for a specific example, use the `DEAE_train.py` script. This script handles the training process for the Deep Ensemble Autoencoder (DEAE) model.

1. Ensure you're in the Poetry virtual environment:
   ```
   poetry shell
   ```

2. Run the training script with the desired parameters:
   ```
   python DEAE_train.py --Example <example_id> --ensemble_seedID <seed_id> --gpu <gpu_number> --verbose <verbosity_level>
   ```

   Parameters:
   - `--Example`: The example identifier (e.g., "4-1-1", "4-4-2")
   - `--ensemble_seedID`: Ensemble seed ID (default: "s1")
   - `--gpu`: GPU device number to use (default: 0)
   - `--verbose`: Verbosity level for training output (default: 1)

   For example, to train the model for example 4-1-1 for ensemble seed ID "s1" using GPU 0:
   ```
   python DEAE_train.py --Example 4-1-1 --ensemble_seedID s1 --gpu 0
   ```

3. The training process consists of two steps, each with its own sample size and number of epochs as specified in the configuration file. The script will:
   - Load the configuration from the corresponding JSON file in the `configs/` directory
   - Load the training data
   - Create and compile the DEAE model
   - Train the model for each step, saving checkpoints and loss history
   - Save the final trained model weights

4. The trained models and associated files will be saved in the `models/` directory, organized by example and ensemble seed ID.

You can monitor the training progress through the console output. The script will display the model summary, training progress, and validation loss for each epoch.



## Project Structure

- `data_generation/`: Contains scripts for generating data
- `configs/`: Configuration files for different examples
- `data/`: Generated data (will be created when running the data generation script)
- `DEAE_lib.py`: Library for DEAE model
- `DEAE_network.py`: Network architecture for DEAE model
- `DEAE_train.py`: Training script for DEAE model
- `DEAE_test.py`: Testing script for DEAE model