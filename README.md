# Modeling Unknown Stochastic Dynamical System via Autoencoder

This project contains the code for the paper "Modeling Unknown Stochastic Dynamical System via Autoencoder" by Zhongshu Xu.

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
| 4-1-2   | Double Well |
| 4-2-1   | Cubic Drift |
| 4-2-2   | Nonlinear Drift |
| 4-2-3   | Nonlinear Diffusion |
| 4-3-1   | Lorenz System |
| 4-3-2   | Duffing Oscillator |
| 4-4-1   | Multidimensional OU Process |
| 4-4-2   | Multidimensional OU Process with Rank Deficiency |



## Project Structure

- `data_generation/`: Contains scripts for generating data
- `configs/`: Configuration files for different examples
- `data/`: Generated data (will be created when running the data generation script)