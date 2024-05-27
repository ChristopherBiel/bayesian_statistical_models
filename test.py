import jax.numpy as jnp
import jax.random as jr
from data_functions.data_creation import create_example_data
from data_functions.data_handling import split_dataset

data, _ = create_example_data(1, 0.0, jr.PRNGKey(420), 0.0, 64, 128)
data, num_trajectories = split_dataset(data)
print(f"Data inputs shape: {data.inputs.shape}")
print(f"Data outputs shape: {data.outputs.shape}")