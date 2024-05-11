from data.data_creation import sample_random_pendulum_data, sample_pendulum_with_input
import jax.numpy as jnp
import jax.random as jr

key = jr.PRNGKey(0)
num_points = 100
u = jr.uniform(key, shape=(num_points, 1), minval=-1, maxval=1)
t, x, x_dot = sample_pendulum_with_input(u, initial_state=jnp.array([-1.0, 0.0, 0.0]))
print(t.shape, x.shape, x_dot.shape)

