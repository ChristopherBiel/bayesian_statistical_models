import jax
import jax.numpy as jnp
import jax.random as jr
import chex

from systems.pendulum_system import PendulumSystem
from bsm.utils.normalization import Data

def example_function(t):
    x = jnp.concatenate([jnp.sin(t) * jnp.cos(0.2*t),
                         0.04*jnp.power(t, 2) + 0.25*t + 1.4,
                         0.3*jnp.sin(t)], axis=1)
    return x 

def example_function_derivative(t):
    x_dot = jnp.concatenate([jnp.sin(t) * (-0.2) * jnp.sin(0.2*t) + jnp.cos(t) * jnp.cos(0.2*t),
                             0.08*t + 0.25,
                             0.3*jnp.cos(t)], axis=1)
    return x_dot

def sample_example_trajectory(num_points, noise_level, d_l, d_u, key):
    t = jnp.linspace(d_l, d_u, num_points, dtype=jnp.float32).reshape(-1, 1)
    x = example_function(t)
    x_dot_true = example_function_derivative(t)
    x = x + noise_level * jr.normal(key=key, shape=x.shape)
    return t, x, x_dot_true

def create_example_data(num_trajectories, noise_level, key, d_l,
               min_samples=48, max_samples=64):
    # Create trajectories of varying length with varying number of points
    keys = jr.split(key, num_trajectories)
    num_points = jr.randint(keys[0], shape=(num_trajectories,), minval=min_samples, maxval=max_samples+1)
    print(f"Number of points: {num_points}")
    t, x, x_dot = sample_example_trajectory(max_samples, noise_level, d_l, d_l+(num_points[0]-1)/10, keys[0])
    for i in range(num_trajectories-1):
        t_traj, x_traj, x_dot_traj = sample_example_trajectory(max_samples, noise_level, d_l, d_l+(num_points[i+1]-1)/10, keys[i+1])
        t = jnp.concatenate([t, t_traj], axis=0)
        x = jnp.concatenate([x, x_traj], axis=0)
        x_dot = jnp.concatenate([x_dot, x_dot_traj], axis=0)
    return Data(inputs=t, outputs=x), Data(inputs=jnp.concatenate([t, x], axis=1), outputs=x_dot)


def sample_random_pendulum_data(num_points: int,
                         noise_level: chex.Array | float | None,
                         key: jr.PRNGKey,
                         num_trajectories: int | None,
                         initial_states: chex.Array | None) -> (chex.Array, chex.Array, chex.Array, chex.Array):
    if num_trajectories is None and initial_states is None:
        num_trajectories = 1
        initial_states = jnp.array([[-1.0, 0.0, 0.0]])
    elif num_trajectories is None:
        num_trajectories = initial_states.shape[0]
    elif initial_states is None:
        initial_states = jnp.array([[-1.0, 0.0, 0.0]] * num_trajectories)
    
    reset_keys = jr.split(key, num_trajectories + 1)
    key = reset_keys[0]
    reset_keys = reset_keys[1:]

    system = PendulumSystem()
    # There is no real randomness in the reset function
    # The reset keys are used to vectorize the reset function
    system_state = jax.vmap(system.reset)(reset_keys)
    system_state.x_next = initial_states

    t = jnp.arange(num_points).reshape(-1, 1) * system_state.system_params.dynamics_params.dt
    t = t.T.reshape(num_trajectories, -1, 1)
    assert t.shape == (num_trajectories, num_points, 1)
    x = jnp.zeros((num_trajectories, num_points, 2))
    u = jnp.zeros((num_trajectories, num_points, 1))
    x_dot = jnp.zeros((num_trajectories, num_points, 1))

    for i in range(num_points):
        action_key, key = jr.split(key, 2)
        actions = jr.uniform(key=action_key, shape=(num_trajectories, 1),
                             minval=-1, maxval=1)
        system_state = jax.vmap(system.step)(system_state.x_next, actions, system_state.system_params)
        x = x.at[:, i, :].set(system_state.x_next[:, :2])
        u = u.at[:, i, :].set(actions)
        x_dot = x_dot.at[:, i, :].set(system_state.x_next[:, 2:])

    return t, x, u, x_dot

def sample_pendulum_with_input(control_input: chex.Array,
                               initial_state: chex.Array = jnp.array([-1.0, 0.0, 0.0]),
                               ) -> (chex.Array, chex.Array, chex.Array):
    chex.assert_shape(control_input, (None, 1))
    chex.assert_shape(initial_state, (3,))
    system = PendulumSystem()
    system_state = system.reset(jr.PRNGKey(0))
    system_state.x_next = initial_state
    num_points = control_input.shape[0]
    t = jnp.arange(num_points).reshape(-1, 1) * system_state.system_params.dynamics_params.dt
    assert t.shape == (num_points, 1)
    x = jnp.zeros((num_points, 2))
    x_dot = jnp.zeros((num_points, 1))
    for i in range(num_points):
        system_state = system.step(system_state.x_next, control_input[i], system_state.system_params)
        x = x.at[i, :].set(system_state.x_next[:2])
        x_dot = x_dot.at[i, :].set(system_state.x_next[2])
    return t, x, x_dot

if __name__ == '__main__':
    num_points = 100
    num_trajectories = 48
    key = jr.PRNGKey(0)
    t, x, u, x_dot = sample_random_pendulum_data(num_points=num_points,
                                              noise_level=0.1,
                                              key=key,
                                              num_trajectories=num_trajectories,
                                              initial_states=None)

    # Plot the data
    import matplotlib.pyplot as plt
    print(f"Time shape: {t.shape}")
    print(f"State shape: {x.shape}")
    fig, axes = plt.subplots(3, min(3,num_trajectories), figsize=(16, 9))
    for i in range(min(3,num_trajectories)):
        axes[0][i].plot(x[i, :, 1], x[i, :, 0])
        axes[0][i].set_xlabel(r'sin($\theta$)')
        axes[0][i].set_ylabel(r'cos($\theta$)')
        axes[0][i].set_title('Trajectory')
        axes[1][i].plot(t[i,:].reshape(-1), x[i, :, 0], label=r'cos($\theta$)')
        axes[1][i].plot(t[i,:].reshape(-1), x[i, :, 1], label=r'sin($\theta$)')
        axes[1][i].plot(t[i,:].reshape(-1), u[i, :, 0], label='Control Input')
        axes[1][i].set_title('State')
        axes[1][i].legend()
        axes[1][i].set_xlabel('Time')
        axes[2][i].plot(t[i,:].reshape(-1), x_dot[i, :, 0])
        axes[2][i].set_title('Derivative')
        axes[2][i].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('out/pendulum_data.png')