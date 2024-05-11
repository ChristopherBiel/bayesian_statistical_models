import jax
import jax.numpy as jnp
import jax.random as jr
import chex

from systems.pendulum_system import PendulumSystem

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