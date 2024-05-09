import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from bsm.utils.normalization import Data
from SmootherNet import SmootherNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble

def createTrajectory(num_points, noise_level, d_l, d_u, key):
    t = jnp.linspace(d_l, d_u, num_points, dtype=jnp.float32).reshape(-1, 1)
    x = f(t)
    x_dot_true = f_dot(t)
    x = x + noise_level * jr.normal(key=key, shape=x.shape)
    return t, x, x_dot_true

def createData(num_trajectories, noise_level, key, d_l,
               min_samples=48, max_samples=64):
    # Create trajectories of varying length with varying number of points
    keys = jr.split(key, num_trajectories)
    num_points = jr.randint(keys[0], shape=(num_trajectories,), minval=min_samples, maxval=max_samples+1)
    print(f"Number of points: {num_points}")
    t, x, x_dot = createTrajectory(max_samples, noise_level, d_l, d_l+(num_points[0]-1)/10, keys[0])
    for i in range(num_trajectories-1):
        t_traj, x_traj, x_dot_traj = createTrajectory(max_samples, noise_level, d_l, d_l+(num_points[i+1]-1)/10, keys[i+1])
        t = jnp.concatenate([t, t_traj], axis=0)
        x = jnp.concatenate([x, x_traj], axis=0)
        x_dot = jnp.concatenate([x_dot, x_dot_traj], axis=0)
    return Data(inputs=t, outputs=x), Data(inputs=jnp.concatenate([t, x], axis=1), outputs=x_dot)

def splitDataset(data: Data) -> (list[Data], int):
        """Splits the full Dataset into the individual trajectories,
        based on the timestamps (every time there is a jump backwards in the timestamp the data is cut)
        The output is still only one dataset, but now with an additional dimension, which is the number of trajectories."""
        t = data.inputs
        assert t.shape[1] == 1
        delta_t = jnp.diff(t, axis=0)
        indices = jnp.where(delta_t < 0.0)[0] + 1
        ts = jnp.split(t, indices)
        xs = jnp.split(data.outputs, indices)
        print(f"Splitting data into {len(ts)} trajectories.")
        # To be able to stack, all the arrays need to have the same shape
        # Therefore, we need to pad the arrays with zeros
        max_length = max([len(traj) for traj in ts])
        for i in range(len(ts)):
            ts[i] = jnp.pad(ts[i], ((0, max_length - len(ts[i])), (0, 0)), mode='wrap')
            xs[i] = jnp.pad(xs[i], ((0, max_length - len(xs[i])), (0, 0)), mode='wrap')
        inputs = jnp.stack(ts, axis=0)
        outputs = jnp.stack(xs, axis=0)
        return Data(inputs=inputs, outputs=outputs), len(ts)

def f(t):
    x = jnp.concatenate([jnp.sin(t) * jnp.cos(0.2*t),
                         0.04*jnp.power(t, 2) + 0.25*t + 1.4,
                         0.3*jnp.sin(t)], axis=1)
    return x 

def f_dot(t):
    x_dot = jnp.concatenate([jnp.sin(t) * (-0.2) * jnp.sin(0.2*t) + jnp.cos(t) * jnp.cos(0.2*t),
                             0.08*t + 0.25,
                             0.3*jnp.cos(t)], axis=1)
    return x_dot

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_traj = 12
    noise_level = 0.1
    d_l, d_u = 0, 10
    key = jr.PRNGKey(0)
    length = 64
    traj_keys= jr.split(key, num_traj)

    data, derivative_data = createData(num_traj, noise_level, key, d_l,
                                       min_samples=length, max_samples=length)
    # Split the different trajectories in the data into separate datasets
    data, num_datasets = splitDataset(data)

    print(f"Data Input shape: {data.inputs.shape}")
    print(f"Data Output shape: {data.outputs.shape}")

    input_dim = data.inputs.shape[-1]
    output_dim = data.outputs.shape[-1]
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    model = SmootherNet(input_dim=input_dim,
                        output_dim=output_dim,
                        output_stds=data_std,
                        logging_wandb=False,
                        beta=jnp.array([1.0, 1.0, 1.0]),
                        num_particles=8,
                        features=[128, 128, 64],
                        bnn_type=DeterministicEnsemble,
                        train_share=0.6,
                        num_training_steps=1000,
                        weight_decay=1e-4,
                        )

    ders = model.calcDerivative(key, data)

    print(f"Derivative mean shape: {ders.mean.shape}")
    print(f"Derivative epistemic_std shape: {ders.epistemic_std.shape}")
    print(f"Derivative aleatoric_std shape: {ders.aleatoric_std.shape}")
    print(f"Derivative beta shape: {ders.statistical_model_state.beta.shape}")

    # Plot the results for the first three trajectories
    fig, axes = plt.subplots(3, min(3, num_traj), figsize=(16, 9))
    for i in range(min(3, num_traj)):
        for j in range(3):
            axes[j][i].plot(data.inputs[i,:], data.outputs[i,:,j], label="x")
            axes[j][i].plot(data.inputs[i,:], ders.mean[i,:,j], label="\dot{x}")
            axes[j][i].fill_between(data.inputs[i,:].reshape(-1),
                                    (ders.mean[i,:,j] - ders.statistical_model_state.beta[i,j] * ders.epistemic_std[i,:,j]).reshape(-1),
                                    (ders.mean[i,:,j] + ders.statistical_model_state.beta[i,j] * ders.epistemic_std[i,:,j]).reshape(-1),
                                    label=r'$2\sigma$', alpha=0.3, color='blue')
            axes[j][i].plot(data.inputs[i,:], f_dot(data.inputs[i,:])[:,j], label="\dot{x}_{TRUE}")
            axes[j][i].set_title(f"Trajectory {i} - x{j}")
            axes[j][i].grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('traj_bnn.pdf')
