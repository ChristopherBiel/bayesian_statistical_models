import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from bsm.utils.normalization import Data
from SmootherNet import SmootherNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from data.data_creation import create_example_data, example_function_derivative
from data.data_creation import sample_pendulum_with_input, sample_random_pendulum_data
from data.data_handling import split_dataset

if __name__ == '__main__':
    
    num_traj = 12
    noise_level = 0.1
    d_l, d_u = 0, 10
    key = jr.PRNGKey(0)
    length = 64
    traj_keys= jr.split(key, num_traj)

    data, derivative_data = create_example_data(num_traj, noise_level, key, d_l,
                                       min_samples=length, max_samples=length)
    # Split the different trajectories in the data into separate datasets
    data, num_datasets = split_dataset(data)

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
                        num_particles=10,
                        features=[32, 32, 16],
                        bnn_type=DeterministicEnsemble,
                        train_share=0.6,
                        num_training_steps=1000,
                        weight_decay=1e-4,
                        )

    model_states = model.learnSmoothers(key, data)
    pred_x = model.smoother_predict(data.inputs, model_states)
    ders = model.calcDerivative(model_states, data)

    # print(f"Derivative mean shape: {ders.mean.shape}")
    # print(f"Derivative epistemic_std shape: {ders.epistemic_std.shape}")
    # print(f"Derivative aleatoric_std shape: {ders.aleatoric_std.shape}")
    # print(f"Derivative beta shape: {ders.statistical_model_state.beta.shape}")

    # Plot the results for the first three trajectories
    fig, axes = plt.subplots(3, min(3, num_traj), figsize=(16, 9))
    for i in range(min(3, num_traj)):
        for j in range(3):
            axes[j][i].plot(data.inputs[i,:], data.outputs[i,:,j], label=r'x')
            axes[j][i].plot(data.inputs[i,:], pred_x.mean[i,:,j], label=r'$x_{SMOOTHER}$')
            axes[j][i].plot(data.inputs[i,:], ders.mean[i,:,j], label=r'$\dot{x}_{SMOOTHER}$')
            axes[j][i].fill_between(data.inputs[i,:].reshape(-1),
                                    (ders.mean[i,:,j] - ders.statistical_model_state.beta[i,j] * ders.epistemic_std[i,:,j]).reshape(-1),
                                    (ders.mean[i,:,j] + ders.statistical_model_state.beta[i,j] * ders.epistemic_std[i,:,j]).reshape(-1),
                                    label=r'$2\sigma$', alpha=0.3, color='blue')
            axes[j][i].plot(data.inputs[i,:], example_function_derivative(data.inputs[i,:])[:,j], label=r'$\dot{x}_{TRUE}$')
            axes[j][i].set_title(f"Trajectory {i} - x{j}")
            axes[j][i].grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('out/traj_bnn.pdf')

    # Now use the Data to train the dynamics model
    # We use the derivative and flatten the different trajectories
    time = data.inputs.reshape(-1, 1)
    inputs = pred_x.mean.reshape(-1, output_dim)
    outputs = ders.mean.reshape(-1, output_dim)
    dyn_data = Data(inputs=inputs, outputs=outputs)

    dyn_model = BNNStatisticalModel(input_dim=output_dim,
                                    output_dim=output_dim,
                                    output_stds=data_std,
                                    logging_wandb=False,
                                    beta=jnp.array([1.0, 1.0, 1.0]),
                                    num_particles=10,
                                    features=[128, 128, 64],
                                    bnn_type=DeterministicEnsemble,
                                    train_share=0.6,
                                    num_training_steps=4000,
                                    weight_decay=1e-4,
                                    )
    
    dyn_model_state = dyn_model.init(key)
    dyn_model_state = dyn_model.update(dyn_model_state, dyn_data)
    dyn_preds = dyn_model.predict_batch(dyn_data.inputs, dyn_model_state)

    # Plot the results for the first trajectory only
    interval = 64*1
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    for i in range(min(3, num_traj)):
            axes[i].plot(time[:interval].reshape(-1), dyn_preds.mean[:interval,i], label=r'$\dot{x}_{DYN. MODEL}$')
            axes[i].fill_between(time[:interval].reshape(-1),
                                    (dyn_preds.mean[:interval,i] - dyn_preds.statistical_model_state.beta[i] * dyn_preds.epistemic_std[:interval,i]).reshape(-1),
                                    (dyn_preds.mean[:interval,i] + dyn_preds.statistical_model_state.beta[i] * dyn_preds.epistemic_std[:interval,i]).reshape(-1),
                                    label=r'$2\sigma$', alpha=0.3, color='blue')
            axes[i].plot(time[:interval].reshape(-1), dyn_data.outputs[:interval,i], label=r'$\dot{x}_{SMOOTHER}$')
            axes[i].plot(time[:interval].reshape(-1), example_function_derivative(time)[:interval,i], label=r'$\dot{x}_{TRUE}$')
            axes[i].set_title(f"x{i}")
            axes[i].grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('out/dyn_bnn.pdf')