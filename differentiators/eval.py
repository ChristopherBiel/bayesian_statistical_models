import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import chex

from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState
from bsm.utils.type_aliases import StatisticalModelState
from bsm.utils.normalization import Data
from data_functions.data_creation import create_random_control_sequence, sample_pendulum_with_input
from data_functions.data_output import plot_derivative_data, plot_prediction_data


def evaluate_dyn_model(dyn_model: BNNStatisticalModel,
                       dyn_model_state: StatisticalModelState[BNNState],
                       num_points: int = 64,
                       seed: int = 0,
                       plot_data: bool = False,
                       return_performance: bool = False,
                       colored_noise_exponent: float = 3,
                       initial_state: chex.Array = jnp.array([-1.0, 0.0, 0.0]),
                       plot_annotation_source: str = "",
                       ):
    
    # Create a random control sequence to test the model
    key = jr.PRNGKey(seed)
    state_dim = dyn_model.output_dim
    control_dim = dyn_model.input_dim - state_dim
    control_input = create_random_control_sequence(num_points, key, control_dim, colored_noise_exponent)
    assert control_input.shape == (num_points, control_dim)

    # Evaluate the model
    current_state = initial_state
    current_state_std = jnp.ones((state_dim)) * 0.001
    t, x_true, x_dot_true = sample_pendulum_with_input(control_input, initial_state)

    x_est = jnp.zeros((num_points, state_dim))
    x_est_std = jnp.zeros((num_points, state_dim))
    x_dot_est = jnp.zeros((num_points, state_dim))
    x_dot_est_std = jnp.zeros((num_points, state_dim))

    for k01 in range(num_points):
        x_est.at[k01, :].set(current_state)
        x_est_std.at[k01, :].set(current_state_std)
        model_inputs = jnp.concatenate([current_state, control_input[k01,:]]).reshape(1,dyn_model.input_dim)
        dyn_prediction = dyn_model.predict_batch(model_inputs, dyn_model_state)
        x_dot_est.at[k01, :].set(dyn_prediction.mean.reshape(state_dim))
        x_dot_est_std.at[k01, :].set(dyn_prediction.epistemic_std.reshape(state_dim))
        if k01 < (num_points - 1):
            current_state += (t[k01+1] - t[k01]) * dyn_prediction.mean.reshape(state_dim)
            current_state_std += (t[k01+1] - t[k01]) * dyn_prediction.epistemic_std.reshape(state_dim)

    print(f"Estimated x: {x_est}")

    if plot_data:
        import matplotlib.pyplot as plt
        plot_derivative_data(t, x_true, x_dot_true, x_dot_est, x_dot_est_std,
                             beta = dyn_prediction.statistical_model_state.beta)
        plot_prediction_data(t, x_true, x_est, x_est_std, source=plot_annotation_source,
                             beta = dyn_prediction.statistical_model_state.beta)
        plt.show()

    if return_performance:
        def mse(x, x_pred):
            return jnp.power((x-x_pred),2).mean()
        state_pred_mse = vmap(mse, in_axes=(1, 1))(x_true, x_est)
        return state_pred_mse
    
if __name__ == "__main__":
    from differentiators.nn_smoother.exp import experiment
    # create a small model just to test everything
    dyn_model, dyn_model_state = experiment(sample_points=32,
                                            num_traj=3,
                                            smoother_particles=5,
                                            smoother_training_steps=1000,
                                            dyn_feature_size=32,
                                            dyn_training_steps=2000,
                                            logging_mode_wandb=0,
                                            return_model_state=True)
    
    state_pred_mse = evaluate_dyn_model(dyn_model = dyn_model,
                                        dyn_model_state = dyn_model_state,
                                        num_points=20,
                                        plot_data = True,
                                        return_performance=True,
                                        plot_annotation_source="DYN,SMOOTHER")
    
    print(f"Prediction error for the model per state is {state_pred_mse}")