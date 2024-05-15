import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import argparse

from bsm.utils.normalization import Data
from differentiators.nn_smoother.smoother_net import SmootherNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from data_handling.data_creation import create_example_data, example_function_derivative
from data_handling.data_creation import sample_pendulum_with_input, sample_random_pendulum_data
from data_handling.data_handling import split_dataset
from data_handling.data_output import plot_derivative_data, plot_data

def experiment(project_name: str = 'LearnDynamicsModel',
               seed: int=0,
               num_traj: int = 12,
               sample_points: int = 64,
               noise_level: float = None,
               smoother_features: list = [64, 64],
               dyn_features: list = [128, 128],
               smoother_particles: int = 10,
               dyn_particles: int = 10,
               smoother_training_steps: int = 1000,
               dyn_training_steps: int = 1000,
               smoother_weight_decay: float = 1e-4,
               dyn_weight_decay: float = 1e-4,
               smoother_type: str = 'DeterministicEnsemble',
               dyn_type: str = 'DeterministicEnsemble',
               logging_mode_wandb: int = 0,):
    
    # Input checks
    assert num_traj > 0
    assert sample_points > 0
    assert smoother_type in ['DeterministicEnsemble', 'ProbabilisticEnsemble', 'DeterministicFSVGDEnsemble',
                             'ProbabilisticFSVGDEnsemble'], f"Unknown smoother BNN type: {smoother_type}"
    assert dyn_type in ['DeterministicEnsemble', 'ProbabilisticEnsemble', 'DeterministicFSVGDEnsemble',
                        'ProbabilisticFSVGDEnsemble'], f"Unknown dyanmics BNN type: {dyn_type}"
    
    config = dict(seed=seed,
                  num_traj=num_traj,
                  sample_points=sample_points,
                  noise_level=noise_level,
                  smoother_features=smoother_features,
                  dyn_features=dyn_features,
                  smoother_particles=smoother_particles,
                  dyn_particles=dyn_particles,
                  smoother_training_steps=smoother_training_steps,
                  dyn_training_steps=dyn_training_steps,
                  smoother_weight_decay=smoother_weight_decay,
                  dyn_weight_decay=dyn_weight_decay,
                  smoother_type=smoother_type,
                  dyn_type=dyn_type,
                  logging_mode_wandb=logging_mode_wandb,)
    
    if logging_mode_wandb > 0:
        import wandb
        wandb.init(project=project_name,
                   config=config,)


    # Create the data
    key = jr.PRNGKey(seed=seed)
    t, x, u, x_dot = sample_random_pendulum_data(num_points=sample_points,
                                                 noise_level=noise_level,
                                                 key=key,
                                                 num_trajectories=num_traj,
                                                 initial_states=None,)
    if logging_mode_wandb > 0:
        fig = plot_data(t, x, u, x_dot, title='One trajectory of the sampled training data (pendulum)')
        wandb.log({'Training Data': wandb.Image(fig)})

    smoother_data = Data(inputs=t, outputs=x)

    input_dim = smoother_data.inputs.shape[-1]
    output_dim = smoother_data.outputs.shape[-1]
    if noise_level is None:
        data_std = jnp.ones(shape=(output_dim,))
    else:
        data_std = noise_level * jnp.ones(shape=(output_dim,))

    if logging_mode_wandb > 2:
        logging_smoother_wandb = True
        logging_dyn_wandb = True
    elif logging_mode_wandb == 2:
        logging_smoother_wandb = False
        logging_dyn_wandb = True
    else:
        logging_smoother_wandb = False
        logging_dyn_wandb = False
    

    # -------------------- Smoother --------------------
    if smoother_type == 'DeterministicEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=DeterministicEnsemble,
                            train_share=0.6,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            )
    elif smoother_type == 'ProbabilisticEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=ProbabilisticEnsemble,
                            train_share=0.6,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            )
    elif smoother_type == 'DeterministicFSVGDEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=DeterministicFSVGDEnsemble,
                            train_share=0.6,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            )
    elif smoother_type == 'ProbabilisticFSVGDEnsemble':
        model = SmootherNet(input_dim=input_dim,
                            output_dim=output_dim,
                            output_stds=data_std,
                            logging_wandb=logging_smoother_wandb,
                            beta=jnp.ones(shape=(output_dim,))*3,
                            num_particles=dyn_particles,
                            features=smoother_features,
                            bnn_type=ProbabilisticFSVGDEnsemble,
                            train_share=0.6,
                            num_training_steps=smoother_training_steps,
                            weight_decay=smoother_weight_decay,
                            )
    else:
        raise NotImplementedError(f"Unknown BNN type: {smoother_type}")

    # Learn the smoother
    model_states = model.learnSmoothers(key, smoother_data)
    pred_x = model.smoother_predict(smoother_data.inputs, model_states)
    ders = model.calcDerivative(model_states, smoother_data)

    print(f"Prediction shape: {pred_x.mean.shape}")
    print(f"Derivatives shape: {ders.mean.shape}")

    # Plot the results for the first three trajectories
    if logging_mode_wandb > 0:
        fig, axes = plt.subplots(output_dim, min(3, num_traj), figsize=(16, 9))
        for i in range(min(3, num_traj)):
            for j in range(output_dim):
                axes[j][i].plot(smoother_data.inputs[i,:], smoother_data.outputs[i,:,j], label=r'x')
                axes[j][i].plot(smoother_data.inputs[i,:], pred_x.mean[i,:,j], label=r'$x_{SMOOTHER}$')
                axes[j][i].plot(smoother_data.inputs[i,:], ders.mean[i,:,j], label=r'$\dot{x}_{SMOOTHER}$')
                axes[j][i].fill_between(smoother_data.inputs[i,:].reshape(-1),
                                        (ders.mean[i,:,j] - ders.statistical_model_state.beta[i,j] * ders.epistemic_std[i,:,j]).reshape(-1),
                                        (ders.mean[i,:,j] + ders.statistical_model_state.beta[i,j] * ders.epistemic_std[i,:,j]).reshape(-1),
                                        label=r'$2\sigma$', alpha=0.3, color='blue')
                axes[j][i].plot(smoother_data.inputs[i,:], x_dot[i,:,j], label=r'$\dot{x}_{TRUE}$')
                axes[j][i].set_title(f"Trajectory {i} - x{j}")
                axes[j][i].grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        wandb.log({'smoother': wandb.Image(plt)})

    # -------------------- Dynamics Model --------------------
    # The split data is concatinated again
    inputs = pred_x.mean.reshape(-1, output_dim)
    outputs = ders.mean.reshape(-1, output_dim)
    dyn_data = Data(inputs=inputs, outputs=outputs)
    if dyn_type == 'DeterministicEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=output_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=DeterministicEnsemble,
                                        train_share=0.6,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    elif dyn_type == 'ProbabilisticEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=output_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=ProbabilisticEnsemble,
                                        train_share=0.6,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    elif dyn_type == 'DeterministicFSVGDEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=output_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=DeterministicFSVGDEnsemble,
                                        train_share=0.6,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    elif dyn_type == 'ProbabilisticFSVGDEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=output_dim,
                                        output_dim=output_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(output_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=ProbabilisticFSVGDEnsemble,
                                        train_share=0.6,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    
    dyn_model_state = dyn_model.init(key)
    dyn_model_state = dyn_model.update(dyn_model_state, dyn_data)
    dyn_preds = dyn_model.predict_batch(dyn_data.inputs, dyn_model_state)

    # Plot the results for the first trajectory only
    if logging_mode_wandb > 0:
        fig = plot_derivative_data(t=t.reshape(-1, 1),
                                   x=smoother_data.outputs.reshape(-1, output_dim),
                                   x_est=pred_x.mean.reshape(-1, output_dim),
                                   x_dot_true=x_dot.reshape(-1, output_dim),
                                   x_dot_est = dyn_preds.mean,
                                   x_dot_est_std=dyn_preds.epistemic_std,
                                   beta=dyn_preds.statistical_model_state.beta,
                                   source='DYN. MODEL',
                                   num_trajectories_to_plot=2,
                                   )
        wandb.log({'dynamics': wandb.Image(fig)})


def main(args):
    experiment(project_name=args.project_name,
               seed=args.seed,
               num_traj=args.num_traj,
               sample_points=args.sample_points,
               noise_level=args.noise_level,
               smoother_features=args.smoother_features,
               dyn_features=args.dyn_features,
               smoother_particles=args.smoother_particles,
               dyn_particles=args.dyn_particles,
               smoother_training_steps=args.smoother_training_steps,
               dyn_training_steps=args.dyn_training_steps,
               smoother_weight_decay=args.smoother_weight_decay,
               dyn_weight_decay=args.dyn_weight_decay,
               smoother_type=args.smoother_type,
               dyn_type=args.dyn_type,
               logging_mode_wandb=args.logging_mode_wandb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='LearnDynamicsModel')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_traj', type=int, default=12)
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--sample_points', type=int, default=48)
    parser.add_argument('--smoother_features', type=list, default=[64, 64])
    parser.add_argument('--dyn_features', type=list, default=[128, 128])
    parser.add_argument('--smoother_particles', type=int, default=10)
    parser.add_argument('--dyn_particles', type=int, default=10)
    parser.add_argument('--smoother_training_steps', type=int, default=1000)
    parser.add_argument('--dyn_training_steps', type=int, default=4000)
    parser.add_argument('--smoother_weight_decay', type=float, default=3e-4)
    parser.add_argument('--dyn_weight_decay', type=float, default=3e-4)
    parser.add_argument('--smoother_type', type=str, default='ProbabilisticFSVGDEnsemble')
    parser.add_argument('--dyn_type', type=str, default='ProbabilisticFSVGDEnsemble')
    parser.add_argument('--logging_mode_wandb', type=int, default=1)
    args = parser.parse_args()
    main(args)