# Train a model on pure dynamics data, to test it's capability to generalize

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import argparse

from bsm.utils.normalization import Data
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from data_functions.data_creation import create_example_data, example_function_derivative
from data_functions.data_creation import sample_pendulum_with_input, sample_random_pendulum_data
from data_functions.data_handling import split_dataset
from data_functions.data_output import plot_derivative_data, plot_data


def experiment(project_name: str = 'LearnDynamicsModel',
               seed: int=0,
               num_traj: int = 12,
               sample_points: int = 64,
               noise_level: float = None,
               num_traj_train: int = 12,
               dyn_feature_size: int = 128,
               dyn_num_hidden_layers: int = 2,
               dyn_particles: int = 10,
               dyn_training_steps: int = 1000,
               dyn_weight_decay: float = 1e-4,
               dyn_train_share: float = 1,
               dyn_type: str = 'DeterministicEnsemble',
               logging_mode_wandb: int = 0,
               ):
    
    # Input checks
    assert num_traj > 0
    assert sample_points > 0
    assert dyn_type in ['DeterministicEnsemble', 'ProbabilisticEnsemble', 'DeterministicFSVGDEnsemble',
                        'ProbabilisticFSVGDEnsemble'], f"Unknown dyanmics BNN type: {dyn_type}"
    
    config = dict(seed=seed,
                  num_traj=num_traj,
                  sample_points=sample_points,
                  noise_level=noise_level,
                  num_traj_train=num_traj_train,
                  dyn_feature_size=dyn_feature_size,
                  dyn_num_hidden_layers=dyn_num_hidden_layers,
                  dyn_particles=dyn_particles,
                  dyn_training_steps=dyn_training_steps,
                  dyn_weight_decay=dyn_weight_decay,
                  dyn_train_share=dyn_train_share,
                  dyn_type=dyn_type,
                  logging_mode_wandb=logging_mode_wandb,
                )
    
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

    state_dim = x.shape[-1]
    control_dim = u.shape[-1]

    if noise_level is None:
        data_std = jnp.ones(shape=(state_dim,)) * 0.001
    else:
        data_std = noise_level * jnp.ones(shape=(state_dim,))

    if logging_mode_wandb > 1:
        logging_dyn_wandb = True
    else:
        logging_dyn_wandb = False

    # -------------------- Dynamics Model --------------------

    x_train = x[:num_traj_train,:,:].reshape(-1,state_dim)
    u_train = u[:num_traj_train,:,:].reshape(-1,control_dim)
    x_dot_train = x_dot[:num_traj_train,:,:].reshape(-1,state_dim)
    dyn_data = Data(inputs=jnp.concatenate([x_train, u_train], axis=-1), outputs=x_dot_train)

    print(f"Using new Data with input shape {dyn_data.inputs.shape} and output shape {dyn_data.outputs.shape}")
    # Change the features to the desired size
    dyn_features = [dyn_feature_size] * dyn_num_hidden_layers
    if dyn_type == 'DeterministicEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=DeterministicEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    elif dyn_type == 'ProbabilisticEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=ProbabilisticEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    elif dyn_type == 'DeterministicFSVGDEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=DeterministicFSVGDEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    elif dyn_type == 'ProbabilisticFSVGDEnsemble':
        dyn_model = BNNStatisticalModel(input_dim=state_dim+control_dim,
                                        output_dim=state_dim,
                                        output_stds=data_std,
                                        logging_wandb=logging_dyn_wandb,
                                        beta=jnp.ones(shape=(state_dim,))*2,
                                        num_particles=dyn_particles,
                                        features=dyn_features,
                                        bnn_type=ProbabilisticFSVGDEnsemble,
                                        train_share=dyn_train_share,
                                        num_training_steps=dyn_training_steps,
                                        weight_decay=dyn_weight_decay
                                        )
    
    dyn_model_state = dyn_model.init(key)
    dyn_model_state = dyn_model.update(dyn_model_state, dyn_data)
    dyn_preds = dyn_model.predict_batch(dyn_data.inputs, dyn_model_state)

    # Plot the results for the first trajectory only
    if logging_mode_wandb > 0:
        fig = plot_derivative_data(t=t[0:num_traj_train,:,:].reshape(-1,1),
                                   x=x_train,
                                   x_dot_true=x_dot_train,
                                   x_dot_est=dyn_preds.mean,
                                   x_dot_est_std=dyn_preds.epistemic_std,
                                   beta=dyn_preds.statistical_model_state.beta,
                                   source='DYN. MODEL',
                                   num_trajectories_to_plot=1,
                                   )
        wandb.log({'dynamics': wandb.Image(fig)})


def main(args):
    experiment(project_name=args.project_name,
               seed=args.seed,
               num_traj=args.num_traj,
               sample_points=args.sample_points,
               num_traj_train = args.num_traj_train,
               noise_level=args.noise_level,
               dyn_feature_size=args.dyn_feature_size,
               dyn_num_hidden_layers=args.dyn_num_hidden_layers,
               dyn_particles=args.dyn_particles,
               dyn_training_steps=args.dyn_training_steps,
               dyn_weight_decay=args.dyn_weight_decay,
               dyn_train_share=args.dyn_train_share,
               dyn_type=args.dyn_type,
               logging_mode_wandb=args.logging_mode_wandb,
               )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='LearnDynamicsModel')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_traj', type=int, default=3)
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--sample_points', type=int, default=64)
    parser.add_argument('--num_traj_train', type=int, default=3)
    parser.add_argument('--dyn_feature_size', type=int, default=32)
    parser.add_argument('--dyn_num_hidden_layers', type=int, default=2)
    parser.add_argument('--dyn_particles', type=int, default=10)
    parser.add_argument('--dyn_training_steps', type=int, default=48000)
    parser.add_argument('--dyn_weight_decay', type=float, default=2e-4)
    parser.add_argument('--dyn_train_share', type=float, default=0.8)
    parser.add_argument('--dyn_type', type=str, default='DeterministicFSVGDEnsemble')
    parser.add_argument('--logging_mode_wandb', type=int, default=2)
    args = parser.parse_args()
    main(args)