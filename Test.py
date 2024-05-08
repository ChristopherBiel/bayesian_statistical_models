import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from bsm.utils.normalization import Data
from Learn_Derivative import SmootherNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble

def createNoisyTrajectory(length, key, noise_level):
    t = jnp.linspace(0, length/10, length).reshape(-1,1)
    x = jnp.concatenate([jnp.sin(t) * jnp.cos(0.2*t),
                         jnp.cos(t) + 0.2*jnp.sin(3*t),
                         2*(t - length)/(length)*jnp.cos(t)], axis=1)
    x = x + noise_level * jr.normal(key, shape=x.shape)

    return Data(inputs=t, outputs=x)

num_traj = 12
noise_level = 0.1
key = jr.PRNGKey(0)
length = 64
traj_keys= jr.split(key, num_traj)

Multi_Data = vmap(createNoisyTrajectory, in_axes=(None, 0, None))(length, traj_keys, noise_level)

input_dim = Multi_Data.inputs.shape[-1]
output_dim = Multi_Data.outputs.shape[-1]
data_std = noise_level * jnp.ones(shape=(output_dim,))

model = SmootherNet(input_dim=input_dim,
                    output_dim=output_dim,
                    output_stds=data_std,
                    logging_wandb=False,
                    beta=jnp.array([1.0, 1.0, 1.0]),
                    num_particles=16,
                    features=[64, 64, 32],
                    bnn_type=DeterministicEnsemble,
                    train_share=0.6,
                    num_training_steps=2000,
                    weight_decay=1e-4,
                    )

model.addDerivativeToDataset(key, Multi_Data)