import matplotlib.pyplot as plt
import jax.numpy as jnp
import chex
from bsm.utils.normalization import Data
from data_handling.data_handling import split_dataset

def plot_derivative_data(t: chex.Array,
                         x: chex.Array,
                         x_est: chex.Array,
                         x_dot_true: chex.Array,
                         x_dot_est: chex.Array,
                         x_dot_est_std: chex.Array,
                         beta: chex.Array,
                         source: str,
                         num_trajectories_to_plot: int = 1,
                         ) -> plt.figure:
    state_dim = x.shape[-1]
    assert x.shape == x_est.shape == x_dot_true.shape == x_dot_est.shape == x_dot_est_std.shape
    assert num_trajectories_to_plot > 0, "The number of trajectories too plot must be more than zero"
    
    # Data has to be split again to be able to plot individual trajectories easier
    data = Data(inputs=t, outputs=jnp.concatenate([x, x_est, x_dot_true, x_dot_est, x_dot_est_std], axis=-1))
    data, num_trajectories = split_dataset(data)

    t = data.inputs
    x = data.outputs[:,:,:state_dim]
    x_est = data.outputs[:,:,state_dim:state_dim*2]
    x_dot_true = data.outputs[:,:,state_dim*2:state_dim*3]
    x_dot_est = data.outputs[:,:,state_dim*3:state_dim*4]
    x_dot_est_std = data.outputs[:,:,state_dim*4:]

    fig, axes = plt.subplots(state_dim, num_trajectories_to_plot, figsize=(16,9))
    for k01 in range(state_dim):
        if num_trajectories_to_plot > 1:
            for k02 in range(num_trajectories_to_plot):
                axes[k01][k02].plot(t[k02,:,0].reshape(-1), x_dot_est[k02,:,k01], label=r'$\dot{x}_{%s}$'%(source))
                axes[k01][k02].fill_between(t[0,:,0].reshape(-1),
                                            (x_dot_est[k02,:,k01] - beta[k01] * x_dot_est_std[k02,:,k01]).reshape(-1),
                                            (x_dot_est[k02,:,k01] + beta[k01] * x_dot_est_std[k02,:,k01]).reshape(-1),
                                            label=r'$2\sigma$', alpha=0.3, color='blue')
                axes[k01][k02].plot(t[0,:,0].reshape(-1), x_dot_true[k02,:,k01], label=r'$\dot{x}_{TRUE}$')
                axes[k01][k02].set_ylabel(r'state $x_{%s}$' %(str(k01)))
                axes[k01][k02].set_xlabel(r'Time [s]')
                axes[k01][k02].set_title(r'Trajectory %s'%(str(k02)))
                axes[k01][k02].grid(True, which='both')
        else:
            axes[k01].plot(t[0,:,0].reshape(-1), x_dot_est[0,:,k01], label=r'$\dot{x}_{%s}$'%(source))
            axes[k01].fill_between(t[0,:,0].reshape(-1),
                                    (x_dot_est[0,:,k01] - beta[k01] * x_dot_est_std[0,:,k01]).reshape(-1),
                                    (x_dot_est[0,:,k01] + beta[k01] * x_dot_est_std[0,:,k01]).reshape(-1),
                                    label=r'$2\sigma$', alpha=0.3, color='blue')
            axes[k01].plot(t[0,:,0].reshape(-1), x_dot_true[0,:,k01], label=r'$\dot{x}_{TRUE}$')
            axes[k01].set_ylabel(r'state $x_{%s}$' %(str(k01)))
            axes[k01].set_xlabel(r'Time [s]')
            axes[k01].grid(True, which='both')
    plt.legend()
    fig.tight_layout()
    return fig

def calc_derivative_RMSE(x_dot_true: chex.Array,
                         x_dot_est: chex.Array,
                         x_dot_est_std: chex.Array) -> float:
    pass