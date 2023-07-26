from collections import OrderedDict
from functools import partial

import chex
import distrax
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from jax import vmap, jit
from jax.scipy.stats import multivariate_normal
from jaxtyping import PyTree

from bsm.models.gaussian_processes.kernels import Kernel, RBF
from bsm.utils.normalization import Normalizer, DataStats, Data


@chex.dataclass
class GPModelState:
    history: Data
    data_stats: DataStats
    params: PyTree


class GaussianProcess:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 output_stds: chex.Array,
                 kernel: Kernel | None = None,
                 weight_decay: float = 0.0,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-2),
                 seed: int = 0
                 ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if kernel is None:
            kernel = RBF(input_dim)
        self.kernel = kernel
        assert output_stds.shape == (output_dim,)
        self.output_stds = output_stds
        self.normalizer = Normalizer()
        self.tx = optax.adamw(learning_rate=lr_rate, weight_decay=weight_decay)
        self.key = jr.PRNGKey(seed)

        self.v_kernel = vmap(self.kernel.apply, in_axes=(0, None, None), out_axes=0)
        self.m_kernel = vmap(self.v_kernel, in_axes=(None, 0, None), out_axes=1)
        self.m_kernel_multiple_output = vmap(self.m_kernel, in_axes=(None, None, 0), out_axes=0)
        self.v_kernel_multiple_output = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)
        self.kernel_multiple_output = vmap(self.kernel.apply, in_axes=(None, None, 0), out_axes=0)

    def init(self, key: chex.PRNGKey) -> PyTree:
        keys = jr.split(key, self.output_dim)
        return vmap(self.kernel.init)(keys)

    def loss(self, vmapped_params, inputs, outputs, data_stats: DataStats):
        assert inputs.shape[0] == outputs.shape[0]

        # Normalize inputs and outputs
        inputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(inputs, data_stats.inputs)
        outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(outputs, data_stats.outputs)
        outputs_stds_norm = self.normalizer.normalize_std(self.output_stds, data_stats.outputs)

        # Compute covariance matrix
        covariance_matrix = self.m_kernel_multiple_output(inputs_norm, inputs_norm, vmapped_params)

        # Add noise term
        extended_eye = jnp.repeat(jnp.eye(covariance_matrix.shape[-1])[None, ...], repeats=self.output_dim, axis=0)
        noise_term = extended_eye * outputs_stds_norm[:, None, None] ** 2
        noisy_covariance_matrix = covariance_matrix + noise_term

        # Compute log pdf
        log_pdf = vmap(multivariate_normal.logpdf, in_axes=(1, None, 0))(outputs_norm, jnp.zeros(inputs.shape[0]),
                                                                         noisy_covariance_matrix)
        return -jnp.mean(log_pdf) / inputs.shape[0]

    @partial(jit, static_argnums=0)
    def _step_jit(self,
                  opt_state: optax.OptState,
                  vmapped_params: chex.PRNGKey,
                  inputs: chex.Array,
                  outputs: chex.Array,
                  data_stats: DataStats) -> (optax.OptState, PyTree, OrderedDict):
        loss, grads = jax.value_and_grad(self.loss)(vmapped_params, inputs, outputs,
                                                    data_stats)
        updates, opt_state = self.tx.update(grads, opt_state, vmapped_params)
        vmapped_params = optax.apply_updates(vmapped_params, updates)
        statiscs = OrderedDict(nll=loss)
        return opt_state, vmapped_params, statiscs

    def fit_model(self,
                  inputs: chex.Array,
                  target_outputs: chex.Array,
                  num_epochs: int,
                  data_stats: DataStats) -> PyTree:
        self.key, key = jr.split(self.key)
        vmapped_params = self.init(key)
        opt_state = self.tx.init(vmapped_params)
        for step in range(1, num_epochs + 1):
            opt_state, vmapped_params, statistics = self._step_jit(opt_state, vmapped_params, inputs,
                                                                   target_outputs, data_stats)

            wandb.log(statistics)

            if step % 100 == 0 or step == 1:
                print(f"Step {step}: {statistics}")

        return vmapped_params

    @partial(jit, static_argnums=0)
    def posterior(self, input, gp_model: GPModelState) -> distrax.Normal:
        assert input.ndim == 1
        input_norm = self.normalizer.normalize(input, gp_model.data_stats.inputs)

        history_inputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(gp_model.history.inputs,
                                                                                 gp_model.data_stats.inputs)
        history_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(gp_model.history.outputs,
                                                                                  gp_model.data_stats.outputs)

        outputs_stds_norm = self.normalizer.normalize_std(self.output_stds, gp_model.data_stats.outputs)

        # Compute covariance matrix
        covariance_matrix = self.m_kernel_multiple_output(history_inputs_norm, history_inputs_norm, gp_model.params)

        # Add noise term
        extended_eye = jnp.repeat(jnp.eye(covariance_matrix.shape[-1])[None, ...], repeats=self.output_dim, axis=0)
        noise_term = extended_eye * outputs_stds_norm[:, None, None] ** 2
        noisy_covariance_matrix = covariance_matrix + noise_term

        k_x_X = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)(history_inputs_norm, input_norm,
                                                                         gp_model.params)
        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        # Compute posterior std
        denoised_var = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 0))((cholesky_tuples[0], False), k_x_X)
        var = vmap(self.kernel.apply, in_axes=(None, None, 0))(input_norm, input_norm, gp_model.params) - vmap(jnp.dot)(
            k_x_X, denoised_var)
        std = jnp.sqrt(var)

        # Compute posterior mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False),
                                                                                 history_outputs_norm)
        mean = vmap(jnp.dot)(k_x_X, denoised_mean)

        # Denormalize
        mean = self.normalizer.denormalize(mean, gp_model.data_stats.outputs)
        std = self.normalizer.denormalize_std(std, gp_model.data_stats.outputs)

        return distrax.Normal(loc=mean, scale=std)


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 64).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(3 * xs)], axis=1)
    ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    normalizer = Normalizer()
    data = DataStats(inputs=xs, outputs=ys)
    data_stats = normalizer.compute_stats(data)

    num_particles = 10
    model = GaussianProcess(input_dim=input_dim, output_dim=output_dim, output_stds=data_std)
    start_time = time.time()
    print('Starting with training')
    wandb.init(
        project='Pendulum',
        group='test group',
    )

    model_params = model.fit_model(inputs=xs, target_outputs=ys, num_epochs=1000, data_stats=data_stats)
    print(f'Training time: {time.time() - start_time:.2f} seconds')

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    gp_model = GPModelState(history=Data(inputs=xs, outputs=ys), params=model_params, data_stats=data_stats)
    preds = vmap(model.posterior, in_axes=(0, None))(test_xs, gp_model)
    pred_means = preds.mean()
    epistemic_stds = preds.scale
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(3 * test_xs)], axis=1)

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        plt.plot(test_xs, pred_means[:, j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (pred_means[:, j] - 2 * epistemic_stds[:, j]).reshape(-1),
                         (pred_means[:, j] + 2 * epistemic_stds[:, j]).reshape(-1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
