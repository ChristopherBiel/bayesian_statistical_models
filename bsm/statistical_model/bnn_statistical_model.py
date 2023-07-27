import chex
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap

from abstract_statistical_model import StatisticalModel
from bsm.models.gaussian_processes.kernels import Kernel
from bsm.models.bayesian_neural_networks.bnn import BNNState
from bsm.utils.normalization import Data, DataStats
from bsm.utils.type_aliases import StatisticalModelState, StatisticalModelOutput


class BNNStatisticalModel(StatisticalModel[BNNState]):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 output_stds: chex.Array,
                 kernel: Kernel | None = None,
                 weight_decay: float = 0.0,
                 lr_rate: optax.Schedule | float = optax.constant_schedule(1e-2),
                 seed: int = 0,
                 f_norm_bound: float = 1.0,
                 delta: float = 0.1,
                 num_training_steps: int = 1000,
                 logging_wandb: bool = True,
                 beta: chex.Array | optax.Schedule | None = None
                 ):
        model = GaussianProcess(input_dim=input_dim, output_dim=output_dim, output_stds=output_stds,
                                kernel=kernel, weight_decay=weight_decay, lr_rate=lr_rate, seed=seed,
                                logging_wandb=logging_wandb)
        super().__init__(input_dim, output_dim, model)
        self.model = model
        self.f_norm_bound = f_norm_bound
        self.delta = delta
        self.num_training_steps = num_training_steps
        if beta is None:
            beta = jnp.ones(shape=(output_dim,))
        if isinstance(beta, chex.Array):
            beta = optax.constant_schedule(beta)
        self._potential_beta = beta

    def init(self, key: chex.PRNGKey) -> BNNState:
        inputs = jnp.zeros(shape=(1, self.input_dim))
        outputs = jnp.zeros(shape=(1, self.output_dim))
        data = Data(inputs=inputs, outputs=outputs)
        data_stats = self.model.normalizer.compute_stats(data.inputs)
        params = self.model.init(key)
        return BNNState(params=params, data_stats=data_stats, history=data)

    def update(self, model_state: BNNState, data: Data) -> StatisticalModelState[BNNState]:
        new_model_state = self.model.fit_model(data, self.num_training_steps)
        beta = self._potential_beta(data.inputs.shape[0])
        assert beta.shape == (self.output_dim,)
        return StatisticalModelState(model_state=new_model_state, beta=beta)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 128).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(3 * xs)], axis=1)
    ys = ys + noise_level * jr.normal(key=jr.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    data = DataStats(inputs=xs, outputs=ys)

    model = BNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                                f_norm_bound=10, beta=jnp.array([1.0, 15.0]))
    init_model_state = model.init(key=jr.PRNGKey(0))
    statistical_model_state = model.update(model_state=init_model_state, data=data)

    # Test on new data
    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(3 * test_xs)], axis=1)

    preds = vmap(model.predict, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(test_xs, statistical_model_state)

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        plt.plot(test_xs, preds.mean[:, j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (preds.mean[:, j] - preds.statistical_model_state.beta[j] * preds.epistemic_std[:,
                                                                                     j]).reshape(-1),
                         (preds.mean[:, j] + preds.statistical_model_state.beta[j] * preds.epistemic_std[:, j]).reshape(
                             -1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()