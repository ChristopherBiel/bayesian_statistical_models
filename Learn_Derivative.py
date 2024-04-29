import jax.numpy as jnp
import jax.random as jr

from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from bsm.utils.normalization import Data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create the data
    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 1

    noise_level = 0.1
    d_l, d_u = 0, 10
    t = jnp.linspace(d_l, d_u, 64).reshape(-1, 1)
    x = jnp.sin(t) * jnp.cos(0.2*t)
    x_dot = jnp.sin(t) * (-0.2) * jnp.sin(0.2*t) + jnp.cos(t) * jnp.cos(0.2*t)
    x = x + noise_level * jr.normal(key=jr.PRNGKey(0), shape=x.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    data = Data(inputs=t, outputs=x)

    model = BNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                                beta=jnp.array([1.0]), num_particles=10, features=[32, 16],
                                bnn_type=DeterministicEnsemble, train_share=0.6, num_training_steps=2000,
                                weight_decay=1e-4, )

    init_model_state = model.init(key=jr.PRNGKey(0))
    statistical_model_state = model.update(stats_model_state=init_model_state, data=data)

    # Test on new data
    test_t = jnp.linspace(d_l-5, d_u+5, 1000).reshape(-1, 1)
    test_x = jnp.sin(test_t) * jnp.cos(0.2*test_t)
    test_xdot = jnp.sin(test_t) * (-0.2) * jnp.sin(0.2*test_t) + jnp.cos(test_t) * jnp.cos(0.2*test_t)

    preds = model.predict_batch(test_t, statistical_model_state)
    derivative = model.derivative_batch(test_t, statistical_model_state)

    plt.scatter(test_t.reshape(-1), test_x, label='Data', color='red')
    plt.plot(test_t, preds.mean, label='Mean', color='blue')
    plt.fill_between(test_t.reshape(-1),
                     (preds.mean - preds.statistical_model_state.beta * preds.epistemic_std).reshape(-1),
                     (preds.mean + preds.statistical_model_state.beta * preds.epistemic_std).reshape(-1),
                     label=r'$2\sigma$', alpha=0.3, color='blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.plot(test_t.reshape(-1), test_x, label='True', color='green')
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    num_test_points = 1000
    in_domain_test_t = jnp.linspace(d_l, d_u, num_test_points).reshape(-1, 1)
    in_domain_test_x = jnp.sin(in_domain_test_t) * jnp.cos(0.2*in_domain_test_t)

    in_domain_preds = model.predict_batch(in_domain_test_t, statistical_model_state)
    for j in range(output_dim):
        plt.plot(in_domain_test_t, in_domain_preds.mean[:, j], label='Mean', color='blue')
        plt.plot(in_domain_test_t, in_domain_test_x[:, j], label='Fun', color='Green')
        plt.legend()
        plt.show()
