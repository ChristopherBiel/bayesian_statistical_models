import jax.numpy as jnp
from jax import grad, vmap

def f(x, y):
    # y is useless here
    return jnp.array([[x[0] * 2 + x[4]], [x[1] * 3], [x[2] * 4]])

# Define a function to compute gradients of each output element w.r.t. input
def compute_gradients(f, x):
    # Use vmap to automatically vectorize gradient computation over outputs
    grad_fn = lambda xi: grad(lambda x: f(x, 4)[xi, 0])(x)
    gradients = vmap(grad_fn)(jnp.arange(f(x, 4).shape[0]))
    return gradients.T

# Example usage
x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Example input
gradients = compute_gradients(f, x)
print(gradients.shape)
