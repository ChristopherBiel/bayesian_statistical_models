import jax
import jax.numpy as jnp
import chex
# Tutorial for jax vmap

def f(x: chex.Array):
    return x.mean()

# Create a random 10x10 matrix and apply f to each row:
x = jnp.arange(100).reshape(10, 10)
f_mapped = jax.vmap(f, in_axes=1)(x)
print(f"Input: {x} Output: {f_mapped}")
print(jnp.mean(jnp.linspace(0,90,10)))