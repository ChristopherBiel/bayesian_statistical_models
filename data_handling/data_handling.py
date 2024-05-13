import jax.numpy as jnp
from bsm.utils.normalization import Data

def split_dataset(data: Data) -> (list[Data], int):
        """Splits the full Dataset into the individual trajectories,
        based on the timestamps (every time there is a jump backwards in the timestamp the data is cut)
        The output is still only one dataset, but now with an additional dimension, which is the number of trajectories."""

        t = data.inputs
        assert t.shape[1] == 1

        # Split based on the distance between the timesteps (split at negative values)
        delta_t = jnp.diff(t, axis=0)
        indices = jnp.where(delta_t < 0.0)[0] + 1
        ts = jnp.split(t, indices)
        xs = jnp.split(data.outputs, indices)

        # To be able to stack, all the arrays need to have the same shape
        # For trajectories with different lengths (which may happen) this is not the case
        # Therefore, we need to pad by wrapping around (so as to not favor certain samples or add new ones)
        max_length = max([len(traj) for traj in ts])
        for i in range(len(ts)):
            ts[i] = jnp.pad(ts[i], ((0, max_length - len(ts[i])), (0, 0)), mode='wrap')
            xs[i] = jnp.pad(xs[i], ((0, max_length - len(xs[i])), (0, 0)), mode='wrap')
        
        inputs = jnp.stack(ts, axis=0)
        outputs = jnp.stack(xs, axis=0)

        return Data(inputs=inputs, outputs=outputs), len(ts)