
from .jax_ray import propagate
import jax
import jax.numpy as jnp
from jax import tree_util

def run_to_end(ray, components):
    for component in components:

        distance = (component.z - ray.z).squeeze()
        ray = propagate(distance, ray)
        ray = component.step(ray)

    return ray.matrix

def run_to_end_model_wrapper(model_flat, unravel_fn, rays):
    # Unravel to get the model pytree
    model = unravel_fn(model_flat)
    output = run_to_end(rays, model)
    # Assuming output is vector-valued, you might want to reduce it to a scalar or
    # use jacobian directly.
    return output