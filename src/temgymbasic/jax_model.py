
from .jax_ray import propagate
import jax
import jax.numpy as jnp


def run_model_to_end(ray, model):
    for component in model:

        distance = component.z - ray.z
        ray = propagate(distance, ray)
        ray = component.step(ray)

    return ray


def run_model_iter(ray, model):
    rays = []
    for component in model:
        distance = component.z - ray.z
        new_ray = propagate(distance, ray)
        new_ray = component.step(new_ray)
        ray = new_ray
        rays.append(new_ray)
    return rays
