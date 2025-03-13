
from .jax_ray import propagate

def run_to_end(ray, components):
    for component in components:
        distance = (component.z - ray.z).squeeze()
        ray = propagate(distance, ray)
        ray = component.step(ray)

    return ray

def run_to_component(ray, component):
    distance = (component.z - ray.z).squeeze()
    ray = propagate(distance, ray)
    ray = component.step(ray)
<<<<<<< HEAD
    return ray

=======
    return ray
>>>>>>> b89c34c8f431d138ede8882a2de4f640625c8708
