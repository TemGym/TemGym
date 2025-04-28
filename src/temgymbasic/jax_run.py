
from .jax_ray import propagate
import jax
import jax.numpy as jnp

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
    return ray

def run_model_for_jacobians(ray, model):

    model_ray_jacobians = []

    # Get all jacobians from one component to another
    for i in range(1, len(model)):
        distance = (model[i-1].z - ray.z).squeeze()
        ray = propagate(distance, ray)
        ray = model[i-1].step(ray)
        
        ray_jacobian = jax.jacobian(run_to_component)(ray, model[i])
        model_ray_jacobians.append(ray_jacobian)

    # Edit the jacobian matrices to include shifts calculated 
    # from the opl derivative
    ABCDs = [] #ABCD matrices at each component

    for ray_jacobian in model_ray_jacobians:
        shift_vector = ray_jacobian.pathlength.matrix # This is the shift vector for each ray, dopl_out/dr_in
        ABCD = ray_jacobian.matrix.matrix # This is the ABCD matrix for each ray, dr_out/dr_in
        ABCD = ABCD.at[:, -1].set(shift_vector[0, :])
        ABCD = ABCD.at[-1, -1].set(1.0)
        ABCDs.append(ABCD)

    return jnp.array(ABCDs)