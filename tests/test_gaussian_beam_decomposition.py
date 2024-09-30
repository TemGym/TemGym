import pytest
import numpy as np
import matplotlib.pyplot as plt

from temgymbasic.model import (
    Model,
)
from temgymbasic.gbd import (
                            propagate_misaligned_gaussian, 
                            gaussian_amplitude,
                            guoy_phase
)
import temgymbasic.components as comp
from temgymbasic.rays import Rays
from temgymbasic.utils import calculate_phi_0
from temgymbasic.plotting import plot_model
from diffractio.scalar_sources_XY import Scalar_source_XY
from scipy.constants import e, m_e, c

try:
    import cupy as cp
except ImportError:
    cp = None

import os

# Check environment variable to decide backend
USE_GPU = os.getenv('USE_GPU', '0') == '1'

if USE_GPU:
    xp = cp
else:
    xp = np

ETA = (abs(e)/(2*m_e))**(1/2)
EPSILON = abs(e)/(2*m_e*c**2)

def zero_phase(u, idx_x, idx_y):
    u_centre = u[idx_x, idx_y]
    phase_difference =  0 - np.angle(u_centre)
    u = u * np.exp(1j * phase_difference)
    
    return u

def FresnelPropagator(E0, ps, lambda0, z):
    """
    Parameters:
        E0 : 2D array
            The initial complex field in the x-y source plane.
        ps : float
            Pixel size in the object plane (same units as wavelength).
        lambda0 : float
            Wavelength of the light (in the same units as ps).
        z : float
            Propagation distance (in the same units as ps).

    Returns:
        Ef : 2D array
            The complex field after propagating a distance z.
    """
    n, m = E0.shape

    fx = np.fft.fftfreq(n, ps)
    fy = np.fft.fftfreq(m, ps)
    Fx, Fy = np.meshgrid(fx, fy)
    
    H = np.exp(-1j * (2 * np.pi / lambda0) * z) * np.exp(-1j * np.pi * lambda0 * z * (Fx**2 + Fy**2))
    E0fft = np.fft.fft2(E0)
    G = H * E0fft
    Ef = np.fft.ifft2(G)
    
    return Ef

@pytest.fixture(params=[
    (0, 0, 0.0),
    (3, 0, 0.0),
    (0, 3, 0.0),
    (3, -3, 0.0),
    (-3, 3, 0.0),
])
def gaussian_beam_freespace_model(request):
    wavelength = 0.01
    size = 2048
    pixel_size = 0.01
    wo = 0.1
    prop_dist = 25

    theta_x, theta_y, x0 = request.param
    
    det_shape = (size, size)

    deg_yx = np.deg2rad((theta_y, theta_x))
    tilt_yx = np.tan(deg_yx)

    components = (
        comp.GaussBeam(
            z=0.0,
            voltage=calculate_phi_0(wavelength),
            radius=x0,
            wo=wo,
            tilt_yx=tilt_yx
        ),
        comp.AccumulatingDetector(
            z=prop_dist,
            pixel_size=pixel_size,
            shape=det_shape,
            buffer_length=64,
        ),
    )
    
    model = Model(components)
    
    return model, wavelength, deg_yx, x0, wo, prop_dist


def test_guoy_phase():
    
    # I should be able to think of a better test here, but for now this will do
    # The test is that with a Qpinv of complex identity matrix, the guoy phase should be pi/2
    Qpinv = xp.array([[[1 + 0j, 0], [0, 1 + 0j]]])
    guoy_gbd = guoy_phase(Qpinv, xp)
    xp.testing.assert_allclose(guoy_gbd, np.pi/2, atol = 1e-5)
    

def test_gaussian_amplitude():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[0, 0], [0, 0]])
    Qinv = np.array([[0 + 0j, 0], [0, 0 + 0j]])
    
    known_result = 1 + 0j
    amplitude = gaussian_amplitude(Qinv, A, B, xp)
    
    xp.testing.assert_allclose(amplitude, known_result, atol = 1e-5)


def test_gaussian_free_space(gaussian_beam_freespace_model):
    n_rays = 1
    
    model, wavelength, deg_yx, x0, wo, prop_dist = gaussian_beam_freespace_model

    rays = tuple(model.run_iter(num_rays=n_rays, random = False))
    gbd_output_field = model.detector.get_image(rays[-1])

    size = gbd_output_field.shape[0]
    gbd_output_field= zero_phase(gbd_output_field, gbd_output_field.shape[0]//2, gbd_output_field.shape[1]//2)
    
    # Calculate theta and phi
    tan_theta_x = np.tan(deg_yx[1])
    tan_theta_y = np.tan(deg_yx[0])
    
    pixel_size = model.components[1].pixel_size
    
    theta = np.arctan(np.sqrt(tan_theta_x**2 + tan_theta_y**2))
    phi = np.arctan2(tan_theta_y, tan_theta_x)
    
    shape = model.components[-1].shape
    det_size_y = shape[0] * pixel_size
    det_size_x = shape[1] * pixel_size

    x_det = xp.linspace(-det_size_x / 2, det_size_x / 2, shape[0])
    y_det = xp.linspace(-det_size_y / 2, det_size_y / 2, shape[1])

    fresnel_input_field = Scalar_source_XY(x=x_det, y=y_det, wavelength=wavelength)
    fresnel_input_field.gauss_beam(A=1, r0=(x0, 0), z0=0, w0=(wo, wo), theta=theta, phi=phi)
    fresnel_output_field = FresnelPropagator(fresnel_input_field.u, pixel_size, wavelength, prop_dist)
    fresnel_output_field = zero_phase(fresnel_output_field, size//2, size//2)
    
    # Create a mask for pixels within a 25 px radius from the center
    center_x, center_y = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    mask = xp.sqrt((x - center_x)**2 + (y - center_y)**2) <= 25

    # Check if all pixels within the mask satisfy the isclose condition
    is_close = np.isclose(
        np.angle(gbd_output_field),
        np.angle(fresnel_output_field),
        atol=0.1,
    )
    
    xp.testing.assert_(np.all(is_close[mask]))
    
    #Uncomment to plot the images - better to do this to be sure that it's working well enough - only the parameters of the last image
    #in the pytest fixture is saved!
        
    print(tan_theta_x, tan_theta_y)
    
    dpi = 200
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.abs(gbd_output_field),cmap='gray')
    ax1.axvline(size // 2, color='white', alpha=0.3)
    ax1.axhline(size // 2, color='white', alpha=0.3)
    ax2.imshow(np.angle(gbd_output_field),cmap='RdBu')
    ax2.axvline(size // 2, color='k', alpha=0.3)
    ax2.axhline(size // 2, color='k', alpha=0.3)
    fig.suptitle("GBD")
    fig.savefig("test_gaussian_gbd.png", dpi = dpi)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.abs(fresnel_output_field),cmap='gray')
    ax1.axvline(size // 2, color='white', alpha=0.3)
    ax1.axhline(size // 2, color='white', alpha=0.3)
    ax2.imshow(np.angle(fresnel_output_field ),cmap='RdBu')
    ax2.axvline(size // 2, color='k', alpha=0.3)
    ax2.axhline(size // 2, color='k', alpha=0.3)
    fig.suptitle("Fresnel")
    fig.savefig("test_gaussian_fresnel.png", dpi = dpi)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    s = np.s_[size // 2, :]
    ax1.plot(np.abs(fresnel_output_field[s]), label="Fresnel")
    ax1.plot(np.abs(gbd_output_field[s]), label="GBD")
    ax1.legend()
    ax2.plot(np.angle(fresnel_output_field[s]), label="Fresnel")
    ax2.plot(np.angle(gbd_output_field[s]), label="GBD")
    ax2.legend()
    fig.savefig("test_gaussian_fresnelvsgaussianline_close.png", dpi = dpi)

    fig, ax1 = plt.subplots()
    ax1.imshow(is_close)
    fig.savefig("test_gaussian_isclose.png", dpi = dpi)

    


