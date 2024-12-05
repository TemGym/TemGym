import numpy as np
from temgymbasic.gbd import (gaussian_amplitude, guoy_phase)
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


def test_guoy_phase():

    # I should be able to think of a better test here, but for now this will do
    # The test is that with a Qpinv of complex identity matrix, the guoy phase should be pi/2
    Qpinv = xp.array([[[1 + 0j, 0], [0, 1 + 0j]]])
    guoy_gbd = guoy_phase(Qpinv, xp)
    xp.testing.assert_allclose(guoy_gbd, np.pi/2, atol=1e-5)


def test_gaussian_amplitude():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[0, 0], [0, 0]])
    Qinv = np.array([[0 + 0j, 0], [0, 0 + 0j]])

    # Test here is that our function should return complex amplitude of 1 + 0j
    known_result = 1 + 0j
    amplitude = gaussian_amplitude(Qinv, A, B, xp)

    xp.testing.assert_allclose(amplitude, known_result, atol=1e-5)
