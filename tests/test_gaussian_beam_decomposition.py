import pytest
import numpy as np

from temgymbasic.model import (
    Model,
)
import temgymbasic.components as comp
from temgymbasic.rays import Rays
from temgymbasic.gbd import propagate_misaligned_gaussian
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

def test_propagate_misaligned_gaussian_free_space():
    
    
    expected_out = np.array([0.36787944+0.j])
    np.testing.assert_allclose(out, expected_out, rtol=1e-5)


