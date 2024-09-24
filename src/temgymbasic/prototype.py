from typing import (
    Generator, Optional,
)

import numpy as np

from .components import Sample
from .rays import Rays


from scipy.constants import c, e, m_e

# Defining epsilon constant from page 18 of principles of electron optics 2017, Volume 1.
EPSILON = abs(e)/(2*m_e*c**2)

class PotentialSample(Sample):
    def __init__(
        self,
        z: float,
        potential,
        Ex,
        Ey,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)

        # We're renaming here some terms to be closer to the math in Hawkes
        # Not sure if this is recommended or breaks any convetions
        self.phi = potential
        self.dphi_dx = Ex
        self.dphi_dy = Ey

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        # See Chapter 2 & 3 of principles of electron optics 2017 Vol 1 for more info
        rho = np.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)  # Equation 3.16
        phi_0_plus_phi = (rays.phi_0 + self.phi((rays.x, rays.y)))  # Part of Equation 2.18

        phi_hat = (phi_0_plus_phi) * (1 + EPSILON * (phi_0_plus_phi))  # Equation 2.18

        # Between Equation 2.22 & 2.23
        dphi_hat_dx = (1 + 2 * EPSILON * (phi_0_plus_phi)) * self.dphi_dx((rays.x, rays.y))
        dphi_hat_dy = (1 + 2 * EPSILON * (phi_0_plus_phi)) * self.dphi_dy((rays.x, rays.y))

        # Perform deflection to ray in slope coordinates
        rays.dx += ((rho ** 2) / (2 * phi_hat)) * dphi_hat_dx  # Equation 3.22
        rays.dy += ((rho ** 2) / (2 * phi_hat)) * dphi_hat_dy  # Equation 3.22

        # Note here we are ignoring the Ez component (dphi/dz) of 3.22,
        # since we have modelled the potential of the atom in a plane
        # only, we won't have an Ez component (At least I don't think this is the case?
        # I could be completely wrong here though - it might actually have an effect.
        # But I'm not sure I can get an Ez from an infinitely thin slice.

        # Equation 5.16 & 5.17 & 3.16, where ds of 5.16 is replaced by ds/dz * dz,
        # where ds/dz = rho (See 3.16 and a little below it)
        rays.path_length += rho * np.sqrt(phi_hat / rays.phi_0)

        # Currently the modifications are all inplace so we only need
        # to change the location, this should be made clearer
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import SampleGUI
        return SampleGUI


class MultisliceSample(Sample):
    def __init__(
        self,
        z: float,
        potential,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)

        # We're renaming here some terms to be closer to the math in Hawkes
        # Not sure if this is recommended or breaks any convetions
        self.phi = potential

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        # See Chapter 2 & 3 of principles of electron optics 2017 Vol 1 for more info
        rho = np.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)  # Equation 3.16

        # Equation 5.16 & 5.17 & 3.16, where ds of 5.16 is replaced by ds/dz * dz,
        # where ds/dz = rho (See 3.16 and a little below it)
        # if np.random.uniform() > 0.5:
        rays.path_length += rho * (np.sqrt(1 - self.phi((rays.x, rays.y)) / rays.phi_0) - 1)
        rays.dx += np.random.uniform(-1, 1, rays.num)
        # else:
        #     pass
        # rays.dy += np.random.uniform(-0.1, 0.1, rays.num)

        # Currently the modifications are all inplace so we only need
        # to change the location, this should be made clearer
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import SampleGUI
        return SampleGUI

