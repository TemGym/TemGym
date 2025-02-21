from typing import (
    Generator, Optional,
)

import numpy as np

from .components import Sample
from .rays import Rays

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

