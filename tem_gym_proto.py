from typing import Generator, Iterable, Tuple, Optional, Union, Literal, Type
from itertools import pairwise
import numpy as np
from numpy.typing import NDArray

from temgymbasic.functions import (
    circular_beam,
    point_beam,
    axial_point_beam,
    x_axial_point_beam,
    get_pixel_coords,
)


class Component:
    def __init__(self, z: float, name: str):
        self._name = name
        self._z = z

    @property
    def z(self) -> float:
        return self._z

    @property
    def entrance_z(self) -> float:
        return self.z

    @property
    def exit_z(self) -> float:
        return self.z

    def __repr__(self):
        return f'{self.__class__.__name__}: {self._name} @ z = {self._z}'

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        raise NotImplementedError

    @staticmethod
    def gui_wrapper() -> Type['ComponentGUIWrapper']:
        return ComponentGUIWrapper


class ComponentGUIWrapper:
    def __init__(self, component: Component):
        self.component = component


class Lens(Component):
    def __init__(self, z: float, f: float, name: str = "Lens"):
        super().__init__(name=name, z=z)
        self._f = f
        self._matrix = self.lens_matrix(f)

    @staticmethod
    def lens_matrix(f):
        '''
        Lens ray transfer matrix

        Parameters
        ----------
        f : float
            Focal length of lens

        Returns
        -------
        ndarray
            Output Ray Transfer Matrix
        '''

        return np.array(
            [[1,      0, 0,      0, 0],
             [-1 / f, 1, 0,      0, 0],
             [0,      0, 1,      0, 0],
             [0,      0, -1 / f, 1, 0],
             [0,      0, 0,      0, 1]]
        )

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        # Just straightforward matrix multiplication
        yield self.z, np.matmul(self._matrix, rays)


class Gun(Component):
    def __init__(
        self,
        z: float,
        beam_type: Literal['parallel', 'point', 'axial', 'x_axial'],
        beam_radius: float,
        beam_semi_angle: float,
        beam_tilt_yx: Tuple[float, float],
        name: str = "Gun",
    ):
        super().__init__(name=name, z=z)
        self.beam_type = beam_type
        self.beam_radius = beam_radius
        self.beam_semi_angle = beam_semi_angle
        self.beam_tilt_yx = beam_tilt_yx

    def get_rays(self, num_rays: int) -> NDArray:
        if self.beam_type == 'parallel':
            r, spot_indices = circular_beam(num_rays, self.beam_radius)
        elif self.beam_type == 'point':
            r, spot_indices = point_beam(num_rays, self.beam_semi_angle)
        elif self.beam_type == 'axial':
            r = axial_point_beam(num_rays, self.beam_semi_angle)
        elif self.beam_type == 'x_axial':
            r = x_axial_point_beam(num_rays, self.beam_semi_angle)
        else:
            raise ValueError()

        r[1, :] += self.beam_tilt_yx[1]
        r[3, :] += self.beam_tilt_yx[0]
        return r

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        # Gun has no effect after get_rays was called
        yield self.z, rays


class Detector(Component):
    def __init__(
        self,
        z: float,
        detector_size: float,
        detector_pixels: int,
        flip_y: bool = False,
        name: str = "Detector",
    ):
        super().__init__(name=name, z=z)
        self.detector_size = detector_size
        self.detector_pixels = detector_pixels
        self.flip_y = flip_y

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        # Detector has no effect on rays
        yield self.z, rays

    def get_image(self, rays: NDArray) -> NDArray:
        rays_y = rays[2]
        rays_x = rays[0]
        # Convert rays from detector positions to pixel positions
        detector_pixel_coords_x, detector_pixel_coords_y = np.round(
            get_pixel_coords(
                rays_x=rays_x,
                rays_y=rays_y,
                size=self.detector_size,
                pixels=self.detector_pixels,
                flip_y=self.flip_y
            )
        ).astype(int)
        mask = np.logical_and(
            np.logical_and(
                0 <= detector_pixel_coords_y,
                detector_pixel_coords_y < self.detector_pixels
            ),
            np.logical_and(
                0 <= detector_pixel_coords_x,
                detector_pixel_coords_x < self.detector_pixels
            )
        )
        image = np.zeros(
            (self.detector_pixels, self.detector_pixels),
            dtype=int,
        )
        flat_icds = np.ravel_multi_index(
            [
                detector_pixel_coords_y[mask],
                detector_pixel_coords_x[mask],
            ],
            image.shape
        )
        # Increment at each pixel for each ray that hits
        np.add.at(
            image.ravel(),
            flat_icds,
            1,
        )
        return image


class Deflector(Component):
    '''Creates a single deflector component and handles calls to GUI creation, updates to GUI
        and stores the component matrix. See Double Deflector component for a more useful version
    '''
    def __init__(
        self,
        z: float,
        defx: float = 0.5,
        defy: float = 0.5,
        name: str = "Deflector",
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        defy : float, optional
            deflection kick in slope units to the incoming ray y angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.defx = defx
        self.defy = defy

    @staticmethod
    def deflector_matrix(def_x, def_y):
        '''Single deflector ray transfer matrix

        Parameters
        ----------
        def_x : float
            deflection in x in slope units
        def_y : _type_
            deflection in y in slope units

        Returns
        -------
        ndarray
            Output ray transfer matrix
        '''

        return np.array(
            [[1, 0, 0, 0,     0],
             [0, 1, 0, 0, def_x],
             [0, 0, 1, 0,     0],
             [0, 0, 0, 1, def_y],
             [0, 0, 0, 0,     1]],
        )

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        yield self.z, np.matmul(
            self.deflector_matrix(self.defx, self.defy),
            rays
        )


class DoubleDeflector(Component):
    def __init__(
        self,
        z: float,
        u_dz: float,
        l_dz: float,
        u_defx: float = 0.5,
        u_defy: float = 0.5,
        l_defx: float = 0.5,
        l_defy: float = 0.5,
        name: str = "DoubleDeflector",
    ):
        super().__init__(z=z, name=name)
        self._upper = Deflector(
            z - u_dz,
            u_defx,
            u_defy,
        )
        self._lower = Deflector(
            z + l_dz,
            l_defx,
            l_defy,
        )

    @property
    def entrance_z(self) -> float:
        return self._upper.z

    @property
    def exit_z(self) -> float:
        return self._lower.z

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        for z, rays in self._upper.step(rays):
            yield z, rays
        Model.propagate(
            self._lower.entrance_z - z,
            rays
        )
        yield from self._lower.step(rays)


class Aperture(Component):
    def __init__(
        self,
        z: float,
        radius_inner: float = 0.005,
        radius_outer: float = 0.25,
        x: float = 0.,
        y: float = 0.,
        name: str = 'Aperture',
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default 'Aperture'
        radius_inner : float, optional
           Inner radius of the aperture, by default 0.005
        radius_outer : float, optional
            Outer radius of the aperture, by default 0.25
        x : int, optional
            X position of the centre of the aperture, by default 0
        y : int, optional
            Y position of the centre of the aperture, by default 0
        '''

        super().__init__(z, name)

        self.x = x
        self.y = y
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        pos_x, pos_y = rays[0], rays[2]
        distance = np.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        mask = np.logical_and(
            distance >= self.radius_inner,
            distance < self.radius_outer,
        )
        yield self.z, rays[:, mask]


class Model:
    def __init__(self, components: Iterable[Component]):
        assert isinstance(components[0], Gun)
        assert isinstance(components[-1], Detector)
        self._components = components

    def __repr__(self):
        repr_string = f'[{self.__class__.__name__}]:'
        for component in self._components:
            repr_string = repr_string + f'\n - {repr(component)}'
        return repr_string

    def run_iter(
        self, num_rays: int
    ) -> Generator[Tuple[Component, float, NDArray], None, None]:
        rays = None
        for this_component, next_component in pairwise(self._components):
            if rays is None:
                # At the gun
                rays = this_component.get_rays(num_rays)
            for rays_z, new_rays in this_component.step(rays):
                yield this_component, rays_z, new_rays
            rays = self.propagate(
                next_component.entrance_z - this_component.exit_z,
                new_rays
            )
        # At detector plane
        yield next_component, next_component.z, rays

    def run_to_end(self, num_rays: int) -> Optional[NDArray]:
        rays = None
        for _, _, rays in self.run_iter(num_rays):
            pass
        return rays

    def run_to_component(
        self,
        component: Component,
        num_rays: int,
        sub_plane: Optional[int] = None,
    ) -> Union[NDArray, Tuple[NDArray, ...]]:
        planes = []
        for this_component, _, rays in self.run_iter(num_rays):
            if len(planes) > 0 and this_component is not component:
                break
            if this_component is component:
                planes.append(rays)
                if sub_plane is not None and len(planes) == (sub_plane + 1):
                    break
        if len(planes) == 1:
            return planes[0]
        elif sub_plane is not None:
            return planes[sub_plane]
        return tuple(planes)

    @staticmethod
    def propagation_matix(z):
        '''
        Propagation matrix

        Parameters
        ----------
        z : float
            Distance to propagate rays

        Returns
        -------
        ndarray
            Propagation matrix
        '''
        return np.array(
            [[1, z, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, z, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]
        )

    @staticmethod
    def propagate(distance: float, rays: NDArray) -> NDArray:
        return np.matmul(
            Model.propagation_matix(distance),
            rays,
        )


class GUIModel:
    def __init__(self, model: Model):
        self._model = model
        self._gui_components = tuple(
            c.gui_wrapper()(c) for c in self._model._components
        )


if __name__ == '__main__':
    components = (
        Gun(
            z=1.,
            beam_type="parallel",
            beam_radius=0.01,
            beam_semi_angle=0.001,
            beam_tilt_yx=(0., 0.),
        ),
        Lens(
            z=0.5,
            f=-0.3,
        ),
        DoubleDeflector(
            z=0.25,
            u_dz=0.025,
            l_dz=0.025,
            u_defy=0.05,
            u_defx=0.05,
            l_defy=-0.025,
            l_defx=0.,
        ),
        Aperture(
            0.15,
            radius_inner=0.,
            radius_outer=0.0075,
        ),
        Detector(
            z=0.,
            detector_size=0.05,
            detector_pixels=128,
        ),
    )
    model = Model(components)
    print(model)

    det_rays = model.run_to_end(512)
    image = components[-1].get_image(det_rays)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

    # rays_y = det_rays[2]
    # rays_x = det_rays[0]
    # plt.plot(rays_x, rays_y, 'o')
    # plt.show()
