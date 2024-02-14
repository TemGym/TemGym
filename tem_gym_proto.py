from typing import Generator, Iterable, Tuple, Optional, Union, Literal, Type, TypeAlias
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


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float


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

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

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
        yield self.z, np.matmul(self.lens_matrix(self.f), rays)


class Sample(Component):
    def __init__(self, z: float, name: str = "Sample"):
        super().__init__(name=name, z=z)

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        # Sample has no effect, yet
        # Could implement ray intensity / attenuation ??
        yield self.z, rays


class STEMSample(Sample):
    def __init__(
        self,
        z: float,
        overfocus: NonNegativeFloat = 0.,
        semiconv_angle: PositiveFloat = 0.01,
        scan_shape: Tuple[int, int] = (8, 8),
        scan_step_yx: Tuple[PositiveFloat, PositiveFloat] = (0.01, 0.01),
        name: str = "STEMSample"
    ):
        super().__init__(name=name, z=z)
        self.overfocus = overfocus
        self.semiconv_angle = semiconv_angle
        self.scan_shape = scan_shape
        self.scan_step_yx = scan_step_yx


class Gun(Component):
    def __init__(
        self,
        z: float,
        beam_type: Literal['parallel', 'point', 'axial', 'x_axial'],
        beam_radius: Optional[float] = None,
        beam_semi_angle: Optional[float] = None,
        beam_tilt_yx: Tuple[float, float] = (0., 0.),
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
        pixel_size: float,
        shape: Tuple[int, int],
        scan_rotation: float = 0.,
        flip_y: bool = False,
        name: str = "Detector",
    ):
        super().__init__(name=name, z=z)
        self.pixel_size = pixel_size
        self.shape = shape
        self.scan_rotation = scan_rotation
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
        pixel_coords_y, pixel_coords_x = np.round(
            get_pixel_coords(
                rays_x=rays_x,
                rays_y=rays_y,
                shape=self.shape,
                pixel_size=self.pixel_size,
                flip_y=self.flip_y,
                scan_rotation=self.scan_rotation,
            )
        ).astype(int)
        sy, sx = self.shape
        mask = np.logical_and(
            np.logical_and(
                0 <= pixel_coords_y,
                pixel_coords_y < sy
            ),
            np.logical_and(
                0 <= pixel_coords_x,
                pixel_coords_x < sx
            )
        )
        image = np.zeros(
            self.shape,
            dtype=int,
        )
        flat_icds = np.ravel_multi_index(
            [
                pixel_coords_y[mask],
                pixel_coords_x[mask],
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
        defx: float = 0.,
        defy: float = 0.,
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
        upper: Deflector,
        lower: Deflector,
        name: str = "DoubleDeflector",
    ):
        super().__init__(
            z=(upper.z + lower.z) / 2,
            name=name,
        )
        self._upper = upper
        self._lower = lower

    @property
    def height(self) -> float:
        return abs(self._upper.z - self._lower.z)

    @property
    def upper(self) -> Deflector:
        return self._upper

    @property
    def lower(self) -> Deflector:
        return self._lower

    @property
    def z(self):
        return (self.upper.z + self.lower.z) / 2

    @property
    def entrance_z(self) -> float:
        return self.upper.z

    @property
    def exit_z(self) -> float:
        return self.lower.z

    def step(
        self, rays: NDArray
    ) -> Generator[Tuple[float, NDArray], None, None]:
        for z, rays in self.upper.step(rays):
            yield z, rays
        Model.propagate(
            self.lower.entrance_z - self.upper.exit_z,
            rays
        )
        yield from self.lower.step(rays)


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
        self._components = components
        assert len(self._components) >= 2
        assert isinstance(self.gun, Gun)
        assert isinstance(self.detector, Detector)

    def __repr__(self):
        repr_string = f'[{self.__class__.__name__}]:'
        for component in self._components:
            repr_string = repr_string + f'\n - {repr(component)}'
        return repr_string

    @property
    def gun(self) -> Gun:
        return self._components[0]

    @property
    def detector(self) -> Detector:
        return self._components[-1]

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


class STEMModel(Model):
    def __init__(
        self,
        overfocus: Optional[NonNegativeFloat] = None,
        semiconv_angle: Optional[PositiveFloat] = None,
        scan_step_yx: Optional[Tuple[PositiveFloat, PositiveFloat]] = None,
        scan_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(self.default_components())
        self.set_stem_params(
            overfocus=overfocus,
            semiconv_angle=semiconv_angle,
            scan_step_yx=scan_step_yx,
            scan_shape=scan_shape,
        )

    @staticmethod
    def default_components():
        return (
            Gun(
                z=1.,
                beam_type="parallel",
                beam_radius=0.01,
            ),
            DoubleDeflector(
                upper=Deflector(z=0.775),
                lower=Deflector(z=0.725),
            ),
            Lens(
                z=0.6,
                f=-0.2,
            ),
            STEMSample(
                z=0.4,
            ),
            DoubleDeflector(
                upper=Deflector(z=0.275),
                lower=Deflector(z=0.225),
            ),
            Detector(
                z=0.,
                pixel_size=0.05,
                shape=(128, 128),
            ),
        )

    @property
    def scan_coils(self) -> DoubleDeflector:
        return self._components[1]

    @property
    def objective(self) -> Lens:
        return self._components[2]

    @property
    def sample(self) -> STEMSample:
        return self._components[3]

    @property
    def descan_coils(self) -> DoubleDeflector:
        return self._components[4]

    def set_stem_params(
        self,
        overfocus: Optional[NonNegativeFloat] = None,
        semiconv_angle: Optional[PositiveFloat] = None,
        scan_step_yx: Optional[Tuple[PositiveFloat, PositiveFloat]] = None,
        scan_shape: Optional[Tuple[int, int]] = None,
    ):
        if overfocus is not None:
            self.sample.overfocus = overfocus
        if semiconv_angle is not None:
            self.sample.semiconv_angle = semiconv_angle
        self.set_obj_lens_f_from_overfocus()
        self.set_beam_radius_from_semiconv()
        # These could be set externally, no invariants to hold
        if scan_step_yx is not None:
            self.sample.scan_step_yx = scan_step_yx
        if scan_shape is not None:
            self.sample.scan_shape = scan_shape

    def set_obj_lens_f_from_overfocus(self):
        if self.sample.overfocus < 0.:
            raise NotImplementedError(
                'Only support positive overfocus values (crossover above sample)'
            )
        # Lens f is always a negative number, and overfocus for now is always a positive number
        self.objective.f = -1 * (self.objective.z - self.sample.z - self.sample.overfocus)

    def set_beam_radius_from_semiconv(self):
        self.gun.beam_radius = abs(self.objective.f) * np.tan(self.sample.semiconv_angle)

    def update_scan_coil_ratios(self, scan_pixel_yx: Tuple[int, int]):
        scan_pixel_y, scan_pixel_x = scan_pixel_yx

        # Distance to front focal plane from bottom deflector
        dist_to_ffp = abs(self.scan_coils.exit_z - (self.objective.z + abs(self.objective.f)))
        dist_to_lens = abs(self.scan_coils.exit_z - self.objective.z)

        # Get the scan position in physical units
        scan_step_y, scan_step_x = self.sample.scan_step_yx
        sy, sx = self.sample.scan_shape
        scan_position_x = (scan_pixel_x - sx / 2.) * scan_step_x
        scan_position_y = (scan_pixel_y - sy / 2.) * scan_step_y

        # Scan coil setting
        sc_height = self.scan_coils.height
        sc_defratio = -1 * (1 + sc_height / dist_to_ffp)

        self.scan_coils.upper.defx = (
            scan_position_x / (sc_height + dist_to_lens * (1 + sc_defratio))
        )
        self.scan_coils.lower.defx = sc_defratio * self.scan_coils.upper.defx

        self.scan_coils.upper.defy = (
            scan_position_y / (sc_height + dist_to_lens * (1 + sc_defratio))
        )
        self.scan_coils.lower.defy = sc_defratio * self.scan_coils.upper.defy

        # Descan coil setting
        desc_height = self.descan_coils.height
        # desc_defratio = -1 * (1 + desc_height / dist_to_ffp)

        self.descan_coils.upper.defx = (
            -self.scan_coils.upper.defx
            * (sc_height + dist_to_lens * (1 + sc_defratio)) / desc_height
        )
        self.descan_coils.lower.defx = -self.descan_coils.upper.defx

        self.descan_coils.upper.defy = (
            -self.scan_coils.upper.defy
            * (sc_height + dist_to_lens * (1 + sc_defratio)) / desc_height
        )
        self.descan_coils.lower.defy = -self.descan_coils.upper.defy

    def scan_point(self, num_rays: int, yx: Tuple[int, int]) -> NDArray:
        self.update_scan_coil_ratios(yx)
        return self.run_to_end(num_rays)

    def scan_point_iter(
        self, num_rays: int, yx: Tuple[int, int]
    ) -> Generator[Tuple[Component, float, NDArray], None, None]:
        self.update_scan_coil_ratios(yx)
        yield from self.run_iter(num_rays)

    def scan(self, num_rays: int) -> Generator[Tuple[Tuple[int, int], NDArray], None, None]:
        sy, sx = self.sample.scan_shape
        for y in range(sy):
            for x in range(sx):
                pos = (y, x)
                yield pos, self.scan_point(num_rays, pos)


class GUIModel:
    def __init__(self, model: Model):
        self._model = model
        self._gui_components = tuple(
            c.gui_wrapper()(c) for c in self._model._components
        )


if __name__ == '__main__':
    model = STEMModel()
    model.set_stem_params(semiconv_angle=0.2)
    rays = model.scan_point(num_rays=512, yx=(3, 5))
    model.detector.shape = (512, 512)
    model.detector.pixel_size = 0.002
    image = model.detector.get_image(rays)

    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(rays[0], rays[2], 'o')
    # br = (np.asarray(model.detector.shape) // 2) * model.detector.pixel_size
    # b, r = br  # noqa
    # tl = -1 * br
    # t, l = tl  # noqa
    # tr = [t, r]
    # bl = [b, l]
    # points = np.stack((tl, tr, br, bl), axis=0)
    # ax.fill(points[:, 1], points[:, 0], facecolor='none', edgecolor='black')
    # ax.invert_yaxis()
    # ax.axis('equal')

    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.show()

    exit(0)

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
            f=-0.2,
        ),
        DoubleDeflector(
            upper=Deflector(
                z=0.275,
                defy=0.05,
                defx=0.05,
            ),
            lower=Deflector(
                z=0.225,
                defy=-0.025,
                defx=0.,
            ),
        ),
        Sample(
            z=0.2
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

    det_rays = model.run_to_component(model.detector, 512)
    image = model.detector.get_image(det_rays)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

    # rays_y = det_rays[2]
    # rays_x = det_rays[0]
    # plt.plot(rays_x, rays_y, 'o')
    # plt.show()
