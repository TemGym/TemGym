from typing import TypedDict

import pytest
import numpy as np

from temgymbasic.model import STEMModel
from scipy.ndimage import center_of_mass


def rotate(radians):
    # copied from libertem.com_analysis
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    return np.array([
        (np.cos(radians), np.sin(radians)),
        (-np.sin(radians), np.cos(radians))
    ])


def rotate_deg(degrees):
    # copied from libertem.com_analysis
    return rotate(np.pi/180*degrees)


def flip_y_mat():
    # copied from libertem.com_analysis
    return np.array([
        (-1, 0),
        (0, 1)
    ])


def identity():
    # copied from libertem.com_analysis
    return np.eye(2)


def apply_correction(y_centers, x_centers, scan_rotation, flip_y, forward=True):
    # copied from libertem.com_analysis
    shape = y_centers.shape
    if flip_y:
        transform = flip_y_mat()
    else:
        transform = identity()
    # Transformations are applied right to left
    transform = rotate_deg(scan_rotation) @ transform
    y_centers = y_centers.reshape(-1)
    x_centers = x_centers.reshape(-1)
    if not forward:
        transform = np.linalg.inv(transform)
    y_transformed, x_transformed = transform @ (y_centers, x_centers)
    y_transformed = y_transformed.reshape(shape)
    x_transformed = x_transformed.reshape(shape)
    return (y_transformed, x_transformed)


class OverfocusParams(TypedDict):
    overfocus: float  # m
    scan_pixel_size: float  # m
    camera_length: float  # m
    detector_pixel_size: float  # m
    semiconv: float  # rad
    cy: float
    cx: float
    scan_rotation: float  # deg
    flip_y: bool


def make_model(params: OverfocusParams, nav_shape, sig_shape) -> STEMModel:
    # Adapted from https://github.com/LiberTEM/Microscope-Calibration/blob/
    # 7e08ca84a6539ac8e4e4068c5f7ea48e97db63d2/src/microscope_calibration/
    # common/stem_overfocus.py#L23-L37
    model = STEMModel()
    model.set_stem_params(
        overfocus=params['overfocus'],
        semiconv_angle=params['semiconv'],
        scan_step_yx=(params['scan_pixel_size'], params['scan_pixel_size']),
        scan_shape=nav_shape,
        camera_length=params['camera_length'],
    )
    model.detector.pixel_size = params['detector_pixel_size']
    model.detector.shape = sig_shape
    model.detector.flip_y = params['flip_y']
    model.detector.rotation = params['scan_rotation']
    model.detector.set_center_px((params['cy'], params['cx']))
    return model


def detector_px_to_specimen_px(
        y_px, x_px, cy, cx, detector_pixel_size, scan_pixel_size, camera_length,
        overfocus, transformation_matrix, fov_size_y, fov_size_x):
    position_y, position_x = (y_px - cy) * detector_pixel_size, (x_px - cx) * detector_pixel_size
    position_y, position_x = transformation_matrix @ np.array((position_y, position_x))
    specimen_position_y = position_y / camera_length * overfocus
    specimen_position_x = position_x / camera_length * overfocus
    specimen_px_x = specimen_position_x / scan_pixel_size + fov_size_x / 2
    specimen_px_y = specimen_position_y / scan_pixel_size + fov_size_y / 2
    return specimen_px_y, specimen_px_x


def get_transformation_matrix(sim_params: OverfocusParams):
    transformation_matrix = np.array(apply_correction(
        y_centers=np.array((1, 0)),
        x_centers=np.array((0, 1)),
        scan_rotation=sim_params['scan_rotation'],
        flip_y=sim_params['flip_y'],
    ))
    return transformation_matrix


@pytest.mark.parametrize(
    'params_update', [
        {},
        {
            'scan_rotation': 23,
            'cy': 2.3,
            'cx': 4.7,
            'overfocus': 0.00012345,
            'camera_length': 0.87654,
            'detector_pixel_size': 0.0009876541,
            'scan_pixel_size': 0.000000019876,
            'flip_y': True,
        },
    ]
)
@pytest.mark.parametrize(
    'nav_shape', [(8, 8), (7, 5)]
)
@pytest.mark.parametrize(
    'sig_shape', [(8, 8), (11, 13)]
)
@pytest.mark.parametrize(
    # make sure the test is sensitive
    'fail', [False, True]
)
def test_rays(params_update, nav_shape, sig_shape, fail) -> np.ndarray:
    params = OverfocusParams(
        overfocus=0.0001,
        scan_pixel_size=0.00000001,
        camera_length=1,
        detector_pixel_size=0.0001,
        semiconv=0.01,
        cy=3,
        cx=3,
        scan_rotation=33.3,
        flip_y=False,
    )

    fail_factor = 1.01 if fail else 1

    params.update(params_update)

    nav_shape = (8, 8)
    sig_shape = (8, 8)

    model = make_model(params, nav_shape=nav_shape, sig_shape=sig_shape)
    yxs = (
        (0, 0),
        (model.sample.scan_shape[0], model.sample.scan_shape[1]),
        (0, model.sample.scan_shape[1]),
        (model.sample.scan_shape[0], 0),
    )
    num_rays = 7

    a = []
    b = []

    for yx in yxs:
        for rays in model.scan_point_iter(num_rays=num_rays, yx=yx):
            if rays.location is model.sample:
                yyxx = np.stack(
                    model.sample.on_grid(rays, as_int=False),
                    axis=-1,
                )
                coordinates = np.tile(
                    np.asarray((*yx, 1)).reshape(-1, 3),
                    (rays.num, 1),
                )
                a.append(np.concatenate((yyxx, coordinates), axis=-1))
            elif rays.location is model.detector:
                yy, xx = model.detector.on_grid(rays, as_int=False)
                b.append(np.stack((yy, xx), axis=-1))

    fail_params = params.copy()
    fail_params['scan_rotation'] *= fail_factor
    transformation_matrix = get_transformation_matrix(fail_params)

    for i, specimen_yxs in enumerate(a):
        for j, (spec_y, spec_x, scan_y, scan_x, one) in enumerate(specimen_yxs):
            assert one == 1
            det_y, det_x = b[i][j]
            ref_specimen_y_u, ref_specimen_x_u = detector_px_to_specimen_px(
                y_px=float(det_y),
                x_px=float(det_x),
                fov_size_y=float(nav_shape[0]),
                fov_size_x=float(nav_shape[1]),
                transformation_matrix=transformation_matrix,
                cy=params['cy'] * fail_factor,
                cx=params['cx'] / fail_factor,
                detector_pixel_size=float(params['detector_pixel_size']) / fail_factor,
                scan_pixel_size=float(params['scan_pixel_size']) * fail_factor,
                camera_length=float(params['camera_length']) * fail_factor,
                overfocus=float(params['overfocus']) / fail_factor,
            )
            offset_y = scan_y - nav_shape[0] / 2
            offset_x = scan_x - nav_shape[1] / 2
            image_px_y = ref_specimen_y_u + offset_y
            image_px_x = ref_specimen_x_u + offset_x
            if fail:
                with pytest.raises(AssertionError):
                    np.testing.assert_allclose(image_px_y, spec_y, rtol=1e-3, atol=1e-3)
                    np.testing.assert_allclose(image_px_x, spec_x, rtol=1e-3, atol=1e-3)
            else:
                np.testing.assert_allclose(image_px_y, spec_y, rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(image_px_x, spec_x, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    'det_coordinate', [(8, 8), (11, 13), (7, 31)]
)
def test_descan_error_simple_offset(det_coordinate):
    model = STEMModel()
    model.sample.descan_error = np.asarray(
        [[det_coordinate, [0, 0], [0, 0]]]
    ).astype(np.float32)
    model.sample.scan_shape = (24, 42)
    model.detector.shape = (48, 46)
    scan_point = tuple(np.random.randint(s) for s in model.detector.shape)
    model.move_to(scan_point)
    rays = model.run_to_end(256)
    image = model.detector.get_image(rays)
    com = center_of_mass(image)
    np.testing.assert_allclose(com, det_coordinate, atol=0.1)


@pytest.mark.parametrize(
    'repeat', range(5)
)
def test_descan_error(repeat):
    model = STEMModel()
    model.sample.descan_error = np.asarray(
        [[[17, 30], [-0.2, 0.5], [0.3, -0.4]]]
    ).astype(np.float32)
    model.sample.scan_shape = (24, 42)
    model.detector.shape = (48, 46)
    max_coord = ((1, *model.detector.shape) @ model.sample.descan_error).ravel()
    assert (m <= s for m, s in zip(max_coord, model.detector.shape))
    scan_point = tuple(np.random.randint(s) for s in model.detector.shape)
    ideal_coord = (np.asarray((1, *scan_point)) @ model.sample.descan_error).ravel()
    model.move_to(scan_point)
    rays = model.run_to_end(256)
    image = model.detector.get_image(rays)
    com = center_of_mass(image)
    np.testing.assert_allclose(com, ideal_coord, atol=0.1)
