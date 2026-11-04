"""Sensor specs for log recap.

A batch run selects groups by name (`depth`, `lidar`, `semantic`). Each group
expands to one or more `SensorSpec`s. Adding a sensor means registering another
group here; the recap loop does not grow a per-sensor branch.
"""


# Surround-camera extrinsics copied from World.setup_sensors.
# Tuple is (view, x, y, z, roll, pitch, yaw) in the ego actor frame.
SURROUND_POSES = (
    ('front', 1.5, 0.0, 1.5, 0.0, 0.0, 0.0),
    ('left', 0.0, -0.8, 1.5, 0.0, -40.0, -90.0),
    ('right', 0.0, 0.8, 1.5, 0.0, -40.0, 90.0),
    ('rear', -2.2, 0.0, 1.5, 0.0, -30.0, 180.0),
)

CAMERA_ATTRIBUTES = {
    'image_size_x': '400',
    'image_size_y': '300',
    'fov': '100',
}

# Checked-in recap lidar. These are the ray-cast fields spawn_lidar already
# reads, plus range and lower_fov which that helper never set.
# rotation_frequency=10 at target_fps=30 completes one sweep per saved frame
# (save_frequency=3). z sits just above the surround cameras (z=1.5) on the
# same actor frame. Edit this block to retune the mount; do not scatter the
# numbers through the recap loop.
LIDAR_ATTRIBUTES = {
    'range': '50',
    'rotation_frequency': '10',
    'points_per_second': '500000',
    'channels': '32',
    'upper_fov': '10',
    'lower_fov': '-30',
    'atmosphere_attenuation_rate': '0.004',
    'dropoff_general_rate': '0.45',
    'dropoff_intensity_limit': '0.8',
    'dropoff_zero_intensity': '0.4',
}

LIDAR_TRANSFORM = {
    'x': 0.0,
    'y': 0.0,
    'z': 1.8,
    'roll': 0.0,
    'pitch': 0.0,
    'yaw': 0.0,
}


class SensorSpec(object):
    """One CARLA sensor attached during replay and one output directory."""

    def __init__(self, group, sensor_id, blueprint, output_dir, file_pattern,
                 attributes, transform):
        self.group = group
        self.sensor_id = sensor_id
        self.blueprint = blueprint
        self.output_dir = output_dir
        self.file_pattern = file_pattern
        self.attributes = dict(attributes)
        self.transform = dict(transform)

    @property
    def suffix(self):
        dot = self.file_pattern.rfind('.')
        if dot < 0:
            raise ValueError('file_pattern has no suffix: %s' % self.file_pattern)
        return self.file_pattern[dot:]

    def filename(self, index):
        return self.file_pattern % index


def _camera_group(group, blueprint, prefix):
    specs = []
    for view, x, y, z, roll, pitch, yaw in SURROUND_POSES:
        sensor_id = '%s_%s' % (prefix, view)
        specs.append(SensorSpec(
            group=group,
            sensor_id=sensor_id,
            blueprint=blueprint,
            output_dir=sensor_id,
            file_pattern='%04d.png',
            attributes=CAMERA_ATTRIBUTES,
            transform={
                'x': x, 'y': y, 'z': z,
                'roll': roll, 'pitch': pitch, 'yaw': yaw,
            },
        ))
    return specs


def _lidar_group():
    return [SensorSpec(
        group='lidar',
        sensor_id='lidar',
        blueprint='sensor.lidar.ray_cast',
        output_dir='lidar',
        file_pattern='%04d.ply',
        attributes=LIDAR_ATTRIBUTES,
        transform=LIDAR_TRANSFORM,
    )]


SENSOR_GROUPS = {
    'depth': _camera_group('depth', 'sensor.camera.depth', 'depth'),
    'lidar': _lidar_group(),
    'semantic': _camera_group('semantic', 'sensor.camera.semantic_segmentation', 'semantic'),
}


def available_groups():
    return tuple(sorted(SENSOR_GROUPS.keys()))


def parse_sensor_groups(text):
    """Parse a comma-separated group list. Reject unknowns and duplicates."""
    if text is None:
        raise ValueError('sensor list is empty')
    names = [part.strip() for part in text.split(',')]
    names = [name for name in names if name]
    if not names:
        raise ValueError('sensor list is empty')
    if len(names) != len(set(names)):
        raise ValueError('duplicate sensor group in %s' % names)
    unknown = [name for name in names if name not in SENSOR_GROUPS]
    if unknown:
        raise ValueError(
            'unknown sensor groups %s; available: %s' % (unknown, list(available_groups())))
    return names


def expand_groups(names):
    specs = []
    for name in names:
        specs.extend(SENSOR_GROUPS[name])
    ids = [spec.sensor_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError('expanded sensor ids collide: %s' % ids)
    return specs
