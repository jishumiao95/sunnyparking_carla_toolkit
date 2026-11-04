import json
import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from data_generation.sensor_specs import (
    LIDAR_ATTRIBUTES,
    SENSOR_GROUPS,
    SURROUND_POSES,
    expand_groups,
    parse_sensor_groups,
)

from data_generation.sensor_recap import (
    RecapError,
    _disable_vehicle_physics,
    _first_safe_saved_index,
    _select_ego,
    ego_pose_for_attach,
    filter_jobs_by_only,
    parse_only_pairs,
    actor_data_path,
    dir_is_complete,
    discover_failed_retries,
    discover_tasks,
    load_actor_records,
    pose_for_step,
    prepare_output_dir,
    prepare_replay_log_path,
    parse_mountinfo,
    read_frame_range,
    translate_server_path,
    read_saved_poses,
    select_ego_index,
    summarize_jobs,
    task_output_decision,
)


def _touch(path, payload=b'x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _measurement_symlink(directory, local_index, original_index):
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / ('%04d.json' % local_index)
    target = '/tmp/original/measurements/%04d.json' % original_index
    os.symlink(target, str(link))


class FrameRangeTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='recap_range_'))

    def tearDown(self):
        shutil.rmtree(str(self.root))

    def test_symlink_slice_uses_target_index(self):
        measurements = self.root / 'measurements'
        for offset in range(17, 198):
            _measurement_symlink(measurements, offset - 17, offset)
        start, end, count = read_frame_range(self.root)
        self.assertEqual((start, end, count), (17, 197, 181))

    def test_real_files_start_at_zero(self):
        measurements = self.root / 'measurements'
        for index in range(4):
            _touch(measurements / ('%04d.json' % index), b'{}')
        self.assertEqual(read_frame_range(self.root), (0, 3, 4))

    def test_gap_in_symlink_targets_fails(self):
        measurements = self.root / 'measurements'
        _measurement_symlink(measurements, 0, 5)
        _measurement_symlink(measurements, 1, 7)
        with self.assertRaises(RecapError):
            read_frame_range(self.root)


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='recap_discover_'))
        self.split = self.root / 'train_town04'

    def tearDown(self):
        shutil.rmtree(str(self.root))

    def _add_task(self, episode, task, log_bytes, frames, original_start):
        task_dir = self.split / episode / task
        replay = task_dir / 'replay'
        replay.mkdir(parents=True)
        header = b'\x01\x00\x0e\x00CARLA_RECORDER' + b'\x00' * 8 + b'\n\x00Town04_Opt'
        blob = header + (b'\x00' * log_bytes)
        (replay / 'recoding.log').write_bytes(blob)
        for index in range(frames):
            _measurement_symlink(task_dir / 'measurements', index, original_start + index)

    def test_classifies_usable_empty_and_missing_logs(self):
        self._add_task('ep_b', 'task2', 128, 3, 10)
        self._add_task('ep_a', 'task11', 128, 2, 4)
        self._add_task('ep_a', 'task1', 0, 1, 0)
        (self.split / 'ep_a' / 'task1' / 'replay' / 'recoding.log').write_bytes(b'short')
        (self.split / 'ep_c' / 'task3').mkdir(parents=True)
        (self.split / 'ep_c' / 'task3' / 'replay').mkdir()

        jobs = discover_tasks([self.split])
        by_key = {(job.episode, job.task_name): job for job in jobs}
        self.assertEqual(
            [(job.episode, job.task_name) for job in jobs],
            [('ep_a', 'task1'), ('ep_a', 'task11'), ('ep_b', 'task2'), ('ep_c', 'task3')])
        self.assertEqual(by_key[('ep_a', 'task1')].status, 'empty_log')
        self.assertEqual(by_key[('ep_c', 'task3')].status, 'no_log')
        runnable = by_key[('ep_a', 'task11')]
        self.assertEqual(runnable.status, 'runnable')
        self.assertEqual((runnable.frame_start, runnable.frame_end, runnable.local_count), (4, 5, 2))
        self.assertIn('runnable=2', summarize_jobs(jobs))
        self.assertIn('empty_log=1', summarize_jobs(jobs))
        self.assertIn('no_log=1', summarize_jobs(jobs))

    def test_episode_and_task_filters(self):
        self._add_task('alpha', 'task1', 128, 1, 0)
        self._add_task('alpha', 'task2', 128, 1, 0)
        self._add_task('beta', 'task1', 128, 1, 0)
        jobs = discover_tasks([self.split], episode_filters=['alpha'], task_filters=['task2'])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].task_name, 'task2')

    def test_retry_failed_reads_manifest_under_output_root(self):
        self._add_task('ep_a', 'task1', 128, 2, 4)
        log_path = self.split / 'ep_a' / 'task1' / 'replay' / 'recoding.log'
        mirror = self.root / 'out' / 'train_town04' / 'ep_a' / 'task1'
        mirror.mkdir(parents=True)
        manifest = {
            'status': 'failed',
            'error': 'ambiguous ego',
            'log': str(log_path.resolve()),
            'frame_range_original': [4, 5],
            'local_count': 2,
        }
        (mirror / 'recap_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        jobs = discover_failed_retries(
            self.root / 'out', [self.split], episode_filters=['ep_a'], task_filters=['task1'])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, 'runnable')
        self.assertEqual(jobs[0].log_path, log_path.resolve())
        self.assertEqual(jobs[0].episode, 'ep_a')


class ActorResynthDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='recap_actor_'))
        self.split = self.root / 'val_town04'

    def tearDown(self):
        shutil.rmtree(str(self.root))

    def _measurements(self, task_dir, count, original_start):
        for index in range(count):
            payload = {
                'x': 1.0 + index, 'y': 2.0, 'z': 0.3,
                'pitch': 0.0, 'yaw': 90.0, 'roll': 0.0,
            }
            real = self.root / 'original' / task_dir.name / ('%04d.json' % (original_start + index))
            _touch(real, json.dumps(payload).encode('utf-8'))
            link = task_dir / 'measurements' / ('%04d.json' % index)
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(real), str(link))

    def _actor(self, task_dir, records):
        path = actor_data_path(task_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            pickle.dump(records, handle)

    def test_missing_log_with_actor_is_resynth(self):
        task_dir = self.split / 'ep' / 'task3'
        (task_dir / 'replay').mkdir(parents=True)
        self._measurements(task_dir, 2, 6)
        self._actor(task_dir, [{'type': 'vehicle.audi.a2', 'x': 1.0, 'y': 2.0, 'yaw': 3.0}])
        jobs = discover_tasks([self.split])
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.status, 'resynth')
        self.assertTrue(job.resynthesized)
        self.assertEqual((job.frame_start, job.frame_end, job.local_count), (6, 7, 2))
        self.assertIn('RESYNTH', job.format_line())
        self.assertIn('resynth=1', summarize_jobs(jobs))
        records = load_actor_records(job.actor_path)
        self.assertEqual(records[0]['type'], 'vehicle.audi.a2')
        poses = read_saved_poses(task_dir, job.frame_start, job.frame_end)
        self.assertEqual(pose_for_step(poses, 6, 7, 0, 3)['x'], 1.0)
        self.assertEqual(pose_for_step(poses, 6, 7, 18, 3)['x'], 1.0)
        self.assertEqual(pose_for_step(poses, 6, 7, 21, 3)['x'], 2.0)

    def test_usable_log_beats_actor_file(self):
        task_dir = self.split / 'ep' / 'task1'
        replay = task_dir / 'replay'
        replay.mkdir(parents=True)
        header = b'\x01\x00\x0e\x00CARLA_RECORDER' + b'\x00' * 8 + b'\n\x00Town04_Opt'
        (replay / 'recoding.log').write_bytes(header + (b'\x00' * 128))
        self._measurements(task_dir, 1, 4)
        self._actor(task_dir, [{'type': 'vehicle.audi.a2', 'x': 0.0, 'y': 0.0, 'yaw': 0.0}])
        job = discover_tasks([self.split])[0]
        self.assertEqual(job.status, 'runnable')
        self.assertFalse(job.resynthesized)

    def test_prepare_replay_log_unlinks_directory_symlink(self):
        original = self.root / 'original_replay'
        original.mkdir()
        (original / 'recoding.log').write_bytes(b'short')
        task_dir = self.split / 'ep' / 'task1'
        task_dir.mkdir(parents=True)
        os.symlink(str(original), str(task_dir / 'replay'))
        log_path = prepare_replay_log_path(task_dir, None, 'val_town04', 'ep', 'task1')
        self.assertFalse(log_path.exists())
        self.assertFalse((task_dir / 'replay').is_symlink())
        self.assertTrue((original / 'recoding.log').is_file())
        self.assertEqual(log_path.parent, task_dir / 'replay')
        self.assertEqual(os.stat(str(task_dir / 'replay')).st_mode & 0o777, 0o777)


class ServerPathTest(unittest.TestCase):
    def test_bind_mount_rewrites_workspace_prefix(self):
        text = (
            '4433 4377 8:18 /home/USER/workspace /workspace rw,relatime - ext4 /dev/sdb2 rw\n'
        )
        maps = parse_mountinfo(text)
        client = '/workspace/dataset/carla_sim_parking/recap_probe/replay/recoding.log'
        self.assertEqual(
            translate_server_path(client, maps),
            '/home/USER/workspace/dataset/carla_sim_parking/recap_probe/replay/recoding.log')

    def test_longest_mount_wins(self):
        maps = parse_mountinfo(
            '1 0 8:1 /data /workspace rw - ext4 /dev/sda1 rw\n'
            '2 0 8:1 /data/set /workspace/dataset rw - ext4 /dev/sda1 rw\n'
        )
        self.assertEqual(
            translate_server_path('/workspace/dataset/a.log', maps),
            '/data/set/a.log')

    def test_unmapped_path_is_unchanged(self):
        self.assertEqual(translate_server_path('/tmp/recoding.log', []), '/tmp/recoding.log')


class OutputPrepTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='recap_out_'))

    def tearDown(self):
        shutil.rmtree(str(self.root))

    def test_directory_symlink_is_unlinked_not_followed(self):
        target = self.root / 'original_depth'
        marker = target / 'keep.txt'
        _touch(marker, b'keep')
        link = self.root / 'depth_front'
        os.symlink(str(target), str(link))
        prepare_output_dir(link)
        self.assertTrue(marker.is_file())
        self.assertFalse(link.is_symlink())
        self.assertTrue(link.is_dir())
        self.assertEqual(list(link.iterdir()), [])

    def test_skip_when_real_files_match_count(self):
        from data_generation.sensor_recap import TaskJob

        spec = SENSOR_GROUPS['depth'][0]
        task_dir = self.root / 'train' / 'ep' / 'task1'
        out = task_dir / spec.output_dir
        for index in range(2):
            _touch(out / spec.filename(index), b'png')
        job = TaskJob(
            self.root / 'train', 'ep', 'task1', task_dir, 'runnable', '',
            frame_start=0, frame_end=1, local_count=2)
        decision, _reason = task_output_decision(
            job, [spec], ['depth'], None, overwrite=False)
        self.assertEqual(decision, 'skip_complete')
        self.assertTrue(dir_is_complete(out, spec, 2))

    def test_partial_real_files_need_overwrite(self):
        from data_generation.sensor_recap import TaskJob

        spec = SENSOR_GROUPS['lidar'][0]
        task_dir = self.root / 'train' / 'ep' / 'task1'
        _touch(task_dir / spec.output_dir / '0000.ply', b'ply')
        job = TaskJob(
            self.root / 'train', 'ep', 'task1', task_dir, 'runnable', '',
            frame_start=0, frame_end=1, local_count=2)
        decision, reason = task_output_decision(
            job, [spec], ['lidar'], None, overwrite=False)
        self.assertEqual(decision, 'skip_partial')
        self.assertIn('--overwrite', reason)

    def test_failed_manifest_allows_rewrite(self):
        from data_generation.sensor_recap import TaskJob

        spec = SENSOR_GROUPS['lidar'][0]
        task_dir = self.root / 'train' / 'ep' / 'task1'
        _touch(task_dir / spec.output_dir / '0000.ply', b'ply')
        manifest = {
            'status': 'failed',
            'sensors': ['lidar'],
        }
        (task_dir / 'recap_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        job = TaskJob(
            self.root / 'train', 'ep', 'task1', task_dir, 'runnable', '',
            frame_start=0, frame_end=1, local_count=2)
        decision, _reason = task_output_decision(
            job, [spec], ['lidar'], None, overwrite=False)
        self.assertEqual(decision, 'write')


class ReplayIndexTest(unittest.TestCase):
    def test_first_safe_saved_frame_is_after_attach_tick(self):
        self.assertEqual(_first_safe_saved_index(0, 3), 1)
        self.assertEqual(_first_safe_saved_index(2, 3), 1)
        self.assertEqual(_first_safe_saved_index(3, 3), 2)


class SensorGroupTest(unittest.TestCase):
    def test_depth_matches_collection_poses(self):
        specs = expand_groups(['depth'])
        self.assertEqual(len(specs), 4)
        for spec, pose in zip(specs, SURROUND_POSES):
            view, x, y, z, roll, pitch, yaw = pose
            self.assertEqual(spec.sensor_id, 'depth_%s' % view)
            self.assertEqual(spec.blueprint, 'sensor.camera.depth')
            self.assertEqual(spec.transform['x'], x)
            self.assertEqual(spec.transform['pitch'], pitch)
            self.assertEqual(spec.transform['yaw'], yaw)
            self.assertEqual(spec.attributes['image_size_x'], '400')
            self.assertEqual(spec.attributes['fov'], '100')

    def test_lidar_spec_is_explicit(self):
        specs = expand_groups(['lidar'])
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].file_pattern, '%04d.ply')
        for key in (
                'range', 'rotation_frequency', 'points_per_second', 'channels',
                'upper_fov', 'lower_fov', 'atmosphere_attenuation_rate',
                'dropoff_general_rate', 'dropoff_intensity_limit',
                'dropoff_zero_intensity'):
            self.assertIn(key, LIDAR_ATTRIBUTES)
            self.assertEqual(specs[0].attributes[key], LIDAR_ATTRIBUTES[key])

    def test_semantic_reuses_camera_path(self):
        specs = expand_groups(['semantic'])
        self.assertEqual([spec.sensor_id for spec in specs],
                         ['semantic_front', 'semantic_left', 'semantic_right', 'semantic_rear'])
        self.assertTrue(all(spec.blueprint == 'sensor.camera.semantic_segmentation' for spec in specs))

    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):
            parse_sensor_groups('depth,radar')

    def test_depth_and_lidar_ids_do_not_collide(self):
        specs = expand_groups(['depth', 'lidar'])
        self.assertEqual(len(specs), len(set(spec.sensor_id for spec in specs)))


class EgoSelectionTest(unittest.TestCase):
    def test_select_ego_index_picks_nearest_model3(self):
        pose = {'x': 10.0, 'y': 5.0, 'yaw': 0.0}
        candidates = [(20.0, 5.0), (10.2, 5.1), (10.0, 8.0)]
        self.assertEqual(select_ego_index(candidates, pose), 1)

    def test_select_ego_index_rejects_far_match(self):
        pose = {'x': 10.0, 'y': 5.0, 'yaw': 0.0}
        candidates = [(20.0, 5.0), (21.0, 5.0)]
        with self.assertRaises(RecapError):
            select_ego_index(candidates, pose, max_dist_m=5.0)

    def test_attach_pose_is_withheld_before_the_pre_attach_step(self):
        poses = {5: {'x': 1.0, 'y': 2.0, 'yaw': 0.0, 'z': 0.0, 'pitch': 0.0, 'roll': 0.0}}
        # frame_start 5, save_frequency 3 -> first attach step is 14.
        self.assertIsNone(ego_pose_for_attach(poses, 5, 5, 0, 3))
        self.assertIsNone(ego_pose_for_attach(poses, 5, 5, 13, 3))
        self.assertEqual(ego_pose_for_attach(poses, 5, 5, 14, 3)['x'], 1.0)
        self.assertEqual(ego_pose_for_attach(poses, 5, 5, 15, 3)['x'], 1.0)


class _FakeLocation(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeTransform(object):
    def __init__(self, x, y):
        self.location = _FakeLocation(x, y)


class _FakeModel3(object):
    def __init__(self, x, y, hero=False):
        self.hero = hero
        self.transform = _FakeTransform(x, y)
        self.attributes = {'role_name': 'hero' if hero else ''}

    def get_transform(self):
        return self.transform


class EgoActorSelectionTest(unittest.TestCase):
    def test_single_model3_waits_until_a_pose_exists(self):
        actor = _FakeModel3(0.0, 0.0, hero=False)
        self.assertIsNone(_select_ego(_FakeWorld([actor]), None))

    def test_single_model3_is_ego_once_a_pose_exists(self):
        actor = _FakeModel3(0.0, 0.0, hero=False)
        pose = {'x': 10.0, 'y': 5.0, 'yaw': 0.0}
        self.assertIs(_select_ego(_FakeWorld([actor]), pose), actor)

    def test_two_far_model3s_wait_instead_of_raising(self):
        actors = [_FakeModel3(0.0, 0.0), _FakeModel3(1.0, 0.0)]
        pose = {'x': 50.0, 'y': 50.0, 'yaw': 0.0}
        self.assertIsNone(_select_ego(_FakeWorld(actors), pose))

    def test_stale_hero_does_not_beat_the_closer_log_ego(self):
        leftover = _FakeModel3(0.0, 0.0, hero=True)
        log_ego = _FakeModel3(10.0, 5.0, hero=False)
        pose = {'x': 10.1, 'y': 5.0, 'yaw': 0.0}
        chosen = _select_ego(_FakeWorld([leftover, log_ego]), pose)
        self.assertIs(chosen, log_ego)

    def test_two_model3s_wait_until_a_pose_exists(self):
        actors = [_FakeModel3(0.0, 0.0, hero=True), _FakeModel3(3.0, 0.0, hero=False)]
        self.assertIsNone(_select_ego(_FakeWorld(actors), None))


class _FakeVehicle(object):
    def __init__(self):
        self.physics = True

    def set_simulate_physics(self, enabled):
        self.physics = enabled


class _FakeActorList(object):
    def __init__(self, actors):
        self._actors = actors

    def filter(self, pattern):
        return self._actors


class _FakeWorld(object):
    def __init__(self, actors):
        self._actors = actors

    def get_actors(self):
        return _FakeActorList(self._actors)


class OnlyFilterTest(unittest.TestCase):
    def test_parse_only_pairs(self):
        self.assertEqual(
            parse_only_pairs(['ep_a/task1', 'ep_b/task9']),
            [('ep_a', 'task1'), ('ep_b', 'task9')])
        with self.assertRaises(RecapError):
            parse_only_pairs(['task1'])

    def test_filter_jobs_by_only_requires_every_pair(self):
        class _Job(object):
            def __init__(self, episode, task_name):
                self.episode = episode
                self.task_name = task_name
        jobs = [_Job('ep_a', 'task1'), _Job('ep_a', 'task2'), _Job('ep_b', 'task9')]
        picked = filter_jobs_by_only(jobs, [('ep_a', 'task1'), ('ep_b', 'task9')])
        self.assertEqual([(job.episode, job.task_name) for job in picked],
                         [('ep_a', 'task1'), ('ep_b', 'task9')])
        with self.assertRaises(RecapError):
            filter_jobs_by_only(jobs, [('ep_a', 'task15')])


class VehiclePhysicsTest(unittest.TestCase):
    def test_disable_vehicle_physics_turns_simulation_off(self):
        vehicles = [_FakeVehicle(), _FakeVehicle()]
        _disable_vehicle_physics(_FakeWorld(vehicles))
        self.assertEqual([vehicle.physics for vehicle in vehicles], [False, False])


if __name__ == '__main__':
    unittest.main()
