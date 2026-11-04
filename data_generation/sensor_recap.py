"""Replay CARLA recorder logs and write a selected sensor set.

Discovery, frame alignment, and output prep do not import CARLA. The client is
opened only when a task is actually replayed.

Tasks with no usable ``replay/recoding.log`` but with ``carla_actor/carla_actor.data``
are resynthesized: static vehicles are spawned from that pickle (the collection
seed was ``time.time()`` and was not stored), the ego is moved through
measurement poses, and a new recorder log is written before the normal replay.
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

from data_generation.sensor_specs import expand_groups, parse_sensor_groups

MIN_USABLE_LOG_BYTES = 64
TASK_DIR_RE = re.compile(r'^task(\d+)$')
RECORDER_MAGIC = b'CARLA_RECORDER'
MAP_TOKEN = b'Town04_Opt'
MANIFEST_NAME = 'recap_manifest.json'
SENSOR_TIMEOUT_SEC = 2.0
EGO_BLUEPRINT = 'vehicle.tesla.model3'
# After set_transform the actor is in meters. Larger than this means the snap
# did not stick and writing depth would repeat one frozen view.
EGO_SNAP_MAX_ERROR_M = 1.0
STATIC_NPC_Z = 0.3
ACTOR_RECORD_KEYS = ('type', 'x', 'y', 'yaw')
POSE_KEYS = ('x', 'y', 'z', 'pitch', 'yaw', 'roll')


class RecapError(Exception):
    pass


class TaskJob(object):
    def __init__(self, split_dir, episode, task_name, task_dir, status, reason,
                 log_path=None, log_bytes=0, frame_start=None, frame_end=None,
                 local_count=None, actor_path=None, actor_bytes=0,
                 resynthesized=False):
        self.split_dir = split_dir
        self.episode = episode
        self.task_name = task_name
        self.task_dir = task_dir
        self.status = status
        self.reason = reason
        self.log_path = log_path
        self.log_bytes = log_bytes
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.local_count = local_count
        self.actor_path = actor_path
        self.actor_bytes = actor_bytes
        self.resynthesized = resynthesized

    @property
    def split_name(self):
        return self.split_dir.name

    def format_line(self):
        if self.status in ('runnable', 'resynth'):
            label = 'RUNNABLE' if self.status == 'runnable' else 'RESYNTH'
            extra = 'bytes %d' % self.log_bytes
            if self.status == 'resynth':
                extra = 'actor_bytes %d' % self.actor_bytes
            return (
                '%s\t%s\t%s\t%s\tframes %d-%d\tcount %d\t%s'
                % (label, self.split_name, self.episode, self.task_name,
                   self.frame_start, self.frame_end, self.local_count, extra))
        return 'SKIP\t%s\t%s\t%s\t%s\t%s' % (
            self.status, self.split_name, self.episode, self.task_name, self.reason)


def _task_sort_key(name):
    match = TASK_DIR_RE.match(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def classify_log(log_path):
    """Return (status, size). status is ok, empty_log, bad_log, or no_log."""
    if not log_path.is_file():
        return 'no_log', 0
    size = log_path.stat().st_size
    if size <= MIN_USABLE_LOG_BYTES:
        return 'empty_log', size
    with log_path.open('rb') as handle:
        header = handle.read(64)
    if RECORDER_MAGIC not in header or MAP_TOKEN not in header:
        return 'bad_log', size
    return 'ok', size


def read_frame_range(task_dir):
    """Return (original_start, original_end, local_count) for one task.

    Symlinked measurements encode the slice: local 0000.json -> original
    NNNN.json. Real measurement files (no symlinks) map to saved frames
    [0, count-1].
    """
    measurements = task_dir / 'measurements'
    if not measurements.is_dir():
        raise RecapError('measurements directory missing: %s' % measurements)
    files = [path for path in measurements.iterdir() if path.name.endswith('.json')]
    if not files:
        raise RecapError('measurements directory empty: %s' % measurements)

    local_ids = []
    for path in files:
        stem = path.name[:-len('.json')]
        try:
            local_ids.append(int(stem))
        except ValueError:
            raise RecapError('measurement name is not an index: %s' % path.name)
    local_ids.sort()
    if local_ids[0] != 0 or local_ids != list(range(local_ids[-1] + 1)):
        raise RecapError('measurement indices are not contiguous from 0 in %s' % measurements)
    count = len(local_ids)

    link_flags = [path.is_symlink() for path in files]
    if any(link_flags) and not all(link_flags):
        raise RecapError('mixed symlink and real measurement files in %s' % measurements)
    if not all(link_flags):
        return 0, count - 1, count

    pairs = []
    for path in files:
        target_name = os.path.basename(os.readlink(str(path)))
        stem, ext = os.path.splitext(target_name)
        if ext != '.json':
            raise RecapError('measurement symlink target is not json: %s' % target_name)
        try:
            original = int(stem)
            local = int(path.name[:-len('.json')])
        except ValueError:
            raise RecapError('measurement symlink target is not an index: %s' % target_name)
        pairs.append((local, original))
    pairs.sort()
    offset = pairs[0][1] - pairs[0][0]
    originals = []
    for local, original in pairs:
        if original - local != offset:
            raise RecapError('measurement remap is not a constant offset in %s' % measurements)
        originals.append(original)
    if originals != list(range(originals[0], originals[-1] + 1)):
        raise RecapError('original measurement indices have gaps in %s' % measurements)
    if originals[-1] - originals[0] + 1 != count:
        raise RecapError('original frame span does not match local count in %s' % measurements)
    return originals[0], originals[-1], count


def actor_data_path(task_dir):
    return Path(task_dir) / 'carla_actor' / 'carla_actor.data'


def load_actor_records(path):
    """Static NPC list saved at collection time.

    The collection seed is not on disk. Each record is the blueprint id and
    pose that ``init_static_npc`` actually spawned.
    """
    path = Path(path)
    try:
        with path.open('rb') as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise RecapError('cannot read carla_actor.data %s: %s' % (path, exc))
    if not isinstance(payload, list) or not payload:
        raise RecapError('carla_actor.data is empty or not a list: %s' % path)
    records = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RecapError('actor record %d is not a dict in %s' % (index, path))
        missing = [key for key in ACTOR_RECORD_KEYS if key not in item]
        if missing:
            raise RecapError('actor record %d missing %s in %s' % (index, ','.join(missing), path))
        try:
            records.append({
                'type': str(item['type']),
                'x': float(item['x']),
                'y': float(item['y']),
                'yaw': float(item['yaw']),
            })
        except (TypeError, ValueError) as exc:
            raise RecapError('actor record %d has a bad pose in %s: %s' % (index, path, exc))
    return records


def read_saved_poses(task_dir, frame_start, frame_end):
    """Map original saved-frame index to the ego pose in measurements."""
    measurements = Path(task_dir) / 'measurements'
    poses = {}
    for saved in range(frame_start, frame_end + 1):
        local = saved - frame_start
        path = measurements / ('%04d.json' % local)
        if not path.is_file():
            raise RecapError('measurement missing for saved frame %d: %s' % (saved, path))
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RecapError('cannot read measurement %s: %s' % (path, exc))
        try:
            poses[saved] = {key: float(data[key]) for key in POSE_KEYS}
        except (KeyError, TypeError, ValueError) as exc:
            raise RecapError('measurement %s missing pose field: %s' % (path, exc))
    return poses


def pose_for_step(poses, frame_start, frame_end, step, save_frequency):
    """Hold the saved-frame pose across the sim steps that belong to it.

    Steps before the slice keep the first stored pose. The slice does not
    contain earlier ego motion.
    """
    saved = step // save_frequency
    if saved < frame_start:
        saved = frame_start
    elif saved > frame_end:
        saved = frame_end
    try:
        return poses[saved]
    except KeyError:
        raise RecapError('no pose for saved frame %d' % saved)


def parse_mountinfo(text):
    """Return (mountpoint, host_root) pairs from /proc/self/mountinfo text.

    A Docker bind of ``/home/USER/workspace`` onto ``/workspace`` shows up as
    host root ``/home/USER/workspace`` and mountpoint ``/workspace``. CARLA
    Server writes recorder files on the host, so client paths must be rewritten.
    """
    maps = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        host_root = fields[3].replace('\\040', ' ')
        mountpoint = fields[4].replace('\\040', ' ')
        if not (host_root.startswith('/') and mountpoint.startswith('/')):
            continue
        if host_root.rstrip('/') == mountpoint.rstrip('/'):
            continue
        maps.append((mountpoint.rstrip('/') or '/', host_root.rstrip('/') or '/'))
    maps.sort(key=lambda item: len(item[0]), reverse=True)
    return maps


def load_server_path_maps(explicit=None, mountinfo_path='/proc/self/mountinfo'):
    """Explicit ``container=host`` entries override auto-detected bind mounts."""
    maps = []
    for item in explicit or []:
        if '=' not in item:
            raise RecapError('--server_path_map must be container=host, got %r' % item)
        container, host = item.split('=', 1)
        container = container.rstrip('/') or '/'
        host = host.rstrip('/') or '/'
        if not container.startswith('/') or not host.startswith('/'):
            raise RecapError('--server_path_map paths must be absolute: %r' % item)
        maps.append((container, host))
    try:
        text = Path(mountinfo_path).read_text(encoding='utf-8')
    except OSError:
        text = ''
    maps.extend(parse_mountinfo(text))
    maps.sort(key=lambda item: len(item[0]), reverse=True)
    return maps


def translate_server_path(path, maps):
    """Rewrite a path the client can see into the path CARLA Server can open."""
    text = str(path)
    for mountpoint, host_root in maps:
        if text == mountpoint or text.startswith(mountpoint + '/'):
            return host_root + text[len(mountpoint):]
    return text


def prepare_replay_log_path(task_dir, output_root, split_name, episode, task_name):
    """Return a real ``recoding.log`` path that does not follow a replay symlink.

    The enhanced dataset points ``replay`` at the original collection tree.
    A synthesized log is written beside the task (or under ``output_root``),
    and the original target is left in place.
    """
    if output_root is not None:
        directory = Path(output_root) / split_name / episode / task_name / 'replay'
    else:
        directory = Path(task_dir) / 'replay'
        if directory.is_symlink():
            directory.unlink()
        elif directory.exists() and not directory.is_dir():
            directory.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise RecapError('replay output is not a real directory: %s' % directory)
    # The directory is often created as root inside Docker (mode 755). CARLA
    # Server runs as the host user and must be able to create recoding.log.
    try:
        os.chmod(str(directory), 0o777)
    except OSError as exc:
        raise RecapError('cannot chmod replay directory %s: %s' % (directory, exc))
    log_path = directory / 'recoding.log'
    if log_path.is_symlink() or log_path.exists():
        log_path.unlink()
    return log_path


def _episode_selected(episode, episode_filters):
    if not episode_filters:
        return True
    for item in episode_filters:
        if episode == item or item in episode:
            return True
    return False


def _task_selected(task_name, task_filters):
    if not task_filters:
        return True
    return task_name in task_filters


def discover_tasks(split_dirs, episode_filters=None, task_filters=None):
    jobs = []
    for split_dir in split_dirs:
        split_dir = Path(split_dir)
        if not split_dir.is_dir():
            raise RecapError('split directory does not exist: %s' % split_dir)
        episodes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        for episode in episodes:
            if not _episode_selected(episode, episode_filters):
                continue
            episode_dir = split_dir / episode
            task_names = sorted(
                (path.name for path in episode_dir.iterdir()
                 if path.is_dir() and path.name.startswith('task')),
                key=_task_sort_key)
            for task_name in task_names:
                if not _task_selected(task_name, task_filters):
                    continue
                task_dir = episode_dir / task_name
                log_path = task_dir / 'replay' / 'recoding.log'
                status, size = classify_log(log_path)
                actor_path = actor_data_path(task_dir)
                has_actor = actor_path.is_file() and actor_path.stat().st_size > 0
                if status != 'ok':
                    if not has_actor:
                        jobs.append(TaskJob(
                            split_dir, episode, task_name, task_dir, status,
                            '%s (%d bytes)' % (status, size),
                            log_path=log_path if log_path.is_file() else None,
                            log_bytes=size))
                        continue
                    try:
                        frame_start, frame_end, local_count = read_frame_range(task_dir)
                    except RecapError as exc:
                        jobs.append(TaskJob(
                            split_dir, episode, task_name, task_dir, 'bad_index',
                            str(exc),
                            log_path=log_path if log_path.is_file() else None,
                            log_bytes=size,
                            actor_path=actor_path.resolve(),
                            actor_bytes=actor_path.stat().st_size))
                        continue
                    jobs.append(TaskJob(
                        split_dir, episode, task_name, task_dir, 'resynth', '',
                        frame_start=frame_start, frame_end=frame_end,
                        local_count=local_count,
                        actor_path=actor_path.resolve(),
                        actor_bytes=actor_path.stat().st_size,
                        resynthesized=True))
                    continue
                try:
                    frame_start, frame_end, local_count = read_frame_range(task_dir)
                except RecapError as exc:
                    jobs.append(TaskJob(
                        split_dir, episode, task_name, task_dir, 'bad_index',
                        str(exc), log_path=log_path.resolve(), log_bytes=size))
                    continue
                jobs.append(TaskJob(
                    split_dir, episode, task_name, task_dir, 'runnable', '',
                    log_path=log_path.resolve(), log_bytes=size,
                    frame_start=frame_start, frame_end=frame_end,
                    local_count=local_count))
    return jobs


def discover_failed_retries(output_root, split_dirs, episode_filters=None, task_filters=None):
    """Build runnable jobs from failed ``recap_manifest.json`` under ``output_root``.

    Measurements and frame range still come from the matching task under ``split_dirs``.
    The recorder log path is taken from the manifest when present.
    """
    if output_root is None:
        raise RecapError('--retry_failed requires --output_root')
    output_root = Path(output_root)
    if not output_root.is_dir():
        raise RecapError('output_root is not a directory: %s' % output_root)
    split_dirs = [Path(item) for item in split_dirs]
    split_by_name = {}
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            raise RecapError('split directory does not exist: %s' % split_dir)
        split_by_name[split_dir.name] = split_dir

    jobs = []
    for manifest_path in sorted(output_root.rglob(MANIFEST_NAME)):
        try:
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            logging.debug('skip unreadable manifest %s: %s', manifest_path, exc)
            continue
        if payload.get('status') != 'failed':
            continue
        task_mirror = manifest_path.parent
        task_name = task_mirror.name
        episode = task_mirror.parent.name
        split_name = task_mirror.parent.parent.name
        split_dir = split_by_name.get(split_name)
        if split_dir is None:
            continue
        if not _episode_selected(episode, episode_filters):
            continue
        if not _task_selected(task_name, task_filters):
            continue
        task_dir = split_dir / episode / task_name
        if not task_dir.is_dir():
            jobs.append(TaskJob(
                split_dir, episode, task_name, task_dir, 'bad_index',
                'task missing under split: %s' % task_dir))
            continue
        try:
            frame_start, frame_end, local_count = read_frame_range(task_dir)
        except RecapError as exc:
            jobs.append(TaskJob(
                split_dir, episode, task_name, task_dir, 'bad_index', str(exc)))
            continue
        manifest_range = payload.get('frame_range_original')
        if (isinstance(manifest_range, list) and len(manifest_range) == 2
                and payload.get('local_count') is not None):
            frame_start = int(manifest_range[0])
            frame_end = int(manifest_range[1])
            local_count = int(payload['local_count'])
        log_text = payload.get('log') or ''
        log_path = Path(log_text) if log_text else (task_dir / 'replay' / 'recoding.log')
        if not log_path.is_file():
            log_path = task_dir / 'replay' / 'recoding.log'
        status, size = ('no_log', 0)
        if log_path.is_file():
            status, size = classify_log(log_path)
        actor_path = actor_data_path(task_dir)
        has_actor = actor_path.is_file() and actor_path.stat().st_size > 0
        if status != 'ok':
            if not has_actor:
                jobs.append(TaskJob(
                    split_dir, episode, task_name, task_dir, status,
                    '%s (%d bytes)' % (status, size),
                    log_path=log_path if log_path.is_file() else None,
                    log_bytes=size,
                    frame_start=frame_start, frame_end=frame_end,
                    local_count=local_count))
                continue
            jobs.append(TaskJob(
                split_dir, episode, task_name, task_dir, 'resynth', '',
                frame_start=frame_start, frame_end=frame_end,
                local_count=local_count,
                actor_path=actor_path.resolve(),
                actor_bytes=actor_path.stat().st_size,
                resynthesized=True))
            continue
        jobs.append(TaskJob(
            split_dir, episode, task_name, task_dir, 'runnable', '',
            log_path=log_path.resolve(), log_bytes=size,
            frame_start=frame_start, frame_end=frame_end,
            local_count=local_count))
    return jobs


def summarize_jobs(jobs):
    counts = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    parts = ['%s=%d' % (key, counts[key]) for key in sorted(counts)]
    return 'SUMMARY ' + ' '.join(parts)


def sensor_output_dir(job, spec, output_root):
    if output_root is None:
        return job.task_dir / spec.output_dir
    return (Path(output_root) / job.split_name / job.episode / job.task_name / spec.output_dir)


def manifest_path(job, output_root):
    if output_root is None:
        return job.task_dir / MANIFEST_NAME
    return Path(output_root) / job.split_name / job.episode / job.task_name / MANIFEST_NAME


def run_log_path(job, output_root):
    if output_root is None:
        return job.split_dir / 'recap_run.log'
    return Path(output_root) / job.split_name / 'recap_run.log'


def _real_named_files(directory, suffix):
    if not directory.is_dir() or directory.is_symlink():
        return []
    names = []
    for child in directory.iterdir():
        if child.is_symlink() or not child.is_file():
            continue
        if child.name.endswith(suffix):
            names.append(child.name)
    return names


def dir_is_complete(directory, spec, count):
    expected = set(spec.filename(index) for index in range(count))
    return set(_real_named_files(directory, spec.suffix)) == expected


def dir_has_real_files(directory):
    if directory.is_symlink() or (not directory.is_dir()):
        return False
    for child in directory.iterdir():
        if child.is_file() and not child.is_symlink():
            return True
    return False


def _manifest_failed_for(path, groups):
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return payload.get('status') == 'failed' and payload.get('sensors') == list(groups)


def task_output_decision(job, specs, groups, output_root, overwrite):
    """Return 'skip_complete', 'skip_partial', or 'write'."""
    if output_root is not None:
        complete = all(
            dir_is_complete(sensor_output_dir(job, spec, output_root), spec, job.local_count)
            for spec in specs)
        if complete and not overwrite:
            return 'skip_complete', ''
        return 'write', ''

    complete = all(
        dir_is_complete(sensor_output_dir(job, spec, None), spec, job.local_count)
        for spec in specs)
    if complete and not overwrite:
        return 'skip_complete', ''

    allow_replace = overwrite or _manifest_failed_for(manifest_path(job, None), groups)
    if allow_replace:
        return 'write', ''

    blocked = []
    for spec in specs:
        directory = sensor_output_dir(job, spec, None)
        if dir_has_real_files(directory) and not dir_is_complete(directory, spec, job.local_count):
            blocked.append(spec.output_dir)
    if blocked:
        return 'skip_partial', 'real files exist in %s; pass --overwrite' % ','.join(blocked)
    return 'write', ''


def prepare_output_dir(path):
    """Create a real directory for new frames.

    A directory symlink is unlinked and not followed, so the old target tree
    stays in place. Symlink files inside a real directory are unlinked.
    """
    path = Path(path)
    if path.is_symlink():
        path.unlink()
    if path.exists():
        if not path.is_dir():
            path.unlink()
        else:
            for child in list(path.iterdir()):
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    raise RecapError('unexpected subdirectory in sensor output: %s' % child)
    if not path.exists():
        path.mkdir(parents=True)
    if path.is_symlink() or not path.is_dir():
        raise RecapError('sensor output is not a real directory: %s' % path)


def _write_manifest(path, job, groups, target_fps, save_frequency, status, error=None):
    payload = {
        'log': str(job.log_path) if job.log_path is not None else '',
        'log_bytes': job.log_bytes,
        'source': 'carla_actor.data' if job.resynthesized else 'recoding.log',
        'sensors': list(groups),
        'save_frequency': save_frequency,
        'target_fps': target_fps,
        'frame_range_original': [job.frame_start, job.frame_end],
        'local_count': job.local_count,
        'status': status,
    }
    if error:
        payload['error'] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _append_run_log(path, job, status, groups, detail):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
        datetime.now().isoformat(timespec='seconds'),
        job.split_name, job.episode, job.task_name, status,
        ','.join(groups), detail)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(line)


def _apply_attributes(blueprint, attributes):
    for key, value in attributes.items():
        if not blueprint.has_attribute(key):
            raise RecapError('blueprint %s has no attribute %s' % (blueprint.id, key))
        blueprint.set_attribute(key, str(value))


def _flush_queue(sensor_queue):
    while True:
        try:
            sensor_queue.get_nowait()
        except Empty:
            return


def _drain_sensors(sensor_queue, expected_ids, timeout_sec):
    found = {}
    deadline = time.time() + timeout_sec
    while len(found) < len(expected_ids):
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            data, name = sensor_queue.get(timeout=remaining)
        except Empty:
            break
        found[name] = data
    return found


def ego_xy_distance2(pose, x, y):
    dx = float(pose['x']) - float(x)
    dy = float(pose['y']) - float(y)
    return dx * dx + dy * dy


def select_ego_index(candidates, expected_pose, max_dist_m=5.0):
    """Pick one index from ``candidates`` as ego using measurement (x, y).

    Each candidate is (x, y). Returns the index with smallest planar distance.
    Raises RecapError when the best match is farther than ``max_dist_m``.
    """
    if not candidates:
        raise RecapError('no ego candidates')
    if expected_pose is None:
        if len(candidates) == 1:
            return 0
        raise RecapError('ambiguous ego: %d vehicle.tesla.model3 actors' % len(candidates))
    best_index = min(
        range(len(candidates)),
        key=lambda index: ego_xy_distance2(expected_pose, candidates[index][0], candidates[index][1]))
    best_dist = ego_xy_distance2(
        expected_pose, candidates[best_index][0], candidates[best_index][1]) ** 0.5
    if best_dist > max_dist_m:
        raise RecapError(
            'no model3 near measurement pose (best dist %.2fm, limit %.1fm)'
            % (best_dist, max_dist_m))
    return best_index


def ego_pose_for_attach(poses, frame_start, frame_end, step, save_frequency):
    """Measurement pose for ego picking, or None until the pre-attach step.

    Sensors spawned at the end of a step capture on the next tick. The last
    step before saved frame ``frame_start`` is ``frame_start * save_frequency - 1``.
    Matching earlier than that uses a parked pose that fits a leftover model3
    better than the log ego, which is still at the start of the recording.
    """
    first_attach_step = frame_start * save_frequency - 1
    if first_attach_step < 0:
        first_attach_step = 0
    if step < first_attach_step:
        return None
    saved = step // save_frequency
    if saved < frame_start:
        saved = frame_start
    elif saved > frame_end:
        saved = frame_end
    try:
        return poses[saved]
    except KeyError:
        raise RecapError('no pose for saved frame %d' % saved)


def _select_ego(world, expected_pose=None):
    """Return the log ego, or None to keep waiting.

    A pose is required. One model3 is that ego even when the replayer has
    left it off the measurement; the caller snaps it onto the pose. Several
    model3s are disambiguated by the measurement, and None means none of them
    is close enough yet. ``role_name=hero`` is not a shortcut: resynth marks
    the ego as hero, human logs do not, and a previous hero can outlive its task.
    """
    actors = list(world.get_actors().filter(EGO_BLUEPRINT))
    if not actors or expected_pose is None:
        return None
    if len(actors) == 1:
        return actors[0]
    candidates = [
        (actor.get_transform().location.x, actor.get_transform().location.y)
        for actor in actors]
    try:
        index = select_ego_index(candidates, expected_pose)
    except RecapError:
        return None
    return actors[index]


def parse_only_pairs(items):
    """Parse repeatable ``episode/task`` selectors."""
    pairs = []
    for item in items or []:
        if '/' not in item or item.startswith('/'):
            raise RecapError('--only must be episode/task, got %r' % item)
        episode, task_name = item.split('/', 1)
        if (not episode) or (not task_name) or ('/' in task_name):
            raise RecapError('--only must be episode/task, got %r' % item)
        pairs.append((episode, task_name))
    return pairs


def filter_jobs_by_only(jobs, pairs):
    """Keep jobs named by ``pairs``. Every pair must match something."""
    if not pairs:
        return jobs
    wanted = set(pairs)
    selected = [job for job in jobs if (job.episode, job.task_name) in wanted]
    found = set((job.episode, job.task_name) for job in selected)
    missing = wanted - found
    if missing:
        rendered = ', '.join('%s/%s' % pair for pair in sorted(missing))
        raise RecapError('--only did not match: %s' % rendered)
    return selected


def _attach_sensors(world, hero, specs, sensor_queue):
    import carla

    actors = []
    library = world.get_blueprint_library()
    for spec in specs:
        blueprint = library.find(spec.blueprint)
        _apply_attributes(blueprint, spec.attributes)
        transform = carla.Transform(
            carla.Location(
                x=float(spec.transform['x']),
                y=float(spec.transform['y']),
                z=float(spec.transform['z'])),
            carla.Rotation(
                pitch=float(spec.transform['pitch']),
                roll=float(spec.transform['roll']),
                yaw=float(spec.transform['yaw'])))
        actor = world.spawn_actor(
            blueprint, transform, attach_to=hero,
            attachment_type=carla.AttachmentType.Rigid)
        if actor is None:
            raise RecapError('failed to spawn %s' % spec.sensor_id)
        sensor_id = spec.sensor_id
        actor.listen(lambda data, name=sensor_id, q=sensor_queue: q.put((data, name)))
        actors.append(actor)
    return actors


def _destroy_sensors(actors):
    for actor in actors:
        try:
            actor.stop()
        except Exception:
            logging.debug('sensor stop failed', exc_info=True)
        try:
            actor.destroy()
        except Exception:
            logging.debug('sensor destroy failed', exc_info=True)


def _snap_ego_to_pose(hero, pose, step):
    """Put ``hero`` on the measurement pose and fail if it does not stay there."""
    import carla

    hero.set_simulate_physics(False)
    hero.set_transform(carla.Transform(
        carla.Location(x=float(pose['x']), y=float(pose['y']), z=float(pose['z'])),
        carla.Rotation(
            pitch=float(pose['pitch']), yaw=float(pose['yaw']), roll=float(pose['roll']))))
    location = hero.get_transform().location
    error_m = ego_xy_distance2(pose, location.x, location.y) ** 0.5
    if error_m > EGO_SNAP_MAX_ERROR_M:
        raise RecapError(
            'ego did not reach measurement pose at step %d (error %.2fm)'
            % (step, error_m))


def _release_replayer_keep_actors(session):
    """Stop replay writes and leave the spawned vehicles in the world.

    Replay applies positions inside ``tick``. Once the ego is bound, measurements
    drive it; a later replay packet must not move it back. Parked vehicles stay
    at the pose they had when replay stopped.
    """
    stop = getattr(session.client, 'stop_replayer', None)
    if not callable(stop):
        return
    try:
        stop(True)
    except TypeError:
        logging.warning('stop_replayer(keep_actors) unavailable; calling stop_replayer()')
        try:
            stop()
        except Exception:
            logging.warning('stop_replayer failed', exc_info=True)
            return
    except Exception:
        logging.warning('stop_replayer(keep_actors=True) failed', exc_info=True)
        return
    try:
        session.world.tick()
    except Exception:
        logging.debug('tick after releasing replayer failed', exc_info=True)
    _disable_vehicle_physics(session.world)


def _first_safe_saved_index(attach_step, save_frequency):
    """Smallest saved-frame index whose tick is strictly after the attach tick.

    Sensors spawned at the end of `attach_step` produce their first sample on
    the next world tick. Saved index S is world step S * save_frequency.
    """
    first_data_step = attach_step + 1
    return (first_data_step + save_frequency - 1) // save_frequency


class RecapSession(object):
    def __init__(self, host, port, map_name, target_fps, server_path_maps=None):
        self.host = host
        self.port = port
        self.map_name = map_name
        self.target_fps = target_fps
        self.server_path_maps = list(server_path_maps or [])
        self.client = None
        self.world = None
        self._original_settings = None

    def to_server_path(self, path):
        return translate_server_path(path, self.server_path_maps)

    def connect(self):
        import carla

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(60.0)
        logging.info('loading map %s', self.map_name)
        self.world = self.client.load_world(self.map_name)
        try:
            self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        except Exception as exc:
            logging.warning('unload ParkedVehicles skipped: %s', exc)
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.target_fps)
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.world.tick()

    def close(self):
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except Exception:
                logging.debug('failed to restore world settings', exc_info=True)
        self.world = None
        self.client = None

    def stop_replay(self):
        if self.client is None:
            return
        stop = getattr(self.client, 'stop_replayer', None)
        if not callable(stop):
            return
        try:
            stop()
        except Exception:
            logging.debug('stop_replayer failed', exc_info=True)
            return
        try:
            self.world.tick()
        except Exception:
            logging.debug('tick after stop_replayer failed', exc_info=True)


def _clear_vehicles_and_sensors(world):
    sensors = list(world.get_actors().filter('sensor.*'))
    vehicles = list(world.get_actors().filter('vehicle.*'))
    for actor in sensors:
        try:
            actor.stop()
        except Exception:
            logging.debug('sensor stop failed', exc_info=True)
        try:
            actor.destroy()
        except Exception:
            logging.debug('sensor destroy failed', exc_info=True)
    for actor in vehicles:
        try:
            actor.destroy()
        except Exception:
            logging.debug('vehicle destroy failed', exc_info=True)
    try:
        world.tick()
    except Exception:
        logging.debug('tick after actor clear failed', exc_info=True)


def _disable_vehicle_physics(world):
    """Force every vehicle kinematic so the replayer pose is not integrated.

    The recorder stores transforms. With physics left on, CARLA still steps the
    vehicle after applying that pose and cars leave the ground. RGB from the
    original log then no longer matches recap depth. Physics is turned off on
    every tick because newly spawned actors default to simulated, and a log
    can turn the ego's physics back on.
    """
    try:
        vehicles = world.get_actors().filter('vehicle.*')
    except Exception:
        logging.debug('vehicle list failed', exc_info=True)
        return
    for actor in vehicles:
        try:
            actor.set_simulate_physics(False)
        except Exception:
            logging.debug('disable physics failed for actor %s', getattr(actor, 'id', '?'),
                          exc_info=True)


def _try_lifted_spawn(world, blueprint, x, y, z, pitch, yaw, roll):
    """Spawn at (x,y,z) with physics off; return None if every attempt overlaps."""
    import carla

    rotation = carla.Rotation(pitch=float(pitch), yaw=float(yaw), roll=float(roll))
    exact = carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z)), rotation)
    for lift in (0.0, 2.0, 5.0, 10.0):
        attempt = exact if lift == 0.0 else carla.Transform(
            carla.Location(x=float(x), y=float(y), z=float(z) + lift), rotation)
        actor = world.try_spawn_actor(blueprint, attempt)
        if actor is None:
            continue
        actor.set_simulate_physics(False)
        actor.set_transform(exact)
        return actor
    return None


def _lifted_spawn(world, blueprint, x, y, z, pitch, yaw, roll):
    actor = _try_lifted_spawn(world, blueprint, x, y, z, pitch, yaw, roll)
    if actor is None:
        raise RecapError(
            'failed to spawn %s at (%.2f, %.2f, %.2f)' % (blueprint.id, x, y, z))
    return actor


def _find_blueprint(library, type_id):
    try:
        return library.find(type_id)
    except RuntimeError as exc:
        raise RecapError('blueprint %s is not in this CARLA server: %s' % (type_id, exc))


def synthesize_recorder_log(session, job, output_root, save_frequency):
    """Spawn the saved static scene, drive the ego, and write a new recorder log.

    Actors are spawned after ``start_recorder`` so the log contains the spawn
    events. Ego pose is the measurement at each saved frame; steps before the
    slice hold that first pose so replay frame indices stay aligned.
    """
    import carla

    if job.actor_path is None:
        raise RecapError('resynth task has no carla_actor.data')
    records = load_actor_records(job.actor_path)
    poses = read_saved_poses(job.task_dir, job.frame_start, job.frame_end)
    log_path = prepare_replay_log_path(
        job.task_dir, output_root, job.split_name, job.episode, job.task_name)
    session.stop_replay()
    _clear_vehicles_and_sensors(session.world)

    recording = False
    recorder_status = ''
    server_log = session.to_server_path(log_path.resolve())
    try:
        recorder_status = session.client.start_recorder(server_log)
        recording = True
        if recorder_status and 'error' in str(recorder_status).lower():
            raise RecapError('start_recorder failed: %s (server path %s)' % (
                recorder_status, server_log))
        library = session.world.get_blueprint_library()
        ego_bp = _find_blueprint(library, EGO_BLUEPRINT)
        if ego_bp.has_attribute('role_name'):
            ego_bp.set_attribute('role_name', 'hero')
        first = poses[job.frame_start]
        # Ego before static NPCs: actor.data can include a vehicle on the slot
        # where measurements place the ego (collection skips that spawn point).
        ego = _lifted_spawn(
            session.world, ego_bp, first['x'], first['y'], first['z'],
            first['pitch'], first['yaw'], first['roll'])
        static_spawned = 0
        for record in records:
            blueprint = _find_blueprint(library, record['type'])
            if _try_lifted_spawn(
                    session.world, blueprint, record['x'], record['y'], STATIC_NPC_Z,
                    0.0, record['yaw'], 0.0) is None:
                logging.warning(
                    'skip static %s at (%.2f, %.2f) for %s/%s/%s',
                    record['type'], record['x'], record['y'],
                    job.split_name, job.episode, job.task_name)
                continue
            static_spawned += 1
        _disable_vehicle_physics(session.world)
        last_step = job.frame_end * save_frequency
        logging.info(
            'record %s/%s/%s static=%d/%d frames %d-%d -> %s (server %s) recorder=%s',
            job.split_name, job.episode, job.task_name, static_spawned, len(records),
            job.frame_start, job.frame_end, log_path, server_log, recorder_status)
        for step in range(0, last_step + 1):
            pose = pose_for_step(
                poses, job.frame_start, job.frame_end, step, save_frequency)
            ego.set_transform(carla.Transform(
                carla.Location(x=pose['x'], y=pose['y'], z=pose['z']),
                carla.Rotation(pitch=pose['pitch'], yaw=pose['yaw'], roll=pose['roll'])))
            _disable_vehicle_physics(session.world)
            session.world.tick()
    finally:
        if recording:
            try:
                session.client.stop_recorder()
            except Exception:
                logging.debug('stop_recorder failed', exc_info=True)
        _clear_vehicles_and_sensors(session.world)

    status, size = classify_log(log_path)
    if status != 'ok':
        raise RecapError(
            'synthesized recorder log is %s (%d bytes): %s (CARLA server path %s, '
            'start_recorder returned %r)' % (status, size, log_path, server_log, recorder_status))
    job.log_path = log_path.resolve()
    job.log_bytes = size
    job.status = 'runnable'
    logging.info('synthesized log %s (%d bytes)', job.log_path, size)
    return job.log_path


def replay_task(session, job, specs, groups, output_root, target_fps, save_frequency,
                ego_timeout_ticks):
    """Replay one log and write `specs` into the task's local frame range."""
    expected_ids = set(spec.sensor_id for spec in specs)
    sensor_queue = Queue()
    sensors = []
    hero = None
    prepared = False
    written = 0
    missed_in_slice = 0
    last_step = job.frame_end * save_frequency
    poses = read_saved_poses(job.task_dir, job.frame_start, job.frame_end)

    try:
        logging.info(
            'replay %s/%s/%s log=%s frames %d-%d',
            job.split_name, job.episode, job.task_name, job.log_path,
            job.frame_start, job.frame_end)
        # Drop the previous task's hero before the log spawns its own model3.
        # Human recordings do not set role_name=hero, so a leftover resynth
        # hero would otherwise win ego selection and the depth camera would
        # stay parked for the whole task.
        session.stop_replay()
        _clear_vehicles_and_sensors(session.world)
        server_log = session.to_server_path(job.log_path)
        logging.info('replay file on server %s', server_log)
        replay_status = session.client.replay_file(server_log, 0.0, 0.0, 0)
        logging.info('replay_file returned %s', replay_status)
        set_factor = getattr(session.client, 'set_replayer_time_factor', None)
        if callable(set_factor):
            set_factor(1.0)
        ignore_hero = getattr(session.client, 'set_replayer_ignore_hero', None)
        if callable(ignore_hero):
            ignore_hero(False)

        for step in range(0, last_step + 1):
            # Snap before tick so the depth camera sees the measurement pose.
            # Replay is released once the ego is bound; until then physics stays
            # off around the tick so newly spawned actors do not integrate.
            if hero is not None:
                _snap_ego_to_pose(
                    hero,
                    pose_for_step(
                        poses, job.frame_start, job.frame_end, step, save_frequency),
                    step)
            _disable_vehicle_physics(session.world)
            session.world.tick()
            _disable_vehicle_physics(session.world)
            if hero is None:
                expected_pose = ego_pose_for_attach(
                    poses, job.frame_start, job.frame_end, step, save_frequency)
                if expected_pose is None:
                    continue
                hero = _select_ego(session.world, expected_pose)
                if hero is None:
                    missed_in_slice += 1
                    if missed_in_slice >= ego_timeout_ticks:
                        present = len(list(session.world.get_actors().filter(EGO_BLUEPRINT)))
                        raise RecapError(
                            'no usable vehicle.tesla.model3 after %d ticks in the slice '
                            '(model3=%d, frame_start=%d)'
                            % (ego_timeout_ticks, present, job.frame_start))
                    continue
                logging.info(
                    'bound ego at step %d; driving it from measurements', step)
                _release_replayer_keep_actors(session)
                _snap_ego_to_pose(hero, expected_pose, step)
                sensors = _attach_sensors(session.world, hero, specs, sensor_queue)
                _flush_queue(sensor_queue)
                attach_step = step
                first_safe = _first_safe_saved_index(attach_step, save_frequency)
                if job.frame_start < first_safe:
                    raise RecapError(
                        'frame_start %d is before sensors produce data '
                        '(first safe saved frame %d, ego attached on step %d)'
                        % (job.frame_start, first_safe, attach_step))
                continue

            need_frame = (step % save_frequency) == 0
            payload = _drain_sensors(sensor_queue, expected_ids, SENSOR_TIMEOUT_SEC)
            if set(payload) != expected_ids:
                raise RecapError(
                    'incomplete sensors at step %d, got %s expected %s'
                    % (step, sorted(payload), sorted(expected_ids)))
            if not need_frame:
                continue

            saved_index = step // save_frequency
            if saved_index < job.frame_start or saved_index > job.frame_end:
                continue
            local_index = saved_index - job.frame_start
            if not prepared:
                for spec in specs:
                    prepare_output_dir(sensor_output_dir(job, spec, output_root))
                prepared = True
            for spec in specs:
                destination = sensor_output_dir(job, spec, output_root) / spec.filename(local_index)
                payload[spec.sensor_id].save_to_disk(str(destination))
                if (not destination.is_file()) or destination.stat().st_size <= 0:
                    raise RecapError('empty sensor file %s' % destination)
            written += 1

        if written != job.local_count:
            raise RecapError('wrote %d frames, expected %d' % (written, job.local_count))
        _write_manifest(
            manifest_path(job, output_root), job, groups, target_fps, save_frequency, 'ok')
        return written
    except Exception as exc:
        _write_manifest(
            manifest_path(job, output_root), job, groups, target_fps, save_frequency,
            'failed', error=str(exc))
        raise
    finally:
        _destroy_sensors(sensors)
        session.stop_replay()


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be > 0')
    return number


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Batch-recap optional sensors from CARLA recorder logs')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--map', default='Town04_Opt', help='map recorded in the logs')
    parser.add_argument(
        '--splits', nargs='+', required=True,
        help='episode split directories, e.g. .../train_town04 .../val_town04')
    parser.add_argument(
        '--sensors', required=True,
        help='comma-separated sensor groups: depth, lidar, semantic')
    parser.add_argument('--episode', action='append', default=None,
                        help='episode directory name or substring; repeatable')
    parser.add_argument('--task', action='append', default=None,
                        help='task directory name, e.g. task1; repeatable')
    parser.add_argument(
        '--only', action='append', default=None,
        help='episode/task pair to replay; repeatable. '
             'Use this instead of --episode plus --task when the pairs are not a cross product')
    parser.add_argument('--limit', type=_positive_int, default=None,
                        help='replay at most N runnable tasks')
    parser.add_argument('--target_fps', type=_positive_int, default=30)
    parser.add_argument('--save_frequency', type=_positive_int, default=3,
                        help='save one frame every N sim steps (collection default: 3)')
    parser.add_argument('--ego_timeout_ticks', type=_positive_int, default=90)
    parser.add_argument('--output_root', default=None,
                        help='mirror outputs here instead of writing into the task directory')
    parser.add_argument(
        '--server_path_map', action='append', default=None,
        help='container_prefix=host_prefix for paths CARLA Server opens. '
             'Repeatable. Bind mounts in /proc/self/mountinfo are used when omitted. '
             'Example: /workspace=/home/USER/workspace')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace sensor dirs that already contain real files')
    parser.add_argument('--dry_run', action='store_true',
                        help='classify tasks and print frame ranges; do not connect to CARLA')
    parser.add_argument(
        '--retry_failed', action='store_true',
        help='only replay tasks whose recap_manifest.json under --output_root has status failed')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        groups = parse_sensor_groups(args.sensors)
    except ValueError as exc:
        logging.error('%s', exc)
        return 2
    specs = expand_groups(groups)

    try:
        if args.retry_failed:
            jobs = discover_failed_retries(
                args.output_root, args.splits, args.episode, args.task)
        else:
            jobs = discover_tasks(args.splits, args.episode, args.task)
        if args.only:
            jobs = filter_jobs_by_only(jobs, parse_only_pairs(args.only))
    except RecapError as exc:
        logging.error('%s', exc)
        return 2

    runnable = [job for job in jobs if job.status in ('runnable', 'resynth')]
    selected = runnable if args.limit is None else runnable[:args.limit]
    if args.dry_run:
        for job in jobs:
            print(job.format_line())
        print(summarize_jobs(jobs))
        if args.limit is not None:
            print('LIMIT running %d of %d runnable/resynth' % (len(selected), len(runnable)))
        return 0

    logging.info(summarize_jobs(jobs))
    if args.limit is not None:
        logging.info('replaying %d of %d runnable/resynth tasks', len(selected), len(runnable))
    if not selected:
        if not jobs:
            logging.info(
                'no tasks matched splits=%r episode=%r task=%r '
                '(check SPLIT_SRC; try --dry_run -v%s)',
                args.splits, args.episode, args.task,
                ' or --retry_failed with --output_root' if args.output_root else '')
        else:
            logging.info('no runnable tasks (%s)', summarize_jobs(jobs))
        return 0

    try:
        server_path_maps = load_server_path_maps(args.server_path_map)
    except RecapError as exc:
        logging.error('%s', exc)
        return 2

    session = RecapSession(
        args.host, args.port, args.map, args.target_fps, server_path_maps)
    failed = 0
    try:
        session.connect()
        for job in selected:
            decision, reason = task_output_decision(
                job, specs, groups, args.output_root, args.overwrite)
            if decision != 'write':
                logging.info('skip %s/%s/%s %s %s',
                             job.split_name, job.episode, job.task_name, decision, reason)
                _append_run_log(
                    run_log_path(job, args.output_root), job, decision, groups, reason)
                continue
            try:
                if job.status == 'resynth':
                    synthesize_recorder_log(
                        session, job, args.output_root, args.save_frequency)
                written = replay_task(
                    session, job, specs, groups, args.output_root,
                    args.target_fps, args.save_frequency, args.ego_timeout_ticks)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failed += 1
                logging.error('failed %s/%s/%s: %s',
                              job.split_name, job.episode, job.task_name, exc)
                _append_run_log(
                    run_log_path(job, args.output_root), job, 'failed', groups, str(exc))
                continue
            logging.info('ok %s/%s/%s frames %d',
                         job.split_name, job.episode, job.task_name, written)
            _append_run_log(
                run_log_path(job, args.output_root), job, 'ok', groups, 'frames=%d' % written)
    except KeyboardInterrupt:
        logging.info('cancelled')
        return 130
    except Exception as exc:
        logging.error('carla session failed: %s', exc)
        return 1
    finally:
        session.close()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
