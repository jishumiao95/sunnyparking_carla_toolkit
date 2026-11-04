#!/usr/bin/env python3
"""Pack recap check images: RGB | depth side-by-side per camera, stacked by view."""

import argparse
import io
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

CAMERAS = ('front', 'left', 'right', 'rear')


def sample_indices(count, num_samples):
    if count <= 0:
        return []
    if count <= num_samples:
        return list(range(count))
    step = (count - 1) / float(num_samples - 1)
    return sorted({int(round(step * index)) for index in range(num_samples)})


def parse_path_maps(items):
    maps = []
    for item in items or []:
        if '=' not in item:
            raise ValueError('path map must be container=host, got: %s' % item)
        container, host = item.split('=', 1)
        maps.append((container.rstrip('/'), host.rstrip('/')))
    return maps


def resolve_file(path, path_maps):
    path = Path(path)
    if path.is_file():
        return path
    if path.is_symlink():
        target = Path(os.readlink(path))
        if not target.is_absolute():
            target = (path.parent / target).resolve()
        candidate = target
        for container, host in path_maps:
            text = str(candidate)
            if text.startswith(container):
                candidate = Path(host + text[len(container):])
                break
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def load_rgb(path, path_maps):
    path = resolve_file(path, path_maps)
    return np.array(Image.open(path).convert('RGB'))


def decode_depth_meters(rgb_uint8):
    """CARLA depth PNG -> meters (0.9.x save_to_disk encoding)."""
    im = rgb_uint8.astype(np.float32)
    norm = (im[:, :, 0] + im[:, :, 1] * 256.0 + im[:, :, 2] * 65536.0) / 16777215.0
    return norm * 1000.0


def depth_to_vis(rgb_uint8, max_m=50.0, sky_m=900.0):
    """Pseudo-color for human review: near=warm, far=cool, sky/infinity=black."""
    depth_m = decode_depth_meters(rgb_uint8)
    valid = depth_m < sky_m
    depth_m = np.clip(depth_m, 0.3, max_m)
    # log scale 0.3..max_m -> 0..1 (1 = near)
    t = 1.0 - (np.log(depth_m) - np.log(0.3)) / (np.log(max_m) - np.log(0.3))
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0.0, 1.0)
    vis = np.stack([r, g, b], axis=-1)
    vis = (vis * 255.0).astype(np.uint8)
    vis[~valid] = (0, 0, 0)
    return vis


def load_depth_panel(path, path_maps, depth_vis):
    raw = load_rgb(path, path_maps)
    if depth_vis:
        return depth_to_vis(raw)
    return raw


def stitch_frame(task_split, task_recap, local_index, path_maps, depth_vis=False):
    """Four rows: front/left/right/rear, each [rgb | depth]."""
    stem = '%04d.png' % local_index
    rows = []
    for cam in CAMERAS:
        rgb = load_rgb(task_split / ('rgb_%s' % cam) / stem, path_maps)
        depth = load_depth_panel(
            task_recap / ('depth_%s' % cam) / stem, path_maps, depth_vis)
        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError('shape mismatch %s local %d' % (cam, local_index))
        rows.append(np.concatenate([rgb, depth], axis=1))
    return np.concatenate(rows, axis=0)


def list_ok_tasks(recap_root, split_name):
    root = Path(recap_root) / split_name
    tasks = []
    for manifest in sorted(root.glob('*/*/recap_manifest.json')):
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if payload.get('status') != 'ok':
            continue
        episode = manifest.parent.parent.name
        task_name = manifest.parent.name
        tasks.append((episode, task_name, payload))
    return tasks


def load_task_list_file(path):
    """One ``episode/taskN`` per line; ``#`` starts a comment."""
    pairs = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '/' not in line:
            raise ValueError('task list line must be episode/task, got: %r' % line)
        episode, task_name = line.split('/', 1)
        pairs.append((episode.strip(), task_name.strip()))
    return pairs


def tasks_retried_after_failure(recap_run_log):
    """Episode/task pairs that had at least one ``failed`` row and end in ``ok``."""
    history = {}
    for line in Path(recap_run_log).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        episode, task_name, status = parts[2], parts[3], parts[4]
        key = (episode, task_name)
        history.setdefault(key, []).append(status)
    return sorted(
        key for key, statuses in history.items()
        if 'failed' in statuses and statuses[-1] == 'ok')


def select_tasks(pool, task_list_file=None, recap_run_log=None, num_tasks=20, seed=20260412):
    if task_list_file and recap_run_log:
        raise ValueError('use only one of task_list_file and recap_run_log')
    by_key = {(episode, task_name): (episode, task_name, payload)
              for episode, task_name, payload in pool}
    if task_list_file:
        wanted = load_task_list_file(task_list_file)
    elif recap_run_log:
        wanted = tasks_retried_after_failure(recap_run_log)
    else:
        random.seed(seed)
        return random.sample(pool, min(num_tasks, len(pool)))
    pick = []
    missing = []
    for episode, task_name in wanted:
        key = (episode, task_name)
        if key not in by_key:
            missing.append('%s/%s' % (episode, task_name))
            continue
        pick.append(by_key[key])
    if missing:
        print('WARNING: %d listed tasks not ok or missing manifest (first 10):' % len(missing))
        for item in missing[:10]:
            print(' ', item)
    if not pick:
        raise SystemExit('no tasks selected')
    return pick


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--split_name', default='val_town04')
    parser.add_argument('--zip', required=True)
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20260412)
    parser.add_argument('--group', default='check')
    parser.add_argument(
        '--path_map', action='append', default=None,
        help='rewrite symlink targets, e.g. /workspace=/home/USER/workspace')
    parser.add_argument(
        '--depth_vis', action='store_true',
        help='decode depth PNG to pseudo-color (near=red, far=blue, sky=black)')
    parser.add_argument(
        '--task_list', default=None,
        help='text file: one episode/task per line (only those tasks)')
    parser.add_argument(
        '--recap_run_log', default=None,
        help='recap_run.log: pack tasks that failed at least once then finished ok')
    args = parser.parse_args()
    path_maps = parse_path_maps(args.path_map)

    split = Path(args.split)
    recap_root = Path(args.output_root) / args.split_name
    pool = list_ok_tasks(args.output_root, args.split_name)
    if not pool:
        raise SystemExit('no ok tasks under %s' % recap_root)
    pick = select_tasks(
        pool, task_list_file=args.task_list, recap_run_log=args.recap_run_log,
        num_tasks=args.num_tasks, seed=args.seed)

    missing = []
    depth_note = (
        'depth pseudo-color (decode CARLA RGB-depth; near=warm, far=cool, sky=black)'
        if args.depth_vis else 'depth (raw CARLA PNG encoding)')
    readme = [
        'Each PNG: 4 rows (front, left, right, rear).',
        'Each row: RGB (split) | %s, same local frame index.' % depth_note,
        'Local index matches measurements/ 0000..M-1.',
        '',
    ]

    with zipfile.ZipFile(args.zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for episode, task_name, payload in sorted(pick):
            count = int(payload.get('local_count', 0))
            source = payload.get('source', '?')
            indices = sample_indices(count, args.samples)
            readme.append('%s/%s/%s  frames=%d  source=%s  samples=%s' % (
                args.group, episode, task_name, count, source, indices))
            task_split = split / episode / task_name
            task_recap = recap_root / episode / task_name
            for local in indices:
                arc = '%s/%s/%s/local_%04d_rgb_depth.png' % (
                    args.group, episode, task_name, local)
                try:
                    canvas = stitch_frame(
                        task_split, task_recap, local, path_maps, args.depth_vis)
                except FileNotFoundError as exc:
                    missing.append(str(exc))
                    continue
                buf = io.BytesIO()
                Image.fromarray(canvas).save(buf, format='PNG')
                zf.writestr(arc, buf.getvalue())
        zf.writestr('README.txt', '\n'.join(readme) + '\n')

    if missing:
        print('WARNING: missing %d paths (first 10):' % len(missing))
        for path in missing[:10]:
            print(' ', path)
    mode = 'listed'
    if args.recap_run_log:
        mode = 'retried-from-log'
    elif not args.task_list:
        mode = 'random seed=%d' % args.seed
    print('Wrote %s (%d bytes, %d tasks, %s)' % (
        args.zip, Path(args.zip).stat().st_size, len(pick), mode))


if __name__ == '__main__':
    main()
