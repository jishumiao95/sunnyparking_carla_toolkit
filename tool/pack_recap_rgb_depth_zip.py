#!/usr/bin/env python3
"""Pack RGB (split tree) and recap depth (output_root) samples into a zip."""

import argparse
import json
import os
import zipfile
from pathlib import Path

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


def parse_task_spec(text):
    """Format: [group:]episode/task_name  (group defaults to 'task')."""
    group = 'task'
    body = text
    if ':' in text:
        group, body = text.split(':', 1)
    if '/' not in body:
        raise ValueError('task must be episode/task_name, got: %s' % text)
    episode, task_name = body.rsplit('/', 1)
    return group, episode, task_name


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


def add_file(zf, arcname, source, path_maps):
    source = resolve_file(source, path_maps)
    zf.write(source, arcname)


def pack_task(zf, group, episode, task_name, split, recap_root, samples, path_maps, readme, missing):
    task_split = split / episode / task_name
    task_recap = recap_root / episode / task_name
    manifest = task_recap / 'recap_manifest.json'
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        count = int(payload.get('local_count', 0))
        status = payload.get('status', '?')
        source = payload.get('source', '?')
    else:
        depth_dir = task_recap / 'depth_front'
        count = len(list(depth_dir.glob('*.png'))) if depth_dir.is_dir() else 0
        status = 'no_manifest'
        source = '?'
    indices = sample_indices(count, samples)
    readme.append('%s/%s/%s  frames=%d  manifest=%s  source=%s  samples=%s' % (
        group, episode, task_name, count, status, source, indices))

    for local in indices:
        stem = '%04d.png' % local
        prefix = '%s/%s/%s/frame_%s' % (group, episode, task_name, stem.replace('.png', ''))
        for cam in CAMERAS:
            rgb = task_split / ('rgb_%s' % cam) / stem
            depth = task_recap / ('depth_%s' % cam) / stem
            for arc_suffix, path in (
                    ('rgb_%s.png' % cam, rgb),
                    ('depth_%s.png' % cam, depth)):
                try:
                    add_file(zf, '%s/%s' % (prefix, arc_suffix), path, path_maps)
                except FileNotFoundError:
                    missing.append(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True, help='e.g. .../gear_max_3_ratio_91_enhanced/val_town04')
    parser.add_argument('--output_root', required=True, help='recap mirror root')
    parser.add_argument('--split_name', default='val_town04')
    parser.add_argument('--zip', required=True, help='output zip path')
    parser.add_argument('--samples', type=int, default=8, help='frames per task (evenly spaced)')
    parser.add_argument(
        '--path_map', action='append', default=None,
        help='rewrite symlink targets, e.g. /workspace=/home/USER/workspace (repeatable)')
    parser.add_argument(
        '--task', action='append', required=True,
        help='[group:]episode/task; repeatable. group defaults to task')
    args = parser.parse_args()
    path_maps = parse_path_maps(args.path_map)

    split = Path(args.split)
    recap_root = Path(args.output_root) / args.split_name
    missing = []
    task_list = [parse_task_spec(item) for item in args.task]

    with zipfile.ZipFile(args.zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        readme = [
            'RGB: split task dir (original collection).',
            'depth: recap output_root mirror.',
            'Local frame id matches measurements/ 0000..M-1.',
            '',
        ]
        for group, episode, task_name in task_list:
            pack_task(
                zf, group, episode, task_name, split, recap_root,
                args.samples, path_maps, readme, missing)
        zf.writestr('README.txt', '\n'.join(readme) + '\n')

    if missing:
        print('WARNING: missing %d files (first 10):' % len(missing))
        for path in missing[:10]:
            print(' ', path)
    print('Wrote %s (%d bytes, %d tasks)' % (
        args.zip, Path(args.zip).stat().st_size, len(task_list)))


if __name__ == '__main__':
    main()
