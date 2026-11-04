# SunnyParking CARLA Toolkit

Toolkit for our IEEE Access paper
[*SunnyParking: Multi-Shot Trajectory Generation and Motion State Awareness for Human-like Parking*](https://doi.org/10.1109/ACCESS.2026.3752444).

Companion to the paper dataset on Hugging Face: [MPRG/sunnyparking](https://huggingface.co/datasets/MPRG/sunnyparking) — collect human park-ins in CARLA, then recapture sensors from recording logs.

Thanks to [e2e-parking-carla](https://github.com/qintonguav/e2e-parking-carla) for the codebase this toolkit builds on.

Keyboard or Logitech G29 (`wheel_config.ini`). Wheel throttle / brake / steer tracks a real car better than WASD.

<img src="resource/sunnyparking_image.jpeg" width="600">

- **[04/11/2026]** code release

## What it does

1. **Collect** — `carla_data_gen.py` in Town04: surround RGB/depth, measurements, `replay/recoding.log`
2. **Recap** — replay the log (or rebuild actors) and write only the sensors you ask for; RGB / measurements stay untouched

Depth recap is production-ready (`depth_front/left/right/rear`, 400×300, fov 100, same extrinsics as collection). Lidar / semantic configs exist but are not fully validated here.

## Setup

- CARLA 0.9.13 (match the Python API)
- `environment.yml` (Python 3.10)
- Recap needs per-task `measurements/`, `rgb_*`, and a replayable `replay/recoding.log`
- Optional: G29 via `wheel_config.ini`

```bash
git clone https://github.com/jishumiao95/sunnyparking_carla_toolkit.git
cd sunnyparking_carla_toolkit
conda env create -f environment.yml
conda activate e2e
```

If CARLA runs on the host and this repo is bind-mounted into Docker (`/workspace`), map paths so the server can open the recorder file:

```bash
--server_path_map /workspace=/home/USER/workspace
```

## 1. Collect

```bash
./CarlaUE4.sh -prefernvidia
python3 carla_data_gen.py
```

Keyboard: `w/a/s/d` throttle/steer/brake, `q` reverse, `BackSpace` reset, `TAB` camera.  
G29: plug in before launch; axes/buttons under `[G29 Racing Wheel]` in `wheel_config.ini`.

Each task writes `rgb_*`, `depth_*`, `measurements/`, `parking_layout/`, `replay/recoding.log`.

## 2. Recap depth

Replay the log and fill in surround depth. Only selected sensors are written.

```bash
export DATASET_ROOT=/path/to/dataset          # has val_town04 / train_town04
export MIRROR_ROOT=/path/to/recap_mirror
```

Typical path: dry-run → one task / `--limit` → `--output_root` mirror → QA → symlink `depth_*` back into train/val.

### Inventory (no CARLA)

```bash
python3 carla_sensor_recap.py \
  --dry_run \
  --sensors depth \
  --splits $DATASET_ROOT/val_town04
```

### Smoke one task

```bash
python3 carla_sensor_recap.py \
  --host 127.0.0.1 --port 2000 \
  --splits $DATASET_ROOT/val_town04 \
  --sensors depth \
  --episode <episode_name> \
  --task task1 \
  --limit 1 \
  --output_root $MIRROR_ROOT \
  --server_path_map /workspace=/home/USER/workspace
```

### Full split → mirror

```bash
python3 carla_sensor_recap.py \
  --host 127.0.0.1 --port 2000 \
  --splits $DATASET_ROOT/val_town04 \
  --sensors depth \
  --output_root $MIRROR_ROOT \
  --target_fps 30 --save_frequency 3 \
  --server_path_map /workspace=/home/USER/workspace
```

Swap the split for train. Re-runs skip finished tasks; `--retry_failed` redoes only failures. Each task gets a `recap_manifest.json`; PNG count should match `measurements/`.

### Wire depth into the train path

`--output_root` is a separate mirror so the original set stays intact. Point `depth_*` at the mirror with symlinks (keep RGB / measurements in place; rename old dirs to `depth_*.old_symlinks` for rollback):

```bash
MIRROR=/path/to/recap_mirror
ORIG=/path/to/dataset
CAMS="depth_front depth_left depth_right depth_rear"

find "$MIRROR/val_town04" -mindepth 2 -maxdepth 2 -type d -name 'task*' | sort | while read -r src; do
  rel=${src#$MIRROR/}
  dst="$ORIG/$rel"
  test -d "$dst/measurements" || { echo "SKIP $rel"; continue; }
  for cam in $CAMS; do
    test -d "$src/$cam" || continue
    if [ -L "$dst/$cam" ]; then
      echo "already linked $rel/$cam"
      continue
    fi
    mv "$dst/$cam" "$dst/${cam}.old_symlinks"
    ln -s "$MIRROR/$rel/$cam" "$dst/$cam"
    echo "linked $rel/$cam"
  done
done
```

Same for `train_town04`. Skip this if training already points at the recap output.

### QA (optional)

```bash
python3 tool/pack_recap_stitched_zip.py \
  --split $DATASET_ROOT/val_town04 \
  --output_root $MIRROR_ROOT \
  --split_name val_town04 \
  --zip /tmp/recap_val_check.zip \
  --depth_vis

python3 tool/pack_recap_rgb_depth_zip.py --help
```

Full notes: [sensor_recap_manual.md](./sensor_recap_manual.md)

## Layout

```text
carla_data_gen.py              # collection entry
carla_sensor_recap.py          # recap CLI
data_generation/sensor_recap.py
data_generation/sensor_specs.py
wheel_config.ini               # G29 axes / buttons
tool/pack_recap_*.py
tests/test_sensor_recap.py
```

## Citation

If you find this work helpful, please cite:

```bibtex
@ARTICLE{11789576,
  author={Miao, Jishu and Chen, Han and Zhai, Jiankun and Liu, Qi and Hirakawa, Tsubasa and Yamashita, Takayoshi and Fujiyoshi, Hironobu},
  journal={IEEE Access}, 
  title={SunnyParking: Multi-Shot Trajectory Generation and Motion State Awareness for Human-like Parking}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Trajectory;Gears;Vehicles;Conferences;Educational institutions;Grounding;Planning;Autonomous driving;Computers;Automated parking;autonomous driving;end-to-end planning;imitation learning},
  doi={10.1109/ACCESS.2026.3752444}}
```

## License

Research use. CARLA, third-party weights, and datasets keep their own licenses. No enterprise checkpoints or private logs ship here.
