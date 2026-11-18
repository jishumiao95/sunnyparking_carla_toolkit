# Sensor Recap Manual

Operator notes for regenerating **depth** (four surround cameras) from existing CARLA recorder logs, then optionally wiring the new depth back into a training split.

README covers the short path. This page is the longer checklist: mirror layout, QA, symlink, rollback.

Paths below use placeholders — set them once:

```bash
export DATASET_ROOT=/path/to/dataset          # has val_town04 / train_town04
export MIRROR_ROOT=/path/to/recap_mirror      # recap --output_root
```

Under Docker with a `/workspace` bind-mount, point these at container paths (e.g. `/workspace/dataset/...`). Host CARLA + container client: pass `--server_path_map /workspace=/home/USER/workspace` as needed.

---

## Layout

```
$DATASET_ROOT/val_town04|train_town04/
└── {episode}/taskN/
    ├── measurements/     ← frame count source of truth
    ├── rgb_*             ← left untouched
    └── depth_*           ← after step 5: symlinks into the mirror

$MIRROR_ROOT/val_town04|train_town04/
└── {episode}/taskN/
    ├── depth_front|left|right|rear/
    └── recap_manifest.json
```

Recap **only** writes the sensors you select. Training configs that still point at `$DATASET_ROOT` will not see new depth until you symlink (or retarget the config).

---

## Flow

```
dry_run → (optional) --limit smoke → full recap to $MIRROR_ROOT
       → check mirror → symlink depth_* → verify → train
```

### 1. Inventory (no CARLA)

```bash
python3 carla_sensor_recap.py \
  --dry_run \
  --sensors depth \
  --splits $DATASET_ROOT/val_town04
```

### 2. Smoke (optional)

```bash
python3 carla_sensor_recap.py \
  --host 127.0.0.1 --port 2000 \
  --splits $DATASET_ROOT/val_town04 \
  --sensors depth \
  --output_root $MIRROR_ROOT \
  --limit 5
```

### 3. Full split → mirror

```bash
python3 carla_sensor_recap.py \
  --host 127.0.0.1 --port 2000 \
  --splits $DATASET_ROOT/val_town04 \
  --sensors depth \
  --output_root $MIRROR_ROOT \
  --target_fps 30 --save_frequency 3
```

Swap in `train_town04` for train. Re-run the same command to skip finished tasks; `--retry_failed` retries only failures (same `--output_root`). Log: `$MIRROR_ROOT/<split>/recap_run.log`.

### 4. Check mirror

Each task should have `recap_manifest.json` with `status: ok`, and PNG count == `measurements/` count.

```bash
python3 - <<'PY'
import json, os
from pathlib import Path

mirror_split = Path(os.environ["MIRROR_ROOT"]) / "val_town04"
orig_split = Path(os.environ["DATASET_ROOT"]) / "val_town04"

tasks = sorted(mirror_split.glob("*/task*"))
bad = []
for task in tasks:
    man = task / "recap_manifest.json"
    if not man.is_file():
        bad.append((task.name, "no manifest"))
        continue
    p = json.loads(man.read_text())
    rel = task.relative_to(mirror_split)
    meas = len(list((orig_split / rel / "measurements").glob("*.json")))
    n = len(list((task / "depth_front").glob("*.png")))
    if p.get("status") != "ok" or p.get("local_count") != meas or n != meas:
        bad.append((str(rel), p.get("status"), meas, p.get("local_count"), n))

print("tasks:", len(tasks), "issues:", len(bad))
for row in bad[:10]:
    print(" ", row)
PY
```

Optional visual pack:

```bash
python3 tool/pack_recap_stitched_zip.py \
  --split $DATASET_ROOT/val_town04 \
  --output_root $MIRROR_ROOT \
  --split_name val_town04 \
  --zip /tmp/recap_val_check.zip \
  --depth_vis
```

### 5. Symlink depth into the training tree

Leaves RGB / measurements in place; renames old `depth_*` to `depth_*.old_symlinks` for rollback.

```bash
MIRROR=$MIRROR_ROOT
ORIG=$DATASET_ROOT
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

For train: use `$MIRROR/train_town04`. Rollback: remove the symlinks and rename `*.old_symlinks` back.

### 6. Verify symlink

```bash
python3 - <<'PY'
import hashlib, os
from pathlib import Path

m_root = Path(os.environ["MIRROR_ROOT"]) / "val_town04"
s_root = Path(os.environ["DATASET_ROOT"]) / "val_town04"

tasks_m = sorted(m_root.glob("*/task*"))
assert tasks_m, "no tasks under mirror split"
task_m = tasks_m[0]
rel = task_m.relative_to(m_root)
df = s_root / rel / "depth_front"
print("check", rel)
assert df.is_symlink()
p = df / "0000.png"
assert p.is_file() and not p.is_symlink()
print("md5 match", hashlib.md5(p.read_bytes()).hexdigest() ==
      hashlib.md5((task_m / "depth_front/0000.png").read_bytes()).hexdigest())

cams = ("depth_front", "depth_left", "depth_right", "depth_rear")
bad = [str(s_root / t.relative_to(m_root) / c)
       for t in tasks_m for c in cams
       if not (s_root / t.relative_to(m_root) / c).is_symlink()]
print("non-symlink depth dirs:", len(bad))
PY
```

Expect `md5 match True` and `non-symlink depth dirs: 0`.

---

## FAQ

| Symptom | Fix |
| --- | --- |
| Symlink script prints nothing | Export `MIRROR` / `ORIG`; under `$MIRROR/val_town04` use `-mindepth 2 -maxdepth 2` |
| "no task" under mirror | Glob from the split dir: `*/task*`, not `*/*/task*` |
| Training still looks like old depth | Mirror alone is not enough — finish the symlink step (or retarget config) |
| Mid-run failure | Check `recap_run.log`, then `--retry_failed` with the same `--output_root` |
| Restore old depth | Delete symlinks, rename `*.old_symlinks` back |

## What this does **not** do

- Does not rewrite RGB, BEV, or measurements
- Does not change training code / yaml by itself
- Skips tasks with no usable `replay/recoding.log` and no `carla_actor.data`
