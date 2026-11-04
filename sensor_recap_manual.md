# Sensor Recap 操作手册

用 CARLA **回放已有录制 log**，批量重新生成 **depth**（四路相机）或 **lidar**，再挂回训练用的数据集目录。  
在 Docker 里操作即可，数据路径统一用容器内的 **`/workspace/...`**。

### 深度 recap 落地状态（2026-10-13）

| Split | Mirror task 数 | `recap_manifest` | 挂回训练树 |
| --- | ---: | --- | --- |
| `gear_max_3_ratio_91_enhanced/val_town04` | 55 | 全部 `ok` | 四路 `depth_*` → `recap_val_town04_depth`（220 条软链） |
| `gear_max_3_ratio_91_enhanced/train_town04` | 507 | 全部 `ok` | 四路 `depth_*` → `recap_train_town04_depth`（2028 条软链） |
| `tiny/val_town04` + `tiny/train_town04` | 3 + 8 | 全部 `ok` | 四路 `depth_*` → `recap_tiny_depth`（44 条软链） |

Mirror 目录保留作备份；训练仍读 `enhanced` / `tiny` 下路径，**不必**改 `config/training.yaml`。旧 depth 在各自 task 下为 `depth_*.old_symlinks`。

快速复检（容器内路径，宿主机把 `/workspace` 换成 `/home/USER/workspace` 即可）：

```bash
python3 - <<'PY'
import json
from pathlib import Path

BASE = Path("/workspace/dataset/carla_sim_parking")
CAMS = ("depth_front", "depth_left", "depth_right", "depth_rear")
CASES = [
    ("enhanced val", BASE / "gear_max_3_ratio_91_enhanced/val_town04", BASE / "recap_val_town04_depth/val_town04"),
    ("enhanced train", BASE / "gear_max_3_ratio_91_enhanced/train_town04", BASE / "recap_train_town04_depth/train_town04"),
    ("tiny val", BASE / "tiny/val_town04", BASE / "recap_tiny_depth/val_town04"),
    ("tiny train", BASE / "tiny/train_town04", BASE / "recap_tiny_depth/train_town04"),
]
for label, ds, mir in CASES:
    tasks = sorted(mir.glob("*/task*"))
    bad = []
    for task in tasks:
        man = task / "recap_manifest.json"
        p = json.loads(man.read_text())
        rel = task.relative_to(mir)
        meas = len(list((ds / rel / "measurements").glob("*.json")))
        n = len(list((task / "depth_front").glob("*.png")))
        if p.get("status") != "ok" or n != meas:
            bad.append(str(rel))
    ns = sum(1 for t in ds.glob("*/task*") for c in CAMS if (t / c).exists() and not (t / c).is_symlink())
    print(label, "mirror_tasks", len(tasks), "manifest_bad", len(bad), "non_symlink_depth_dirs", ns)
PY
```

期望四行均为 `manifest_bad 0`、`non_symlink_depth_dirs 0`。

---

## 1. 准备

| 项目 | 说明 |
| --- | --- |
| CARLA | **0.9.13** Server 已启动（与容器里 `import carla` 版本一致） |
| 容器 | 能访问 `/workspace/dataset/carla_sim_parking` 和本仓库 |
| 训练数据 | 默认指 `gear_max_3_ratio_91_enhanced` 下的 `train_town04` / `val_town04` |
| 入口命令 | `python3 carla_sensor_recap.py` |

Recap **只写你选的传感器**（如 depth），**不会**改 RGB、measurements、topdown 等。

---

## 2. 两个目录各干什么

```
训练 / 评测读这里（config 里配置的 split）
gear_max_3_ratio_91_enhanced/
└── val_town04 或 train_town04/
    └── {episode}/taskN/
        ├── measurements/     ← 帧数以此为准
        ├── rgb_*             ← 不动
        └── depth_*           ← 挂链后要指向下面 mirror 里的新 depth

Recap 先写到这里（--output_root，叫 mirror）
recap_val_town04_depth/       ← val 全量 depth 示例
recap_train_town04_depth/     ← train 全量 depth
recap_tiny_depth/             ← 小规模试跑
└── val_town04 或 train_town04/
    └── {episode}/taskN/
        ├── depth_front|left|right|rear/
        └── recap_manifest.json
```

**训练程序只认 enhanced（或 tiny）里的路径**，不会自动读 mirror。mirror 跑完后，需要把 enhanced 里四个 `depth_*` **目录**改成指向 mirror 的**软链**（见第 5 步）。

---

## 3. 推荐流程（按顺序做）

```
[1] dry_run 盘点  →  [2]（可选）tiny 试跑  →  [3] 全量 recap 到 mirror
        →  [4] 检查 mirror  →  [5] 软链挂回 enhanced  →  [6] 验证  →  [7] 正常训练
```

### 第 1 步：盘点（不用开 CARLA）

看有多少 task 能 replay、会不会被 skip：

```bash
python3 carla_sensor_recap.py \
  --dry_run \
  --sensors depth \
  --splits /workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced/val_town04
```

train 把路径换成 `.../train_town04`。需要时可同时写两个 `--splits`。

### 第 2 步：（可选）小规模试跑

先写到独立 mirror，避免动正式数据：

```bash
python3 carla_sensor_recap.py \
  --host 127.0.0.1 --port 2000 \
  --splits /workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced/val_town04 \
  --sensors depth \
  --output_root /workspace/dataset/carla_sim_parking/recap_tiny_depth \
  --limit 5
```

试跑满意后再做全量。

### 第 3 步：全量 recap 到 mirror

**Val 示例**（已常用的 mirror 名）：

```bash
python3 carla_sensor_recap.py \
  --host 127.0.0.1 --port 2000 \
  --splits /workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced/val_town04 \
  --sensors depth \
  --output_root /workspace/dataset/carla_sim_parking/recap_val_town04_depth \
  --target_fps 30 --save_frequency 3
```

**Train**：split 改为 `train_town04`，mirror 改为 `recap_train_town04_depth`。

- 中断后**再跑同一命令**：已完成的 task 会自动跳过。  
- 只有失败的 task：加 `--retry_failed`（必须同一个 `--output_root`）。  
- 日志：`{output_root}/val_town04/recap_run.log`（或 train 对应路径）。

### 第 4 步：检查 mirror 是否齐全

每个 task 应有 `recap_manifest.json` 且 `status` 为 `ok`，PNG 数量等于 `measurements` 条数。

```bash
python3 - <<'PY'
import json
from pathlib import Path

# val 检查：train 时改下面两行路径
mirror_split = Path("/workspace/dataset/carla_sim_parking/recap_val_town04_depth/val_town04")
orig_split = Path("/workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced/val_town04")

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

`issues: 0` 再挂链。想目视对比 RGB 与新 depth，可用：

```bash
python3 tool/pack_recap_stitched_zip.py \
  --split /workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced/val_town04 \
  --output_root /workspace/dataset/carla_sim_parking/recap_val_town04_depth \
  --split_name val_town04 \
  --zip /tmp/recap_val_check.zip \
  --depth_vis
```

### 第 5 步：软链挂回训练目录

在容器里**整段复制**（含变量）。只复制 `while` 循环而不设 `MIRROR`/`ORIG` 会**没有任何输出**。

**Val + enhanced**：

```bash
MIRROR=/workspace/dataset/carla_sim_parking/recap_val_town04_depth
ORIG=/workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced
CAMS="depth_front depth_left depth_right depth_rear"

echo "tasks: $(find "$MIRROR/val_town04" -mindepth 2 -maxdepth 2 -type d -name 'task*' | wc -l)"

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

| 场景 | 改什么 |
| --- | --- |
| **Train**（mirror 全部 ok 后再做） | `MIRROR=.../recap_train_town04_depth`，`find` 路径里 `val_town04` → `train_town04` |
| **Tiny 数据集** | `MIRROR=.../recap_tiny_depth`，`ORIG=.../tiny`，`find "$MIRROR" -mindepth 3 -maxdepth 3 -name 'task*'`，`rel` 含 `train_town04/...` 或 `val_town04/...` |

旧 depth 目录会改名为 `depth_*.old_symlinks` 保留；**回滚**：删掉软链，把 `.old_symlinks` 改回原名。

### 第 6 步：验证挂链是否生效

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path

# val：train / tiny 时改 m_root、s_root
m_root = Path("/workspace/dataset/carla_sim_parking/recap_val_town04_depth/val_town04")
s_root = Path("/workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced/val_town04")

tasks_m = sorted(m_root.glob("*/task*"))   # 注意：两层 */task*，不要 */*/task*
assert tasks_m, "mirror 下没有 task"
task_m = tasks_m[0]
rel = task_m.relative_to(m_root)
df = s_root / rel / "depth_front"
print("check", rel)
assert df.is_symlink()
p = df / "0000.png"
assert p.is_file() and not p.is_symlink()
print("md5 match", hashlib.md5(p.read_bytes()).hexdigest() ==
      hashlib.md5((task_m / "depth_front/0000.png").read_bytes()).hexdigest())

bad = [str(s_root / t.relative_to(m_root) / c)
       for t in tasks_m for c in ("depth_front","depth_left","depth_right","depth_rear")
       if not (s_root / t.relative_to(m_root) / c).is_symlink()]
print("non-symlink depth dirs:", len(bad))
PY
```

期望：`md5 match True`，`non-symlink depth dirs: 0`。

### 第 7 步：训练

`config/training.yaml` 里数据路径**不用改**（仍指向 `gear_max_3_ratio_91_enhanced/...`）。mirror 目录可长期保留作备份。

---

## 4. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 挂链脚本无任何输出 | 未设置 `MIRROR`/`ORIG`，或 `find` 深度错了 | 必须 export 三个变量；在 `$MIRROR/val_town04` 下用 `-mindepth 2 -maxdepth 2` |
| Python 报 mirror 没有 task | 用了 `glob("*/*/task*")` | 在 split 目录下用 `glob("*/task*")` |
| 训练仍像旧 depth | 只写了 mirror，未做第 5 步 | 对 enhanced 做目录软链 |
| recap 中途失败 | 单个 task 报错 | 看 `recap_run.log`；修 CARLA/log 后 `--retry_failed` |
| 想恢复旧 depth | 已挂链 | 删软链，恢复 `*.old_symlinks` |

---

## 5. 不会自动做的事

- 不重录 RGB、BEV、measurements  
- 不改训练代码和 yaml  
- 没有可用 replay log（且没有 `carla_actor.data`）的 task 会跳过，无法凭空生成  

---

## 6. 路径速查

| 目的 | 路径 |
| --- | --- |
| 训练 val split | `/workspace/dataset/carla_sim_parking/gear_max_3_ratio_91_enhanced/val_town04` |
| 训练 train split | `.../gear_max_3_ratio_91_enhanced/train_town04` |
| val depth mirror | `.../recap_val_town04_depth` |
| train depth mirror | `.../recap_train_town04_depth` |
| 试跑 / tiny mirror | `.../recap_tiny_depth` |
| tiny 调试 split | `.../tiny/train_town04`、`.../tiny/val_town04` |
