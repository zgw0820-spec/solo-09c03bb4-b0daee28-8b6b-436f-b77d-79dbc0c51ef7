"""测试共享工具:构造确定性检查点。"""
import json

# 在导入 app.main(模块级 create_app)之前,把默认目录指到可写的临时位置
import os
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reshaper-test-data-"))
os.environ.setdefault("CHECKPOINTS_DIR", tempfile.mkdtemp(prefix="reshaper-test-ckpts-"))

import numpy as np


def expected_array(total: int) -> np.ndarray:
    """下标 i 的三列: 权重=i, 一阶矩=i+0.5, 二阶矩=i*0.25。"""
    idx = np.arange(total, dtype=np.float64)
    return np.stack([idx, idx + 0.5, idx * 0.25], axis=1).astype("<f4")


def make_checkpoint(root: Path, name: str, step: int, sizes: list[int]) -> Path:
    """在 root/name 下写清单与分片,返回清单路径。"""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    shards = []
    pos = 0
    for i, n in enumerate(sizes):
        arr = expected_array(pos + n)[pos:]
        fname = f"shard_{i:03d}.npy"
        np.save(d / fname, arr)
        shards.append({"file": fname, "start": pos, "end": pos + n})
        pos += n
    manifest = {"step": step, "total_params": pos, "shards": shards}
    mpath = d / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return mpath


def read_batch(batch_dir: Path) -> tuple[dict, np.ndarray]:
    """读回一个批次的清单与拼接后的完整数组。"""
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    parts = [np.load(batch_dir / s["file"]) for s in manifest["shards"]]
    return manifest, np.concatenate(parts) if parts else np.empty((0, 3), "<f4")
