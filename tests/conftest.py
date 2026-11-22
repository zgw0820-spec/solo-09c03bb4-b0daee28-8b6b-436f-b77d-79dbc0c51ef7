"""测试共享工具:构造带参数清单(params)的确定性检查点。"""
import json

# 在导入 app.main(模块级 create_app)之前,把默认目录指到可写的临时位置
import os
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reshaper-test-data-"))
os.environ.setdefault("CHECKPOINTS_DIR", tempfile.mkdtemp(prefix="reshaper-test-ckpts-"))

import numpy as np


def weights_array(total: int, base: float = 0.0) -> np.ndarray:
    """长度 total 的确定性权重: base + i(float32)。"""
    return (base + np.arange(total, dtype=np.float64)).astype("<f4")


def make_checkpoint(
    root: Path,
    name: str,
    step: int,
    shard_sizes: list[int],
    params: list[dict],
    weights: np.ndarray | None = None,
    moments: tuple[float, float] = (0.5, 0.25),
) -> Path:
    """在 root/name 下写带 params 的清单与分片,返回 manifest.json 路径。

    params: [{"name", "shape", "start", "end"}],区间可跨分片、可重复(共享)。
    weights 缺省取 arange(total);分片按 shard_sizes 切。
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    total = sum(shard_sizes)
    if weights is None:
        weights = weights_array(total)
    m1 = np.full(total, moments[0], dtype="<f4")
    m2 = np.full(total, moments[1], dtype="<f4")
    full = np.stack([weights.astype("<f4"), m1, m2], axis=1)

    shards = []
    pos = 0
    for i, n in enumerate(shard_sizes):
        fname = f"shard_{i:03d}.npy"
        np.save(d / fname, full[pos:pos + n])
        shards.append({"file": fname, "start": pos, "end": pos + n})
        pos += n

    manifest = {
        "step": step,
        "total_params": total,
        "shards": shards,
        "params": params,
    }
    mpath = d / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return mpath


def read_batch(batch_dir: Path) -> tuple[dict, np.ndarray]:
    """读回一个批次的清单与拼接后的完整数组。"""
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    parts = [np.load(batch_dir / s["file"]) for s in manifest["shards"]]
    return manifest, np.concatenate(parts) if parts else np.empty((0, 3), "<f4")
