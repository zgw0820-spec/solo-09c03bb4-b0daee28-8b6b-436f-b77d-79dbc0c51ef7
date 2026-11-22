#!/usr/bin/env python3
"""生成本地样例检查点。

用法: python3 scripts/make_sample.py [输出目录]
默认写到 sample_data/checkpoints/,生成:
  ckpt-demo/  训练步 12000,1000 个参数,3 片 (334/333/333)
  ckpt-tiny/  训练步 30,10 个参数,2 片 (5/5)

每个下标 i 的三列取确定性值,便于核对:
  权重 = i, 一阶矩 = i + 0.5, 二阶矩 = i * 0.25  (float32)
"""
import json
import sys
from pathlib import Path

import numpy as np


def write_checkpoint(root: Path, name: str, step: int, sizes: list[int]) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    shards = []
    pos = 0
    for i, n in enumerate(sizes):
        idx = np.arange(pos, pos + n, dtype=np.float64)
        arr = np.stack([idx, idx + 0.5, idx * 0.25], axis=1).astype("<f4")
        fname = f"shard_{i:03d}.npy"
        np.save(d / fname, arr)
        shards.append({"file": fname, "start": pos, "end": pos + n})
        pos += n
    manifest = {"step": step, "total_params": pos, "shards": shards}
    (d / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"已生成 {d} (训练步 {step}, {pos} 个参数, {len(sizes)} 片)")


if __name__ == "__main__":
    root = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "sample_data" / "checkpoints"
    )
    write_checkpoint(root, "ckpt-demo", 12000, [334, 333, 333])
    write_checkpoint(root, "ckpt-tiny", 30, [5, 5])
