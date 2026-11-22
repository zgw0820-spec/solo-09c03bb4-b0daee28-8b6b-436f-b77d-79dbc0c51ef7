"""源/目标检查点清单的解析与校验。

清单为 manifest.json:
{
  "step": 12000,                # 训练步
  "total_params": 1000,         # 参数总数
  "shards": [                   # 各片的全局下标区间 [start, end)
    {"file": "shard_000.npy", "start": 0, "end": 334},
    ...
  ]
}

每个分片文件是 NumPy .npy,形状 (end-start, 3),dtype float32,
三列依次是:权重、Adam 一阶矩、Adam 二阶矩。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePath

import numpy as np

DTYPE = np.dtype("<f4")
NCOLS = 3


class ManifestError(ValueError):
    """清单或分片文件不符合约定。"""


@dataclass(frozen=True)
class ShardRef:
    file: str
    start: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Manifest:
    step: int
    total_params: int
    shards: tuple[ShardRef, ...]


def _check_file_name(file: object) -> str:
    if not isinstance(file, str) or not file:
        raise ManifestError("shard.file 必须是非空字符串")
    p = PurePath(file)
    if p.is_absolute() or ".." in p.parts:
        raise ManifestError(f"非法分片路径(不允许绝对路径或 '..'): {file!r}")
    return file


def load_manifest(path: Path) -> Manifest:
    """读取并做结构校验(不校验区间连续性,见 validate_manifest)。"""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"清单不存在: {path}")
    except json.JSONDecodeError as e:
        raise ManifestError(f"清单不是合法 JSON: {e}")
    if not isinstance(data, dict):
        raise ManifestError("清单顶层必须是对象")

    step = data.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ManifestError("step 必须是非负整数")

    total = data.get("total_params")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ManifestError("total_params 必须是正整数")

    raw_shards = data.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ManifestError("shards 必须是非空数组")

    shards = []
    for i, s in enumerate(raw_shards):
        if not isinstance(s, dict):
            raise ManifestError(f"shards[{i}] 必须是对象")
        file = _check_file_name(s.get("file"))
        start, end = s.get("start"), s.get("end")
        for name, v in (("start", start), ("end", end)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ManifestError(f"shards[{i}].{name} 必须是非负整数")
        if not start < end:
            raise ManifestError(f"shards[{i}] 区间为空: [{start}, {end})")
        shards.append(ShardRef(file=file, start=start, end=end))

    return Manifest(step=step, total_params=total, shards=tuple(shards))


def validate_manifest(m: Manifest) -> None:
    """校验区间从 0 开始连续覆盖 [0, total_params),无缺口、无重叠。"""
    pos = 0
    for sh in m.shards:
        if sh.start < pos:
            raise ManifestError(
                f"分片区间重叠: [{sh.start}, {sh.end}) 与前一片 (结束于 {pos}) 重叠"
            )
        if sh.start > pos:
            raise ManifestError(
                f"分片区间有缺口: 全局下标 [{pos}, {sh.start}) 不属于任何分片"
            )
        pos = sh.end
    if pos != m.total_params:
        raise ManifestError(
            f"分片区间合计覆盖 [0, {pos}),与 total_params={m.total_params} 不符"
        )


def validate_shard_files(m: Manifest, base_dir: Path) -> None:
    """校验每个分片文件存在,且形状/dtype 与清单区间一致。"""
    base_dir = Path(base_dir)
    for sh in m.shards:
        f = base_dir / sh.file
        if not f.is_file():
            raise ManifestError(f"缺少分片文件: {sh.file}")
        try:
            arr = np.load(f, mmap_mode="r")
        except Exception as e:
            raise ManifestError(f"无法读取分片 {sh.file}: {e}")
        if arr.dtype != DTYPE:
            raise ManifestError(
                f"分片 {sh.file} dtype 为 {arr.dtype},应为 float32 (<f4)"
            )
        if arr.shape != (sh.rows, NCOLS):
            raise ManifestError(
                f"分片 {sh.file} 形状为 {arr.shape},应为 ({sh.rows}, {NCOLS})"
                f"(清单区间 [{sh.start}, {sh.end}))"
            )


def load_and_validate(manifest_path: Path) -> Manifest:
    m = load_manifest(manifest_path)
    validate_manifest(m)
    validate_shard_files(m, Path(manifest_path).parent)
    return m
