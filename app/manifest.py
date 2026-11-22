"""源检查点 / 新模型模板清单的解析与校验。

清单为 manifest.json:
{
  "step": 12000,                # 训练步(模板的初值会被忽略,新批次统一归零)
  "total_params": 29,           # 去重后的全局下标总数
  "shards": [                   # 各分片的全局下标区间 [start, end),可跨参数
    {"file": "shard_000.npy", "start": 0, "end": 10},
    ...
  ],
  "params": [                   # 参数清单:参数名、形状、全局下标区间
    {"name": "embed.weight", "shape": [10], "start": 0, "end": 10},
    ...
  ]
}

约定:
- 参数名在同一份清单内唯一;shape 为正整数数组,其乘积必须等于区间长度;
- 共享权重的多个参数名使用相同的 [start, end) 与相同 shape(区间在 params 中重复出现);
- 去掉重复区间后,params 的区间必须从 0 连续覆盖到 total_params,无缺口、无重叠;
- shards 同样必须从 0 连续覆盖 [0, total_params),分片之间互不重叠;
- 每个分片文件是 NumPy .npy,形状 (end-start, 3),dtype float32,
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
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def numel(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Manifest:
    step: int
    total_params: int
    shards: tuple[ShardRef, ...]
    params: tuple[ParamSpec, ...]


def _check_file_name(file: object) -> str:
    if not isinstance(file, str) or not file:
        raise ManifestError("shard.file 必须是非空字符串")
    p = PurePath(file)
    if p.is_absolute() or ".." in p.parts:
        raise ManifestError(f"非法分片路径(不允许绝对路径或 '..'): {file!r}")
    return file


def _check_range(start: object, end: object, where: str, *, allow_empty: bool = False
                 ) -> tuple[int, int]:
    for name, v in (("start", start), ("end", end)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ManifestError(f"{where}.{name} 必须是非负整数")
    if allow_empty:
        if not start <= end:
            raise ManifestError(f"{where} 区间非法: [{start}, {end})")
    elif not start < end:
        raise ManifestError(f"{where} 区间为空或非法: [{start}, {end})")
    return start, end


def _check_cover(ranges: list[tuple[int, int]], total: int, what: str) -> None:
    """去重后校验区间从 0 连续覆盖 [0, total),无缺口、无重叠。

    允许完全相同的区间重复出现(共享权重的多个参数名共用同一段区间)。
    """
    unique = sorted(set(ranges))
    pos = 0
    for s, e in unique:
        if s < pos:
            raise ManifestError(f"{what} 区间重叠: [{s}, {e}) 与前一段(结束于 {pos})重叠")
        if s > pos:
            raise ManifestError(f"{what} 区间有缺口: 全局下标 [{pos}, {s}) 未被覆盖")
        pos = e
    if pos != total:
        raise ManifestError(
            f"{what} 去重后覆盖 [0, {pos}),与 total_params={total} 不符"
        )


def load_manifest(path: Path) -> Manifest:
    """读取并做结构校验(含区间连续性,分片文件另见 validate_shard_files)。"""
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

    shards: list[ShardRef] = []
    shard_files: set[str] = set()
    shard_ranges: list[tuple[int, int]] = []
    for i, s in enumerate(raw_shards):
        if not isinstance(s, dict):
            raise ManifestError(f"shards[{i}] 必须是对象")
        file = _check_file_name(s.get("file"))
        if file in shard_files:
            raise ManifestError(f"分片文件名重复: {file!r}")
        shard_files.add(file)
        # 允许 0 行空片(进程/模板边界可能产生空分片,仍是合法 .npy)
        start, end = _check_range(s.get("start"), s.get("end"),
                                  f"shards[{i}]", allow_empty=True)
        shards.append(ShardRef(file=file, start=start, end=end))
        shard_ranges.append((start, end))
    _check_cover(shard_ranges, total, "分片(shards)")

    raw_params = data.get("params")
    if not isinstance(raw_params, list) or not raw_params:
        raise ManifestError("params 必须是非空数组(每个参数含 name/shape/start/end)")

    params: list[ParamSpec] = []
    names: set[str] = set()
    param_ranges: list[tuple[int, int]] = []
    for i, p in enumerate(raw_params):
        where = f"params[{i}]"
        if not isinstance(p, dict):
            raise ManifestError(f"{where} 必须是对象")
        name = p.get("name")
        if not isinstance(name, str) or not name:
            raise ManifestError(f"{where}.name 必须是非空字符串")
        if name in names:
            raise ManifestError(f"参数名重复: {name!r}(参数名在清单内必须唯一)")
        names.add(name)

        shape = p.get("shape")
        if (not isinstance(shape, list) or not shape
                or any((not isinstance(d, int) or isinstance(d, bool) or d <= 0)
                       for d in shape)):
            raise ManifestError(f"{where}.shape 必须是非空正整数数组,实际为 {shape!r}")
        numel = 1
        for d in shape:
            numel *= d

        start, end = _check_range(p.get("start"), p.get("end"), where)
        if end - start != numel:
            raise ManifestError(
                f"参数 {name!r} 形状 {tuple(shape)} 共 {numel} 个元素,"
                f"与区间长度 {end - start} 不符"
            )
        params.append(ParamSpec(name=name, shape=tuple(shape), start=start, end=end))
        param_ranges.append((start, end))
    _check_cover(param_ranges, total, "参数(params)")

    return Manifest(
        step=step,
        total_params=total,
        shards=tuple(shards),
        params=tuple(params),
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
    validate_shard_files(m, Path(manifest_path).parent)
    return m
