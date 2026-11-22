"""旧检查点 -> 新模型模板的参数级迁移(支持层重排与共享权重)。

迁移规则(见 README):
- 新批次沿用模板的分片边界(文件与 [start, end) 与模板一致);
- 每个目标参数区间按 (参数名, 形状) 在旧清单里找同名同形状的旧参数:
  找到则继承旧权重,找不到则保留模板初值;
- 共享权重:多个目标参数名共用同一目标区间时,若该区间在旧检查点对应
  多份(不同区间的)旧权重,原始位值完全一致才继承,否则报冲突并列出参数名;
- 训练步与两阶 Adam 矩(第 2、3 列)在新批次中一律归零;
- 流式处理,读块最多 READ_ROWS 行、写缓冲最多 WRITE_ROWS 行,
  合计 (READ_ROWS*2 + WRITE_ROWS) * 12 字节 <= 64 KiB
  (两个读缓冲:旧检查点与模板各一,按 12 字节/行上界计;写缓冲按 12 字节/行)。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manifest import (
    DTYPE,
    NCOLS,
    Manifest,
    ManifestError,
    ParamSpec,
    load_and_validate,
)

ROW_BYTES = DTYPE.itemsize * NCOLS  # 12 字节/行
MAX_BUFFER_BYTES = 64 * 1024  # 分片读写缓冲合计上限
READ_ROWS = 1024  # 每个读块: 1024 * 12 = 12 KiB
WRITE_ROWS = 2048  # 写缓冲: 2048 * 12 = 24 KiB,合计 48 KiB <= 64 KiB
# 实际读缓冲只取权重一列(4 字节/行),这里按整行 12 字节保守计上界。


def buffer_footprint_bytes(read_rows: int = READ_ROWS, write_rows: int = WRITE_ROWS) -> int:
    """分片读写缓冲的合计上界(字节):旧/模板两个读缓冲 + 一个写缓冲。"""
    return (read_rows * 2 + write_rows) * ROW_BYTES


def check_buffer_limits(read_rows: int = READ_ROWS, write_rows: int = WRITE_ROWS) -> None:
    if read_rows < 1 or write_rows < 1:
        raise ValueError("读写缓冲行数必须 >= 1")
    footprint = buffer_footprint_bytes(read_rows, write_rows)
    if footprint > MAX_BUFFER_BYTES:
        raise ValueError(
            f"读写缓冲合计 {footprint} 字节,超过上限 {MAX_BUFFER_BYTES} 字节 (64 KiB)"
        )


class ShardWriter:
    """增量写一个 .npy 分片(3 列),内存缓冲最多 buf_rows 行。

    先写 <name>.tmp,close 时校验行数、落盘并 rename 成正式文件。
    """

    def __init__(self, path: Path, nrows: int, buf_rows: int = WRITE_ROWS):
        self.path = Path(path)
        self.nrows = nrows
        self.buf_rows = buf_rows
        self.rows_written = 0
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._buf = bytearray()
        self._fh = open(self._tmp, "wb")
        np.lib.format.write_array_header_1_0(
            self._fh, {"descr": DTYPE.str, "fortran_order": False, "shape": (nrows, NCOLS)}
        )

    @property
    def remaining(self) -> int:
        return self.nrows - self.rows_written

    def write(self, rows: np.ndarray) -> None:
        if self._fh is None:
            raise IOError(f"分片 {self.path.name} 已关闭")
        if rows.dtype != DTYPE:
            rows = rows.astype(DTYPE)
        if not rows.flags.c_contiguous:
            rows = np.ascontiguousarray(rows)
        self._buf += rows.tobytes()
        self.rows_written += rows.shape[0]
        limit = self.buf_rows * ROW_BYTES
        while len(self._buf) >= limit:
            self._fh.write(self._buf[:limit])
            del self._buf[:limit]

    def close(self) -> None:
        if self._fh is None:
            return
        if self.rows_written != self.nrows:
            self._fh.close()
            self._fh = None
            self._tmp.unlink(missing_ok=True)
            raise IOError(
                f"分片 {self.path.name} 只写入 {self.rows_written}/{self.nrows} 行,已放弃"
            )
        if self._buf:
            self._fh.write(self._buf)
            self._buf.clear()
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None
        os.rename(self._tmp, self.path)


class WeightStreamReader:
    """按全局下标顺序流式读取一个检查点的权重列(第 1 列),自动跨分片。

    mmap 只做文件映射不计入缓冲;真正驻留内存的读块每次最多 read_rows 行,
    每次只复制权重一列。支持 seek 到任意全局下标(参数区间起点)。
    """

    def __init__(self, manifest: Manifest, base_dir: Path, read_rows: int):
        self.manifest = manifest
        self.base_dir = Path(base_dir)
        self.read_rows = read_rows
        self._pos = 0
        self._shard_idx = 0
        self._shard_pos = 0
        self._mm: np.memmap | None = None
        self.seek(0)

    @property
    def pos(self) -> int:
        return self._pos

    def seek(self, global_pos: int) -> None:
        for idx, sh in enumerate(self.manifest.shards):
            if sh.start <= global_pos < sh.end or (
                global_pos == sh.end == self.manifest.total_params
            ):
                self._mm = np.load(self.base_dir / sh.file, mmap_mode="r")
                self._shard_idx = idx
                self._shard_pos = global_pos - sh.start
                self._pos = global_pos
                return
        raise ManifestError(f"全局下标 {global_pos} 不属于任何分片")

    def read(self, n: int) -> np.ndarray:
        """从当前全局位置读 n 个权重值,返回 shape (n,) float32。"""
        out = np.empty(n, dtype=DTYPE)
        got = 0
        while got < n:
            if self._shard_idx >= len(self.manifest.shards):
                raise ManifestError("源数据行数不足,读取越过检查点末尾")
            sh = self.manifest.shards[self._shard_idx]
            avail = sh.rows - self._shard_pos
            take = min(n - got, avail, self.read_rows)
            out[got:got + take] = self._mm[self._shard_pos:self._shard_pos + take, 0]
            self._shard_pos += take
            got += take
            self._pos += take
            if self._shard_pos == sh.rows and got < n:
                next_idx = self._shard_idx + 1
                self._mm = np.load(
                    self.base_dir / self.manifest.shards[next_idx].file, mmap_mode="r"
                )
                self._shard_idx = next_idx
                self._shard_pos = 0
        return out


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class TargetPlan:
    """一个目标参数区间的迁移计划。"""
    start: int
    end: int
    names: tuple[str, ...]          # 共用该区间的全部目标参数名
    old_intervals: tuple[tuple[int, int], ...]  # 命中的旧参数区间(去重排序)
    inherited: bool                 # True=继承旧权重,False=保留模板初值


def _build_plan(old: Manifest, template: Manifest) -> list[TargetPlan]:
    """按 (参数名, 形状) 为每个去重后的目标区间找旧权重来源。"""
    old_by_key: dict[tuple[str, tuple[int, ...]], ParamSpec] = {}
    for p in old.params:
        old_by_key[(p.name, p.shape)] = p

    groups: dict[tuple[int, int], list[ParamSpec]] = {}
    for p in template.params:
        groups.setdefault((p.start, p.end), []).append(p)

    plan: list[TargetPlan] = []
    for (s, e), specs in sorted(groups.items()):
        hits: set[tuple[int, int]] = set()
        for spec in specs:
            old_p = old_by_key.get((spec.name, spec.shape))
            if old_p is not None:
                hits.add((old_p.start, old_p.end))
        plan.append(TargetPlan(
            start=s, end=e,
            names=tuple(sorted(spec.name for spec in specs)),
            old_intervals=tuple(sorted(hits)),
            inherited=bool(hits),
        ))
    return plan


def _check_shared_conflicts(
    plan: list[TargetPlan], old: Manifest, old_dir: Path, read_rows: int
) -> None:
    """共享区间命中多份不同的旧权重时,逐块比较原始位值;不一致即报冲突。"""
    messages: list[str] = []
    for t in plan:
        if len(t.old_intervals) < 2:
            continue
        length = t.end - t.start
        readers = [
            WeightStreamReader(old, old_dir, read_rows) for _ in t.old_intervals
        ]
        for reader, interval in zip(readers, t.old_intervals):
            reader.seek(interval[0])
        mismatch = False
        for off in range(0, length, read_rows):
            n = min(read_rows, length - off)
            chunks = [r.read(n) for r in readers]
            base = chunks[0]
            if any(not np.array_equal(base, ch, equal_nan=True) for ch in chunks[1:]):
                mismatch = True
                break
        if mismatch:
            messages.append(
                f"目标区间 [{t.start}, {t.end}) 的共享参数 {list(t.names)} "
                f"命中多份旧权重且原始位值不一致(旧区间 {[list(x) for x in t.old_intervals]})"
            )
    if messages:
        raise ManifestError("共享权重冲突: " + "；".join(messages))


def reshard(
    old_manifest_path: Path,
    template_manifest_path: Path,
    out_dir: Path,
    read_rows: int = READ_ROWS,
    write_rows: int = WRITE_ROWS,
) -> dict:
    """按模板把旧检查点迁移为新批次,写入 out_dir,返回新清单(dict)。

    清单无效、共享权重冲突或分片写入失败时抛异常,out_dir 由调用方清理。
    """
    check_buffer_limits(read_rows, write_rows)
    old_manifest_path = Path(old_manifest_path)
    template_manifest_path = Path(template_manifest_path)
    old = load_and_validate(old_manifest_path)
    template = load_and_validate(template_manifest_path)

    plan = _build_plan(old, template)
    # 所有前置校验(含共享冲突)在创建输出目录之前完成,失败不留下任何文件
    _check_shared_conflicts(plan, old, old_manifest_path.parent, read_rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    tpl_reader = WeightStreamReader(template, template_manifest_path.parent, read_rows)
    # 每个旧区间一个独立游标(共享区间可能在不同旧位置各取一份,且访问顺序交错)
    old_readers: dict[tuple[int, int], WeightStreamReader] = {}

    def old_weights(interval: tuple[int, int], n: int) -> np.ndarray:
        reader = old_readers.get(interval)
        if reader is None:
            reader = WeightStreamReader(old, old_manifest_path.parent, read_rows)
            reader.seek(interval[0])
            old_readers[interval] = reader
        return reader.read(n)

    out_shards: list[dict] = []
    ordered = sorted(plan, key=lambda t: t.start)
    chunk_buf = np.empty((write_rows, NCOLS), dtype=DTYPE)
    try:
        for sh in template.shards:
            out_shards.append({"file": sh.file, "start": sh.start, "end": sh.end})
            writer = ShardWriter(out_dir / sh.file, sh.rows, write_rows)
            if sh.rows == 0:  # 模板出现 0 行空片(合法 .npy,仅头部)
                writer.close()
                continue
            pos = sh.start
            plan_idx = next(i for i, t in enumerate(ordered) if t.start <= pos < t.end)
            t = ordered[plan_idx]
            while pos < sh.end:
                # 参数区间可跨分片:单次取块不得越过参数区间,也不得越过当前分片
                n = min(read_rows, t.end - pos, sh.end - pos)
                if t.inherited:
                    # 计划阶段已保证多份旧权原位值一致,取第一份即可;
                    # 每个旧区间独立游标,跨分片时由 reader 自动接续
                    weights = old_weights(t.old_intervals[0], n)
                else:
                    # 保留模板初值:模板游标可能因前序继承段而落后,先对齐全局位置
                    if tpl_reader.pos != pos:
                        tpl_reader.seek(pos)
                    weights = tpl_reader.read(n)
                rows = chunk_buf[:n]
                rows[:, 0] = weights
                rows[:, 1] = 0.0  # Adam 一阶矩归零
                rows[:, 2] = 0.0  # Adam 二阶矩归零
                writer.write(rows)
                pos += n
                if pos == t.end:  # 当前参数区间写完(可能在分片边界处)
                    plan_idx += 1
                    if plan_idx < len(ordered):
                        t = ordered[plan_idx]
            writer.close()
    except BaseException:
        for f in out_dir.glob("*.tmp"):
            f.unlink(missing_ok=True)
        raise

    out_manifest = {
        "step": 0,  # 训练步归零
        "total_params": template.total_params,
        "shards": out_shards,
        "params": [
            {"name": p.name, "shape": list(p.shape), "start": p.start, "end": p.end}
            for p in template.params
        ],
    }
    mtmp = out_dir / "manifest.json.tmp"
    mtmp.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with open(mtmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.rename(mtmp, out_dir / "manifest.json")
    _fsync_dir(out_dir)
    return out_manifest
