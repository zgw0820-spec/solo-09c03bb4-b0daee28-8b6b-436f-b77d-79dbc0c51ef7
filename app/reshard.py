"""按目标进程数重分 Adam 检查点。

- 每个进程分一片,按全局下标连续划分,各片参数数量最多相差 1。
- 保留训练步与每个下标的三个原值(权重、一阶矩、二阶矩)。
- 流式读写:读块最多 READ_ROWS 行,写缓冲最多 WRITE_ROWS 行,
  合计 (READ_ROWS + WRITE_ROWS) * 12 字节 <= 64 KiB。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .manifest import (
    DTYPE,
    NCOLS,
    ManifestError,
    load_and_validate,
)

ROW_BYTES = DTYPE.itemsize * NCOLS  # 12 字节/行
MAX_BUFFER_BYTES = 64 * 1024  # 分片读写缓冲合计上限
READ_ROWS = 2048  # 读块: 2048 * 12 = 24 KiB
WRITE_ROWS = 2048  # 写缓冲: 2048 * 12 = 24 KiB,合计 48 KiB <= 64 KiB


def buffer_footprint_bytes(read_rows: int = READ_ROWS, write_rows: int = WRITE_ROWS) -> int:
    """分片读写缓冲的合计上界(字节)。"""
    return (read_rows + write_rows) * ROW_BYTES


def check_buffer_limits(read_rows: int = READ_ROWS, write_rows: int = WRITE_ROWS) -> None:
    if read_rows < 1 or write_rows < 1:
        raise ValueError("读写缓冲行数必须 >= 1")
    if buffer_footprint_bytes(read_rows, write_rows) > MAX_BUFFER_BYTES:
        raise ValueError(
            f"读写缓冲合计 {buffer_footprint_bytes(read_rows, write_rows)} 字节,"
            f"超过上限 {MAX_BUFFER_BYTES} 字节 (64 KiB)"
        )


def balanced_partition(total: int, parts: int) -> list[tuple[int, int]]:
    """把 [0, total) 连续划分成 parts 段,各段长度最多相差 1。

    前 total % parts 段各多 1 个;parts > total 时末尾出现长度为 0 的段。
    """
    if parts < 1:
        raise ValueError("进程数必须 >= 1")
    if total < 0:
        raise ValueError("total 必须 >= 0")
    base, rem = divmod(total, parts)
    ranges = []
    pos = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        ranges.append((pos, pos + size))
        pos += size
    return ranges


class ShardWriter:
    """增量写一个 .npy 分片,内存缓冲最多 buf_rows 行。

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


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def convert(
    manifest_path: Path,
    out_dir: Path,
    processes: int,
    read_rows: int = READ_ROWS,
    write_rows: int = WRITE_ROWS,
) -> dict:
    """把 manifest_path 描述的检查点重分成 processes 片,写入 out_dir。

    成功时 out_dir 内含 manifest.json 与全部分片;返回新清单(dict)。
    中途失败抛异常,out_dir 由调用方清理。
    """
    check_buffer_limits(read_rows, write_rows)
    if processes < 1:
        raise ValueError("进程数必须 >= 1")
    manifest_path = Path(manifest_path)
    m = load_and_validate(manifest_path)
    targets = balanced_partition(m.total_params, processes)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    out_shards: list[dict] = []
    t_idx = 0
    writer: ShardWriter | None = None

    def open_next() -> None:
        nonlocal t_idx, writer
        writer = None
        while t_idx < len(targets):
            s, e = targets[t_idx]
            name = f"shard_{t_idx:05d}.npy"
            out_shards.append({"file": name, "start": s, "end": e})
            t_idx += 1
            w = ShardWriter(out_dir / name, e - s, write_rows)
            if e > s:
                writer = w
                return
            w.close()  # 长度为 0 的片:只含头部

    open_next()
    try:
        for shard in m.shards:
            arr = np.load(manifest_path.parent / shard.file, mmap_mode="r")
            off = 0
            while off < shard.rows:
                chunk = np.asarray(arr[off : off + read_rows])
                off += chunk.shape[0]
                pos = 0
                while pos < chunk.shape[0]:
                    if writer is None:
                        raise ManifestError("源数据行数超过 total_params")
                    take = min(writer.remaining, chunk.shape[0] - pos)
                    writer.write(chunk[pos : pos + take])
                    pos += take
                    if writer.remaining == 0:
                        writer.close()
                        open_next()
        if writer is not None:
            writer.close()  # 行数不足时 close 会抛 IOError
        if t_idx != len(targets):
            raise ManifestError("源数据行数不足 total_params")
    except BaseException:
        # 尽力清理未完成的临时文件,正式文件由调用方整体删除
        for f in out_dir.glob("*.tmp"):
            f.unlink(missing_ok=True)
        raise

    out_manifest = {
        "step": m.step,
        "total_params": m.total_params,
        "shards": out_shards,
        "processes": processes,
    }
    mtmp = out_dir / "manifest.json.tmp"
    mtmp.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with open(mtmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.rename(mtmp, out_dir / "manifest.json")
    _fsync_dir(out_dir)
    return out_manifest
