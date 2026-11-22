"""重分片核心逻辑测试:划分、跨片转换、缓冲上限、校验。"""
import json

import numpy as np
import pytest

from app.manifest import ManifestError, load_and_validate
from app.reshard import (
    MAX_BUFFER_BYTES,
    READ_ROWS,
    WRITE_ROWS,
    balanced_partition,
    buffer_footprint_bytes,
    check_buffer_limits,
    convert,
)
from conftest import expected_array, make_checkpoint, read_batch


class TestPartition:
    @pytest.mark.parametrize("total,parts", [(1000, 3), (1000, 5), (7, 2), (1, 1), (10, 4)])
    def test_balanced(self, total, parts):
        ranges = balanced_partition(total, parts)
        assert len(ranges) == parts
        assert ranges[0][0] == 0 and ranges[-1][1] == total
        for (s1, e1), (s2, e2) in zip(ranges, ranges[1:]):
            assert e1 == s2  # 连续,无缺口无重叠
        sizes = [e - s for s, e in ranges]
        assert max(sizes) - min(sizes) <= 1  # 最多相差 1

    def test_more_parts_than_params(self):
        ranges = balanced_partition(3, 5)
        assert [e - s for s, e in ranges] == [1, 1, 1, 0, 0]

    def test_invalid_parts(self):
        with pytest.raises(ValueError):
            balanced_partition(10, 0)


class TestBufferLimit:
    def test_defaults_within_64kib(self):
        assert buffer_footprint_bytes() <= MAX_BUFFER_BYTES
        assert (READ_ROWS + WRITE_ROWS) * 12 <= 64 * 1024

    def test_over_limit_rejected(self):
        with pytest.raises(ValueError):
            check_buffer_limits(read_rows=4096, write_rows=4096)  # 96 KiB

    def test_convert_rejects_over_limit(self, tmp_path):
        mpath = make_checkpoint(tmp_path, "ckpt", 5, [4, 4])
        with pytest.raises(ValueError):
            convert(mpath, tmp_path / "out", 2, read_rows=4096, write_rows=4096)


class TestConvert:
    def test_cross_shard_values_preserved(self, tmp_path):
        """3 片 (334/333/333) -> 5 片,逐下标核对三个原值与训练步。"""
        mpath = make_checkpoint(tmp_path, "ckpt", 12000, [334, 333, 333])
        out = tmp_path / "out"
        manifest = convert(mpath, out, 5)

        assert manifest["step"] == 12000
        assert manifest["total_params"] == 1000
        sizes = [s["end"] - s["start"] for s in manifest["shards"]]
        assert sizes == [200, 200, 200, 200, 200]

        loaded = load_and_validate(out / "manifest.json")  # 新清单本身合法
        assert loaded.step == 12000
        _, arr = read_batch(out)
        np.testing.assert_array_equal(arr, expected_array(1000))
        assert arr.dtype == np.dtype("<f4")

    def test_uneven_split_and_chaining(self, tmp_path):
        """4 片 -> 7 片 -> 2 片,跨片边界取值不变,输出可再作输入。"""
        m1 = make_checkpoint(tmp_path / "a", "ckpt", 99, [100, 150, 60, 90])  # 共 400
        out1 = tmp_path / "b"
        man1 = convert(m1, out1, 7)
        assert [s["end"] - s["start"] for s in man1["shards"]] == [58, 57, 57, 57, 57, 57, 57]
        _, arr1 = read_batch(out1)
        np.testing.assert_array_equal(arr1, expected_array(400))

        out2 = tmp_path / "c"
        man2 = convert(out1 / "manifest.json", out2, 2)  # 链式:输出再作输入
        assert [s["end"] - s["start"] for s in man2["shards"]] == [200, 200]
        _, arr2 = read_batch(out2)
        np.testing.assert_array_equal(arr2, expected_array(400))
        assert man2["step"] == 99

    def test_tiny_chunks_still_correct(self, tmp_path):
        """读块/写缓冲设为 1 行,强制大量跨片边界,结果仍一致。"""
        mpath = make_checkpoint(tmp_path, "ckpt", 7, [3, 5, 2])
        out = tmp_path / "out"
        convert(mpath, out, 4, read_rows=1, write_rows=1)
        _, arr = read_batch(out)
        np.testing.assert_array_equal(arr, expected_array(10))

    def test_zero_row_shards(self, tmp_path):
        """进程数多于参数数:末尾出现 0 行分片,仍是合法 .npy。"""
        mpath = make_checkpoint(tmp_path, "ckpt", 3, [2, 2])
        out = tmp_path / "out"
        manifest = convert(mpath, out, 6)
        assert [s["end"] - s["start"] for s in manifest["shards"]] == [1, 1, 1, 1, 0, 0]
        empty = np.load(out / manifest["shards"][4]["file"])
        assert empty.shape == (0, 3) and empty.dtype == np.dtype("<f4")
        _, arr = read_batch(out)
        np.testing.assert_array_equal(arr, expected_array(4))

    def test_output_is_deterministic_bytes(self, tmp_path):
        mpath = make_checkpoint(tmp_path, "ckpt", 1, [6, 6])
        a = convert(mpath, tmp_path / "o1", 3)
        b = convert(mpath, tmp_path / "o2", 3)
        for sa, sb in zip(a["shards"], b["shards"]):
            assert (tmp_path / "o1" / sa["file"]).read_bytes() == \
                   (tmp_path / "o2" / sb["file"]).read_bytes()


class TestValidation:
    def _write_manifest(self, tmp_path, shards, total=10, step=5):
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(
            {"step": step, "total_params": total, "shards": shards}))
        return mpath

    def test_gap(self, tmp_path):
        mpath = self._write_manifest(tmp_path, [
            {"file": "a.npy", "start": 0, "end": 5},
            {"file": "b.npy", "start": 7, "end": 10},  # [5,7) 缺口
        ])
        with pytest.raises(ManifestError, match="缺口"):
            load_and_validate(mpath)

    def test_overlap(self, tmp_path):
        mpath = self._write_manifest(tmp_path, [
            {"file": "a.npy", "start": 0, "end": 6},
            {"file": "b.npy", "start": 4, "end": 10},  # 与 [0,6) 重叠
        ])
        with pytest.raises(ManifestError, match="重叠"):
            load_and_validate(mpath)

    def test_total_mismatch(self, tmp_path):
        mpath = self._write_manifest(tmp_path, [
            {"file": "a.npy", "start": 0, "end": 8},
        ], total=10)
        with pytest.raises(ManifestError, match="total_params"):
            load_and_validate(mpath)

    def test_missing_file(self, tmp_path):
        mpath = self._write_manifest(tmp_path, [
            {"file": "nope.npy", "start": 0, "end": 10},
        ])
        with pytest.raises(ManifestError, match="缺少分片文件"):
            load_and_validate(mpath)

    def test_shape_mismatch(self, tmp_path):
        np.save(tmp_path / "a.npy", np.zeros((9, 3), "<f4"))  # 清单说 10 行
        mpath = self._write_manifest(tmp_path, [
            {"file": "a.npy", "start": 0, "end": 10},
        ])
        with pytest.raises(ManifestError, match="形状"):
            load_and_validate(mpath)

    def test_dtype_mismatch(self, tmp_path):
        np.save(tmp_path / "a.npy", np.zeros((10, 3), np.float64))
        mpath = self._write_manifest(tmp_path, [
            {"file": "a.npy", "start": 0, "end": 10},
        ])
        with pytest.raises(ManifestError, match="float32"):
            load_and_validate(mpath)

    def test_two_columns_rejected(self, tmp_path):
        np.save(tmp_path / "a.npy", np.zeros((10, 2), "<f4"))
        mpath = self._write_manifest(tmp_path, [
            {"file": "a.npy", "start": 0, "end": 10},
        ])
        with pytest.raises(ManifestError, match="形状"):
            load_and_validate(mpath)

    def test_truncated_file_rejected(self, tmp_path):
        """文件被截断(字节数不足清单声明的行数)时拒绝。"""
        np.save(tmp_path / "a.npy", np.zeros((10, 3), "<f4"))
        with open(tmp_path / "a.npy", "r+b") as f:
            f.truncate(f.seek(0, 2) - 5 * 12)  # 砍掉 5 行数据
        mpath = self._write_manifest(tmp_path, [
            {"file": "a.npy", "start": 0, "end": 10},
        ])
        with pytest.raises(ManifestError, match="无法读取分片"):
            load_and_validate(mpath)

    def test_path_traversal_rejected(self, tmp_path):
        mpath = self._write_manifest(tmp_path, [
            {"file": "../escape.npy", "start": 0, "end": 10},
        ])
        with pytest.raises(ManifestError, match="非法分片路径"):
            load_and_validate(mpath)
