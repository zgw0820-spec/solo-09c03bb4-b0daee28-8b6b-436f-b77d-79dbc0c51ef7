"""清单校验测试:params 唯一性、形状/区间一致性、共享区间、覆盖性、分片文件。"""
import json

import numpy as np
import pytest

from app.manifest import ManifestError, load_and_validate, load_manifest
from conftest import make_checkpoint


def _reload_manifest(mpath):
    return json.loads(mpath.read_text(encoding="utf-8"))


def _write(mpath, data):
    mpath.write_text(json.dumps(data) + "\n", encoding="utf-8")


class TestValidManifests:
    def test_basic_valid(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 3, [4, 6],
            [{"name": "a", "shape": [10], "start": 0, "end": 10}],
        )
        m = load_and_validate(mpath)
        assert m.total_params == 10 and len(m.shards) == 2 and len(m.params) == 1
        assert m.params[0].shape == (10,)

    def test_shared_interval_and_cross_shard(self, tmp_path):
        # 同一区间两个参数名(共享),且参数区间跨两个分片
        params = [
            {"name": "p.a", "shape": [2, 3], "start": 2, "end": 8},
            {"name": "p.b", "shape": [2, 3], "start": 2, "end": 8},
            {"name": "q", "shape": [2], "start": 0, "end": 2},
            {"name": "r", "shape": [2], "start": 8, "end": 10},
        ]
        mpath = make_checkpoint(tmp_path / "c", "m", 0, [5, 5], params)
        m = load_and_validate(mpath)
        assert {p.name for p in m.params} == {"p.a", "p.b", "q", "r"}


class TestParamValidation:
    def test_duplicate_param_name_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [4],
            [
                {"name": "p", "shape": [2], "start": 0, "end": 2},
                {"name": "p", "shape": [2], "start": 2, "end": 4},
            ],
        )
        with pytest.raises(ManifestError, match="参数名重复"):
            load_manifest(mpath)

    def test_shape_length_mismatch_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        data = _reload_manifest(mpath)
        data["params"][0]["shape"] = [3]
        _write(mpath, data)
        with pytest.raises(ManifestError, match="与区间长度"):
            load_manifest(mpath)

    def test_multidim_shape_product_ok(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [6],
            [{"name": "p", "shape": [2, 3], "start": 0, "end": 6}],
        )
        assert load_manifest(mpath).params[0].shape == (2, 3)

    def test_empty_shape_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        data = _reload_manifest(mpath)
        data["params"][0]["shape"] = []
        _write(mpath, data)
        with pytest.raises(ManifestError, match="shape"):
            load_manifest(mpath)

    def test_missing_params_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        data = _reload_manifest(mpath)
        del data["params"]
        _write(mpath, data)
        with pytest.raises(ManifestError, match="params"):
            load_manifest(mpath)

    def test_param_gap_rejected(self, tmp_path):
        # 去重后 [0,2) 与 [4,6),缺 [2,4)
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [6],
            [
                {"name": "a", "shape": [2], "start": 0, "end": 2},
                {"name": "b", "shape": [2], "start": 4, "end": 6},
            ],
        )
        with pytest.raises(ManifestError, match="缺口"):
            load_manifest(mpath)

    def test_param_overlap_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [6],
            [
                {"name": "a", "shape": [4], "start": 0, "end": 4},
                {"name": "b", "shape": [4], "start": 2, "end": 6},
            ],
        )
        with pytest.raises(ManifestError, match="重叠"):
            load_manifest(mpath)

    def test_shared_interval_different_shape_rejected(self, tmp_path):
        # 同区间但形状乘积不同 → 第二个形状与区间长度不符
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [4],
            [
                {"name": "a", "shape": [4], "start": 0, "end": 4},
                {"name": "b", "shape": [4], "start": 0, "end": 4},
            ],
        )
        data = _reload_manifest(mpath)
        data["params"][1]["shape"] = [2, 3]  # 乘积 6 != 4
        _write(mpath, data)
        with pytest.raises(ManifestError, match="与区间长度"):
            load_manifest(mpath)

    def test_param_range_beyond_total_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        data = _reload_manifest(mpath)
        data["params"][0]["end"] = 5
        _write(mpath, data)
        with pytest.raises(ManifestError):
            load_manifest(mpath)


class TestShardValidation:
    def test_shard_gap(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        data = _reload_manifest(mpath)
        data["shards"][1]["start"] = 7  # [5,7) 缺口
        _write(mpath, data)
        with pytest.raises(ManifestError, match="缺口"):
            load_manifest(mpath)

    def test_shard_overlap(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        data = _reload_manifest(mpath)
        data["shards"][1]["start"] = 3
        _write(mpath, data)
        with pytest.raises(ManifestError, match="重叠"):
            load_manifest(mpath)

    def test_duplicate_shard_file_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        data = _reload_manifest(mpath)
        data["shards"][1]["file"] = data["shards"][0]["file"]
        _write(mpath, data)
        with pytest.raises(ManifestError, match="分片文件名重复"):
            load_manifest(mpath)

    def test_missing_file(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        (mpath.parent / "shard_001.npy").unlink()
        with pytest.raises(ManifestError, match="缺少分片文件"):
            load_and_validate(mpath)

    def test_shape_mismatch(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        np.save(mpath.parent / "shard_000.npy", np.zeros((4, 3), "<f4"))  # 清单说 5 行
        with pytest.raises(ManifestError, match="形状"):
            load_and_validate(mpath)

    def test_dtype_mismatch(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        np.save(mpath.parent / "shard_000.npy", np.zeros((5, 3), np.float64))
        with pytest.raises(ManifestError, match="float32"):
            load_and_validate(mpath)

    def test_truncated_file_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [10],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        with open(mpath.parent / "shard_000.npy", "r+b") as f:
            f.truncate(f.seek(0, 2) - 5 * 12)
        with pytest.raises(ManifestError, match="无法读取分片"):
            load_and_validate(mpath)

    def test_path_traversal_rejected(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 0, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        data = _reload_manifest(mpath)
        data["shards"][0]["file"] = "../escape.npy"
        _write(mpath, data)
        with pytest.raises(ManifestError, match="非法分片路径"):
            load_manifest(mpath)

    def test_bad_step_and_total(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "m", 3, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        data = _reload_manifest(mpath)
        data["step"] = -1
        _write(mpath, data)
        with pytest.raises(ManifestError, match="step"):
            load_manifest(mpath)

        data = _reload_manifest(mpath)
        data["step"] = 0
        data["total_params"] = 0
        _write(mpath, data)
        with pytest.raises(ManifestError, match="total_params"):
            load_manifest(mpath)
