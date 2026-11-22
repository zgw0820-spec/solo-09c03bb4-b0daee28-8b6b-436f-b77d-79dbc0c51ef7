"""参数迁移核心逻辑测试:按名继承、层重排、共享权重、冲突、零 Adam、缓冲上限。"""
import json

import numpy as np
import pytest

from app.manifest import ManifestError, load_and_validate
from app.reshard import (
    MAX_BUFFER_BYTES,
    READ_ROWS,
    WRITE_ROWS,
    WeightStreamReader,
    buffer_footprint_bytes,
    check_buffer_limits,
    reshard,
)
from conftest import make_checkpoint, read_batch, weights_array


# ---------- 夹具:旧检查点与新模型模板 ----------

OLD_PARAMS = [
    {"name": "embed.token", "shape": [8], "start": 0, "end": 8},
    {"name": "embed.position", "shape": [4], "start": 8, "end": 12},
    {"name": "tied.input_proj", "shape": [2, 4], "start": 12, "end": 20},
    {"name": "tied.output_proj", "shape": [2, 4], "start": 20, "end": 28},
    {"name": "head.bias", "shape": [2], "start": 28, "end": 30},
]


@pytest.fixture
def old_mpath(tmp_path):
    # output_proj 与 input_proj 位值完全相同(共享成功场景)
    w = weights_array(30)
    w[20:28] = w[12:20]
    return make_checkpoint(tmp_path / "ckpts", "old", 12000, [10, 10, 10],
                           OLD_PARAMS, weights=w)


@pytest.fixture
def old_conflict_mpath(tmp_path):
    # output_proj 与 input_proj 位值不同(冲突场景)
    w = weights_array(30)
    w[20:28] = weights_array(8, base=100.0)
    return make_checkpoint(tmp_path / "ckpts", "old-conflict", 9000, [10, 10, 10],
                           OLD_PARAMS, weights=w)


# 模板:层顺序重排、共享权重、新参数保留初值,分片边界与旧检查点不同(12/12/6)
TEMPLATE_PARAMS = [
    {"name": "head.bias", "shape": [2], "start": 0, "end": 2},
    {"name": "embed.position", "shape": [4], "start": 2, "end": 6},
    {"name": "embed.token", "shape": [8], "start": 6, "end": 14},
    {"name": "new_block.norm", "shape": [2, 3], "start": 14, "end": 20},
    {"name": "tied.input_proj", "shape": [2, 4], "start": 20, "end": 28},
    {"name": "tied.output_proj", "shape": [2, 4], "start": 20, "end": 28},
    {"name": "extra.tail", "shape": [2], "start": 28, "end": 30},
]


@pytest.fixture
def template_mpath(tmp_path):
    return make_checkpoint(
        tmp_path / "ckpts", "template", 0, [12, 12, 6], TEMPLATE_PARAMS,
        weights=weights_array(30, base=1000.0),
    )


class TestBufferLimit:
    def test_defaults_within_64kib(self):
        assert buffer_footprint_bytes() <= MAX_BUFFER_BYTES
        assert (READ_ROWS * 2 + WRITE_ROWS) * 12 <= 64 * 1024

    def test_over_limit_rejected(self):
        with pytest.raises(ValueError):
            check_buffer_limits(read_rows=4096, write_rows=4096)  # 远超 64 KiB

    def test_reshard_rejects_over_limit(self, old_mpath, template_mpath, tmp_path):
        with pytest.raises(ValueError):
            reshard(old_mpath, template_mpath, tmp_path / "out",
                    read_rows=4096, write_rows=4096)


class TestReshardSemantics:
    def test_inherit_rename_reorder_and_template_fallback(
        self, old_mpath, template_mpath, tmp_path
    ):
        out = tmp_path / "out"
        manifest = reshard(old_mpath, template_mpath, out)

        # 新批次沿用模板分片边界(文件名与区间完全一致)
        assert manifest["shards"] == [
            {"file": "shard_000.npy", "start": 0, "end": 12},
            {"file": "shard_001.npy", "start": 12, "end": 24},
            {"file": "shard_002.npy", "start": 24, "end": 30},
        ]
        # 训练步归零
        assert manifest["step"] == 0
        assert manifest["total_params"] == 30
        # 参数清单原样保留(含共享区间的两个名字)
        assert manifest["params"] == TEMPLATE_PARAMS

        loaded = load_and_validate(out / "manifest.json")  # 新清单自身合法
        _, arr = read_batch(out)
        old_w = np.load(old_mpath.parent / "shard_000.npy")  # 仅用于取值参照
        old_full = np.concatenate([
            np.load(old_mpath.parent / s["file"]) for s in
            json.loads(old_mpath.read_text())["shards"]
        ])

        def old_value(global_idx):
            return old_full[global_idx, 0]

        # 逐参数核对权重来源(下标已随层重排搬到新位置)
        # head.bias: 新 [0,2) 继承旧 [28,30)
        np.testing.assert_array_equal(arr[0:2, 0], old_full[28:30, 0])
        # embed.position: 新 [2,6) 继承旧 [8,12)
        np.testing.assert_array_equal(arr[2:6, 0], old_full[8:12, 0])
        # embed.token: 新 [6,14) 继承旧 [0,8)
        np.testing.assert_array_equal(arr[6:14, 0], old_full[0:8, 0])
        # new_block.norm: 旧清单没有 → 保留模板初值(模板全局下标 14..19 → 1014..1019)
        np.testing.assert_array_equal(arr[14:20, 0], np.arange(1014, 1020, dtype="<f4"))
        # 共享区间 [20,28): 两个名字都继承,旧 input_proj=旧 output_proj 位值
        np.testing.assert_array_equal(arr[20:28, 0], old_full[12:20, 0])
        np.testing.assert_array_equal(arr[20:28, 0], old_full[20:28, 0])
        # extra.tail: 新参数 → 模板初值(模板全局下标 28..29 → 1028..1029)
        np.testing.assert_array_equal(arr[28:30, 0], np.arange(1028, 1030, dtype="<f4"))

        # 两阶 Adam 矩全部归零(模板里原本是非零/零都不影响)
        assert np.all(arr[:, 1] == 0.0)
        assert np.all(arr[:, 2] == 0.0)
        assert arr.dtype == np.dtype("<f4")
        # 引用 old_value 避免静态检查抱怨未使用,同时语义上锁定旧列含义
        assert old_value(0) == old_w[0, 0]

    def test_adam_columns_zero_even_when_template_nonzero(self, tmp_path):
        old = make_checkpoint(
            tmp_path / "c", "old", 5, [6],
            [{"name": "p", "shape": [6], "start": 0, "end": 6}],
            weights=weights_array(6), moments=(7.0, 8.0),
        )
        # 模板的 Adam 列故意写非零
        tpl = make_checkpoint(
            tmp_path / "c", "tpl", 0, [3, 3],
            [{"name": "p", "shape": [6], "start": 0, "end": 6}],
            weights=weights_array(6, base=500.0), moments=(9.0, 9.0),
        )
        out = tmp_path / "out"
        reshard(old, tpl, out)
        _, arr = read_batch(out)
        np.testing.assert_array_equal(arr[:, 0], np.arange(6, dtype="<f4"))  # 继承旧权重
        assert np.all(arr[:, 1:] == 0.0)

    def test_shape_mismatch_falls_back_to_template(self, tmp_path):
        """同名但形状不同:不视为命中,保留模板初值。"""
        old = make_checkpoint(
            tmp_path / "c", "old", 1, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        tpl = make_checkpoint(
            tmp_path / "c", "tpl", 0, [4],
            [{"name": "p", "shape": [2, 2], "start": 0, "end": 4}],
            weights=weights_array(4, base=300.0),
        )
        out = tmp_path / "out"
        reshard(old, tpl, out)
        _, arr = read_batch(out)
        np.testing.assert_array_equal(arr[:, 0], np.arange(300, 304, dtype="<f4"))

    def test_shared_conflict_raises_and_lists_names(
        self, old_conflict_mpath, tmp_path
    ):
        tpl_params = [
            {"name": "tied.input_proj", "shape": [2, 4], "start": 0, "end": 8},
            {"name": "tied.output_proj", "shape": [2, 4], "start": 0, "end": 8},
            {"name": "embed.token", "shape": [8], "start": 8, "end": 16},
            {"name": "embed.position", "shape": [4], "start": 16, "end": 20},
            {"name": "head.bias", "shape": [2], "start": 20, "end": 22},
            {"name": "pad.fill", "shape": [8], "start": 22, "end": 30},
        ]
        tpl = make_checkpoint(
            tmp_path / "ckpts", "tpl-conflict", 0, [15, 15], tpl_params,
            weights=weights_array(30, base=1000.0),
        )
        out = tmp_path / "out"
        with pytest.raises(ManifestError, match="共享权重冲突") as ei:
            reshard(old_conflict_mpath, tpl, out)
        msg = str(ei.value)
        # 冲突信息列出两个共享参数名与目标区间
        assert "tied.input_proj" in msg and "tied.output_proj" in msg
        assert "[0, 8)" in msg
        # 冲突时不创建输出目录(前置校验先于写盘)
        assert not out.exists()

    def test_shared_conflict_with_tiny_buffers(self, old_conflict_mpath, tmp_path):
        """读块 1 行,冲突检测在大量分块下仍逐值比较、正确报错。"""
        tpl_params = [
            {"name": "tied.input_proj", "shape": [2, 4], "start": 0, "end": 8},
            {"name": "tied.output_proj", "shape": [2, 4], "start": 0, "end": 8},
            {"name": "rest", "shape": [22], "start": 8, "end": 30},
        ]
        tpl = make_checkpoint(
            tmp_path / "ckpts", "tpl", 0, [30], tpl_params,
            weights=weights_array(30, base=1000.0),
        )
        with pytest.raises(ManifestError, match="共享权重冲突"):
            reshard(old_conflict_mpath, tpl, tmp_path / "out",
                    read_rows=1, write_rows=1)

    def test_cross_shard_params_with_row_sized_buffers(
        self, old_mpath, template_mpath, tmp_path
    ):
        """1 行读块/写缓冲强制每个参数跨分片,结果仍与默认缓冲一致。"""
        out1 = tmp_path / "o1"
        out2 = tmp_path / "o2"
        m1 = reshard(old_mpath, template_mpath, out1)
        m2 = reshard(old_mpath, template_mpath, out2, read_rows=1, write_rows=1)
        assert m1["shards"] == m2["shards"]
        _, a1 = read_batch(out1)
        _, a2 = read_batch(out2)
        np.testing.assert_array_equal(a1, a2)

    def test_output_deterministic_bytes(self, old_mpath, template_mpath, tmp_path):
        a = reshard(old_mpath, template_mpath, tmp_path / "o1")
        reshard(old_mpath, template_mpath, tmp_path / "o2")
        for s in a["shards"]:
            assert (tmp_path / "o1" / s["file"]).read_bytes() == \
                   (tmp_path / "o2" / s["file"]).read_bytes()

    def test_zero_row_shard_allowed(self, tmp_path):
        """模板含 0 行空片时仍输出合法 .npy。"""
        old = make_checkpoint(
            tmp_path / "c", "old", 1, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        tpl = make_checkpoint(
            tmp_path / "c", "tpl", 0, [4, 0],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
            weights=weights_array(4, base=200.0),
        )
        # 手工把第二个分片改成 0 行空片
        m = json.loads(tpl.read_text())
        m["shards"][1] = {"file": "shard_001.npy", "start": 4, "end": 4}
        tpl.write_text(json.dumps(m) + "\n")
        np.save(tpl.parent / "shard_001.npy", np.zeros((0, 3), "<f4"))
        out = tmp_path / "out"
        manifest = reshard(old, tpl, out)
        assert [s["end"] - s["start"] for s in manifest["shards"]] == [4, 0]
        empty = np.load(out / "shard_001.npy")
        assert empty.shape == (0, 3) and empty.dtype == np.dtype("<f4")


class TestWeightStreamReader:
    def test_seek_and_cross_shard_read(self, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "c", "old", 1, [3, 3, 4],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        m = load_and_validate(mpath)
        r = WeightStreamReader(m, mpath.parent, read_rows=2)
        r.seek(7)  # 落在第三片
        np.testing.assert_array_equal(r.read(3), np.array([7, 8, 9], "<f4"))
        r.seek(0)
        np.testing.assert_array_equal(r.read(10), np.arange(10, dtype="<f4"))
