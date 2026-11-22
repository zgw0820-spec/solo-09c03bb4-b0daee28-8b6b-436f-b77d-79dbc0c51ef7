"""批次存储测试:原子发布、写入中断、冲突不发布、重启后凭编号取旧批次。"""
import json

import numpy as np
import pytest

from app import reshard
from app.manifest import ManifestError
from app.store import BatchStore
from conftest import make_checkpoint, read_batch, weights_array

OLD_PARAMS = [
    {"name": "a.weight", "shape": [6], "start": 0, "end": 6},
    {"name": "b.bias", "shape": [4], "start": 6, "end": 10},
]
TPL_PARAMS = [
    {"name": "b.bias", "shape": [4], "start": 0, "end": 4},      # 继承旧 [6,10)
    {"name": "a.weight", "shape": [6], "start": 4, "end": 10},   # 继承旧 [0,6)
]


@pytest.fixture
def pair(tmp_path):
    old = make_checkpoint(tmp_path / "ckpts", "old", 12000, [5, 5],
                          OLD_PARAMS, weights=weights_array(10))
    tpl = make_checkpoint(tmp_path / "ckpts", "tpl", 0, [3, 7],
                          TPL_PARAMS, weights=weights_array(10, base=500.0))
    return old, tpl


class TestPublish:
    def test_publish_and_content(self, tmp_path, pair):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        meta = store.create_batch(old, tpl, source="old", template="tpl")
        assert meta["id"] == 1 and meta["step"] == 0
        assert meta["shards"] == 2 and meta["params"] == 2
        manifest, arr = read_batch(store.batch_dir(1))
        # 新位置 [0,4) 是旧 b.bias(旧下标 6..9);[4,10) 是旧 a.weight(0..5)
        np.testing.assert_array_equal(arr[0:4, 0], np.arange(6, 10, dtype="<f4"))
        np.testing.assert_array_equal(arr[4:10, 0], np.arange(0, 6, dtype="<f4"))
        assert np.all(arr[:, 1:] == 0.0)
        # 分片边界沿用模板
        assert [(s["start"], s["end"]) for s in manifest["shards"]] == [(0, 3), (3, 10)]
        assert manifest["step"] == 0 and manifest["params"] == TPL_PARAMS

    def test_ids_increase_and_old_batches_kept(self, tmp_path, pair):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        m1 = store.create_batch(old, tpl, source="old", template="tpl")
        m2 = store.create_batch(old, tpl, source="old", template="tpl")
        assert (m1["id"], m2["id"]) == (1, 2)
        assert [b["id"] for b in store.list_batches()] == [1, 2]
        for bid in (1, 2):  # 新批次发布后旧批次仍凭编号可取
            _, arr = read_batch(store.batch_dir(bid))
            np.testing.assert_array_equal(arr[0:4, 0], np.arange(6, 10, dtype="<f4"))


class TestFailedMigration:
    def test_shared_conflict_not_published(self, tmp_path):
        old_params = [
            {"name": "tied.a", "shape": [4], "start": 0, "end": 4},
            {"name": "tied.b", "shape": [4], "start": 4, "end": 8},
            {"name": "tail", "shape": [2], "start": 8, "end": 10},
        ]
        w = weights_array(10)
        w[4:8] = weights_array(4, base=100.0)  # tied.b 与 tied.a 位值不同
        old = make_checkpoint(tmp_path / "ckpts", "old", 1, [5, 5], old_params, weights=w)
        tpl_params = [
            {"name": "tied.a", "shape": [4], "start": 0, "end": 4},
            {"name": "tied.b", "shape": [4], "start": 0, "end": 4},  # 共享区间
            {"name": "tail", "shape": [6], "start": 4, "end": 10},
        ]
        tpl = make_checkpoint(tmp_path / "ckpts", "tpl", 0, [10], tpl_params,
                              weights=weights_array(10, base=900.0))
        store = BatchStore(tmp_path / "data")
        # 先发布一个成功批次,后续冲突迁移不得影响它
        ok_tpl_params = [
            {"name": "tied.a", "shape": [4], "start": 0, "end": 4},
            {"name": "tail", "shape": [6], "start": 4, "end": 10},
        ]
        ok_tpl = make_checkpoint(tmp_path / "ckpts", "ok-tpl", 0, [10], ok_tpl_params,
                                 weights=weights_array(10, base=800.0))
        meta = store.create_batch(old, ok_tpl, source="old", template="ok-tpl")
        assert meta["id"] == 1

        with pytest.raises(ManifestError, match="共享权重冲突"):
            store.create_batch(old, tpl, source="old", template="tpl")

    def test_invalid_manifest_keeps_previous_batch(self, tmp_path, pair):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        store.create_batch(old, tpl, source="old", template="tpl")
        bad = make_checkpoint(
            tmp_path / "ckpts", "bad", 0, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        data = json.loads(bad.read_text())
        data["params"][0]["shape"] = [9]
        bad.write_text(json.dumps(data) + "\n")
        with pytest.raises(Exception, match="与区间长度"):
            store.create_batch(old, bad, source="old", template="bad")
        assert [b["id"] for b in store.list_batches()] == [1]

    def test_failed_write_keeps_previous_batch(self, tmp_path, pair, monkeypatch):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        store.create_batch(old, tpl, source="old", template="tpl")  # 批次 1 成功

        orig_write = reshard.ShardWriter.write
        state = {"rows": 0}

        def flaky_write(self, rows):
            state["rows"] += rows.shape[0]
            if state["rows"] > 3:
                raise IOError("模拟磁盘写失败")
            return orig_write(self, rows)

        monkeypatch.setattr(reshard.ShardWriter, "write", flaky_write)
        with pytest.raises(IOError, match="模拟磁盘写失败"):
            store.create_batch(old, tpl, source="old", template="tpl")
        assert [b["id"] for b in store.list_batches()] == [1]
        assert not list(store.batches_dir.glob(".staging-*"))

    def test_leftover_staging_cleaned_on_startup(self, tmp_path, pair):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        store.create_batch(old, tpl, source="old", template="tpl")
        junk = store.batches_dir / ".staging-9"
        junk.mkdir()
        (junk / "shard_000.npy.tmp").write_bytes(b"partial")
        store2 = BatchStore(tmp_path / "data")  # 重启
        assert not junk.exists()
        assert [b["id"] for b in store2.list_batches()] == [1]

    def test_concurrent_migrations_get_unique_ids(self, tmp_path, pair):
        import threading

        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        results, errors = [], []

        def work():
            try:
                results.append(store.create_batch(old, tpl, source="old", template="tpl"))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert sorted(m["id"] for m in results) == [1, 2, 3, 4]
        for m in results:
            _, arr = read_batch(store.batch_dir(m["id"]))
            np.testing.assert_array_equal(arr[0:4, 0], np.arange(6, 10, dtype="<f4"))


class TestRestart:
    def test_batches_survive_restart(self, tmp_path, pair):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        store.create_batch(old, tpl, source="old", template="tpl")
        store.create_batch(old, tpl, source="old", template="tpl")

        store2 = BatchStore(tmp_path / "data")  # 模拟重启
        assert [b["id"] for b in store2.list_batches()] == [1, 2]
        _, arr = read_batch(store2.batch_dir(1))
        np.testing.assert_array_equal(arr[4:10, 0], np.arange(0, 6, dtype="<f4"))
        meta = store2.create_batch(old, tpl, source="old", template="tpl")
        assert meta["id"] == 3  # 重启后编号不冲突

    def test_registry_loss_rebuilt_from_dirs(self, tmp_path, pair):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        store.create_batch(old, tpl, source="old", template="tpl")
        store.registry_path.unlink()  # 注册表丢失

        store2 = BatchStore(tmp_path / "data")
        batches = store2.list_batches()
        assert [b["id"] for b in batches] == [1]
        assert batches[0]["step"] == 0 and batches[0]["shards"] == 2
        _, arr = read_batch(store2.batch_dir(1))
        np.testing.assert_array_equal(arr[0:4, 0], np.arange(6, 10, dtype="<f4"))

    def test_incomplete_batch_dir_not_listed(self, tmp_path, pair):
        old, tpl = pair
        store = BatchStore(tmp_path / "data")
        store.create_batch(old, tpl, source="old", template="tpl")
        victim = store.batch_dir(1)
        first = json.loads((victim / "manifest.json").read_text())["shards"][0]["file"]
        (victim / first).unlink()
        store2 = BatchStore(tmp_path / "data")
        assert store2.list_batches() == []
