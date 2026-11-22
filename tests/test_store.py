"""批次存储测试:原子发布、写入中断、重启后凭编号取旧批次。"""
import json

import numpy as np
import pytest

from app import reshard
from app.store import BatchStore
from conftest import expected_array, make_checkpoint, read_batch


@pytest.fixture
def src_checkpoint(tmp_path):
    return make_checkpoint(tmp_path / "ckpts", "ckpt-a", 12000, [334, 333, 333])


class TestPublish:
    def test_publish_and_content(self, tmp_path, src_checkpoint):
        store = BatchStore(tmp_path / "data")
        meta = store.create_batch(src_checkpoint, 4, source="ckpt-a")
        assert meta["id"] == 1 and meta["step"] == 12000 and meta["processes"] == 4
        _, arr = read_batch(store.batch_dir(1))
        np.testing.assert_array_equal(arr, expected_array(1000))

    def test_ids_increase_and_old_batches_kept(self, tmp_path, src_checkpoint):
        store = BatchStore(tmp_path / "data")
        m1 = store.create_batch(src_checkpoint, 2, source="ckpt-a")
        m2 = store.create_batch(src_checkpoint, 5, source="ckpt-a")
        assert (m1["id"], m2["id"]) == (1, 2)
        assert [b["id"] for b in store.list_batches()] == [1, 2]
        # 新批次发布后旧批次仍凭编号可取
        _, arr1 = read_batch(store.batch_dir(1))
        _, arr2 = read_batch(store.batch_dir(2))
        np.testing.assert_array_equal(arr1, expected_array(1000))
        np.testing.assert_array_equal(arr2, expected_array(1000))
        m1j = json.loads(store.manifest_path(1).read_text())
        m2j = json.loads(store.manifest_path(2).read_text())
        assert len(m1j["shards"]) == 2 and len(m2j["shards"]) == 5


class TestInterruptedWrite:
    def test_failed_write_keeps_previous_batch(self, tmp_path, src_checkpoint, monkeypatch):
        store = BatchStore(tmp_path / "data")
        store.create_batch(src_checkpoint, 2, source="ckpt-a")  # 批次 1 成功

        # 模拟写盘失败:累计写满 100 行后抛 IOError
        orig_write = reshard.ShardWriter.write
        state = {"rows": 0}

        def flaky_write(self, rows):
            state["rows"] += rows.shape[0]
            if state["rows"] > 100:
                raise IOError("模拟磁盘写失败")
            return orig_write(self, rows)

        monkeypatch.setattr(reshard.ShardWriter, "write", flaky_write)
        with pytest.raises(IOError, match="模拟磁盘写失败"):
            store.create_batch(src_checkpoint, 7, source="ckpt-a")

        # 没有产生新批次,暂存目录已清理,旧批次完好可取
        assert [b["id"] for b in store.list_batches()] == [1]
        assert not list(store.batches_dir.glob(".staging-*"))
        assert not (store.batches_dir / "2").exists()
        _, arr = read_batch(store.batch_dir(1))
        np.testing.assert_array_equal(arr, expected_array(1000))

    def test_invalid_source_keeps_previous_batch(self, tmp_path, src_checkpoint):
        store = BatchStore(tmp_path / "data")
        store.create_batch(src_checkpoint, 2, source="ckpt-a")
        bad = make_checkpoint(tmp_path / "ckpts", "ckpt-bad", 1, [5, 5])
        (tmp_path / "ckpts" / "ckpt-bad" / "shard_001.npy").unlink()  # 缺片
        with pytest.raises(Exception, match="缺少分片文件"):
            store.create_batch(bad, 3, source="ckpt-bad")
        assert [b["id"] for b in store.list_batches()] == [1]

    def test_leftover_staging_cleaned_on_startup(self, tmp_path, src_checkpoint):
        store = BatchStore(tmp_path / "data")
        store.create_batch(src_checkpoint, 2, source="ckpt-a")
        # 模拟崩溃现场:残留的暂存目录与半成品文件
        junk = store.batches_dir / ".staging-9"
        junk.mkdir()
        (junk / "shard_00000.npy.tmp").write_bytes(b"partial")
        store2 = BatchStore(tmp_path / "data")  # 重启
        assert not junk.exists()
        assert [b["id"] for b in store2.list_batches()] == [1]


    def test_concurrent_conversions_get_unique_ids(self, tmp_path, src_checkpoint):
        """并发转换:编号各自唯一,暂存目录不冲突,两个批次都齐备。"""
        import threading

        store = BatchStore(tmp_path / "data")
        results, errors = [], []

        def work(p):
            try:
                results.append(store.create_batch(src_checkpoint, p, source="ckpt-a"))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=work, args=(p,)) for p in (2, 3, 4, 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert sorted(m["id"] for m in results) == [1, 2, 3, 4]
        for m in results:
            _, arr = read_batch(store.batch_dir(m["id"]))
            np.testing.assert_array_equal(arr, expected_array(1000))


class TestRestart:
    def test_batches_survive_restart(self, tmp_path, src_checkpoint):
        store = BatchStore(tmp_path / "data")
        store.create_batch(src_checkpoint, 3, source="ckpt-a")
        store.create_batch(src_checkpoint, 8, source="ckpt-a")

        store2 = BatchStore(tmp_path / "data")  # 模拟重启
        assert [b["id"] for b in store2.list_batches()] == [1, 2]
        _, arr = read_batch(store2.batch_dir(1))
        np.testing.assert_array_equal(arr, expected_array(1000))
        # 重启后新批次编号不冲突
        meta = store2.create_batch(src_checkpoint, 4, source="ckpt-a")
        assert meta["id"] == 3

    def test_registry_loss_rebuilt_from_dirs(self, tmp_path, src_checkpoint):
        store = BatchStore(tmp_path / "data")
        store.create_batch(src_checkpoint, 3, source="ckpt-a")
        store.registry_path.unlink()  # 注册表丢失

        store2 = BatchStore(tmp_path / "data")
        batches = store2.list_batches()
        assert [b["id"] for b in batches] == [1]
        assert batches[0]["step"] == 12000 and batches[0]["processes"] == 3
        _, arr = read_batch(store2.batch_dir(1))
        np.testing.assert_array_equal(arr, expected_array(1000))

    def test_incomplete_batch_dir_not_listed(self, tmp_path, src_checkpoint):
        store = BatchStore(tmp_path / "data")
        store.create_batch(src_checkpoint, 3, source="ckpt-a")
        # 人为删掉一个分片 -> 批次不再齐备,重启后不再出现
        victim = store.batch_dir(1)
        first = json.loads((victim / "manifest.json").read_text())["shards"][0]["file"]
        (victim / first).unlink()
        store2 = BatchStore(tmp_path / "data")
        assert store2.list_batches() == []
