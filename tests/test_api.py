"""API 端到端测试:选检查点、转换、下载、失败后旧批次仍可取。"""
import io
import json
import tarfile
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from conftest import expected_array, make_checkpoint


@pytest.fixture
def client(tmp_path):
    ckpts = tmp_path / "ckpts"
    make_checkpoint(ckpts, "ckpt-a", 12000, [334, 333, 333])
    app = create_app(checkpoints_dir=ckpts, data_dir=tmp_path / "data")
    return TestClient(app)


def wait_job(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError(f"任务未在 {timeout}s 内结束: {job_id}")


def submit(client, checkpoint="ckpt-a", processes=4):
    r = client.post("/api/convert", json={"checkpoint": checkpoint, "processes": processes})
    assert r.status_code == 202, r.text
    return wait_job(client, r.json()["job_id"])


class TestCheckpoints:
    def test_list(self, client):
        d = client.get("/api/checkpoints").json()
        assert d["checkpoints"] == [
            {"path": "ckpt-a", "valid": True, "step": 12000,
             "total_params": 1000, "shards": 3}
        ]

    def test_invalid_checkpoint_reported(self, client, tmp_path):
        make_checkpoint(tmp_path / "ckpts", "ckpt-broken", 1, [5, 5])
        (tmp_path / "ckpts" / "ckpt-broken" / "shard_000.npy").unlink()
        d = client.get("/api/checkpoints").json()
        broken = next(c for c in d["checkpoints"] if c["path"] == "ckpt-broken")
        assert broken["valid"] is False and "缺少分片文件" in broken["error"]

    def test_index_page(self, client):
        r = client.get("/")
        assert r.status_code == 200 and "Adam 检查点重分片" in r.text


class TestConvertFlow:
    def test_convert_and_download(self, client):
        job = submit(client, processes=4)
        assert job["status"] == "done" and job["batch_id"] == 1

        batches = client.get("/api/batches").json()["batches"]
        assert len(batches) == 1 and batches[0]["processes"] == 4

        manifest = client.get("/api/batches/1/manifest.json").json()
        assert manifest["step"] == 12000 and len(manifest["shards"]) == 4

        # 逐个分片下载并拼回,逐下标核对三个原值
        parts = []
        for s in manifest["shards"]:
            r = client.get(f"/api/batches/1/shards/{s['file']}")
            assert r.status_code == 200
            arr = np.load(io.BytesIO(r.content))
            assert arr.shape == (s["end"] - s["start"], 3)
            parts.append(arr)
        np.testing.assert_array_equal(np.concatenate(parts), expected_array(1000))

        # tar.gz 整包下载
        r = client.get("/api/batches/1/download")
        assert r.status_code == 200
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            names = sorted(m.name for m in tf.getmembers() if m.isfile())
        assert "batch-1/manifest.json" in names
        assert sum(1 for n in names if n.endswith(".npy")) == 4

    def test_old_batch_available_after_new_publish(self, client):
        assert submit(client, processes=2)["batch_id"] == 1
        assert submit(client, processes=5)["batch_id"] == 2
        ids = [b["id"] for b in client.get("/api/batches").json()["batches"]]
        assert ids == [1, 2]
        for bid, nshards in ((1, 2), (2, 5)):
            m = client.get(f"/api/batches/{bid}/manifest.json").json()
            assert len(m["shards"]) == nshards

    def test_missing_shard_fails_and_previous_batch_served(self, client, tmp_path):
        assert submit(client, processes=2)["status"] == "done"  # 批次 1
        # 制造缺片的检查点并发起转换
        make_checkpoint(tmp_path / "ckpts", "ckpt-bad", 7, [5, 5])
        (tmp_path / "ckpts" / "ckpt-bad" / "shard_001.npy").unlink()
        job = submit(client, checkpoint="ckpt-bad", processes=3)
        assert job["status"] == "failed" and "缺少分片文件" in job["error"]
        # 网页继续提供上次成功的批次
        ids = [b["id"] for b in client.get("/api/batches").json()["batches"]]
        assert ids == [1]
        assert client.get("/api/batches/1/download").status_code == 200

    def test_not_found_and_bad_requests(self, client):
        assert client.get("/api/batches/99").status_code == 404
        assert client.get("/api/batches/99/download").status_code == 404
        assert client.get("/api/batches/1/shards/../manifest.json").status_code in (404, 422)
        r = client.post("/api/convert", json={"checkpoint": "ckpt-a", "processes": 0})
        assert r.status_code == 422
        r = client.post("/api/convert", json={"checkpoint": "../etc", "processes": 2})
        assert r.status_code in (400, 404)
        r = client.post("/api/convert", json={"checkpoint": "nope", "processes": 2})
        assert r.status_code == 404
