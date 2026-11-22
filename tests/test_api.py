"""API 端到端测试:选旧检查点与模板、迁移、下载、失败后旧批次仍可取。"""
import io
import json
import tarfile
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from conftest import make_checkpoint, weights_array

OLD_PARAMS = [
    {"name": "embed.token", "shape": [8], "start": 0, "end": 8},
    {"name": "embed.position", "shape": [4], "start": 8, "end": 12},
    {"name": "tied.input_proj", "shape": [2, 4], "start": 12, "end": 20},
    {"name": "tied.output_proj", "shape": [2, 4], "start": 20, "end": 28},
    {"name": "head.bias", "shape": [2], "start": 28, "end": 30},
]
TPL_PARAMS = [
    {"name": "head.bias", "shape": [2], "start": 0, "end": 2},
    {"name": "embed.position", "shape": [4], "start": 2, "end": 6},
    {"name": "embed.token", "shape": [8], "start": 6, "end": 14},
    {"name": "new.norm", "shape": [2, 3], "start": 14, "end": 20},
    {"name": "tied.input_proj", "shape": [2, 4], "start": 20, "end": 28},
    {"name": "tied.output_proj", "shape": [2, 4], "start": 20, "end": 28},
    {"name": "extra", "shape": [2], "start": 28, "end": 30},
]


def _make_sources(ckpts: "Path"):
    w = weights_array(30)
    w[20:28] = w[12:20]  # 共享来源位值一致
    old = make_checkpoint(ckpts, "ckpt-a", 12000, [10, 10, 10], OLD_PARAMS, weights=w)
    tpl = make_checkpoint(ckpts, "tpl-a", 0, [12, 12, 6], TPL_PARAMS,
                          weights=weights_array(30, base=1000.0))
    return old, tpl


@pytest.fixture
def client(tmp_path):
    ckpts = tmp_path / "ckpts"
    _make_sources(ckpts)
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


def submit(client, checkpoint="ckpt-a", template="tpl-a"):
    r = client.post("/api/convert",
                    json={"checkpoint": checkpoint, "template": template})
    assert r.status_code == 202, r.text
    return wait_job(client, r.json()["job_id"])


class TestCheckpoints:
    def test_list(self, client):
        d = client.get("/api/checkpoints").json()
        assert d["checkpoints"] == [
            {"path": "ckpt-a", "valid": True, "step": 12000,
             "total_params": 30, "shards": 3, "params": 5},
            {"path": "tpl-a", "valid": True, "step": 0,
             "total_params": 30, "shards": 3, "params": 7},
        ]

    def test_invalid_manifest_reported(self, client, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "ckpts", "broken", 0, [4],
            [{"name": "p", "shape": [4], "start": 0, "end": 4}],
        )
        (mpath.parent / "shard_000.npy").unlink()
        d = client.get("/api/checkpoints").json()
        broken = next(c for c in d["checkpoints"] if c["path"] == "broken")
        assert broken["valid"] is False and "缺少分片文件" in broken["error"]

    def test_index_page(self, client):
        r = client.get("/")
        assert r.status_code == 200 and "参数迁移" in r.text
        assert "新模型模板" in r.text


class TestConvertFlow:
    def test_convert_and_download(self, client):
        job = submit(client)
        assert job["status"] == "done" and job["batch_id"] == 1

        batches = client.get("/api/batches").json()["batches"]
        assert len(batches) == 1
        assert batches[0]["shards"] == 3 and batches[0]["params"] == 7
        assert batches[0]["source"] == "ckpt-a" and batches[0]["template"] == "tpl-a"

        manifest = client.get("/api/batches/1/manifest.json").json()
        assert manifest["step"] == 0 and len(manifest["shards"]) == 3
        assert [(s["start"], s["end"]) for s in manifest["shards"]] == [
            (0, 12), (12, 24), (24, 30)
        ]

        # 逐分片下载并拼回,核对权重来源与 Adam 归零
        parts = []
        for s in manifest["shards"]:
            r = client.get(f"/api/batches/1/shards/{s['file']}")
            assert r.status_code == 200
            arr = np.load(io.BytesIO(r.content))
            assert arr.shape == (s["end"] - s["start"], 3)
            parts.append(arr)
        full = np.concatenate(parts)
        # 新 [6,14) 继承旧 embed.token [0,8)
        np.testing.assert_array_equal(full[6:14, 0], np.arange(8, dtype="<f4"))
        # 新 [14,20) 是旧清单没有的新参数 → 模板初值 1014..1019
        np.testing.assert_array_equal(full[14:20, 0], np.arange(1014, 1020, dtype="<f4"))
        # 共享区间继承,Adam 两列全零
        np.testing.assert_array_equal(full[20:28, 0], np.arange(12, 20, dtype="<f4"))
        assert np.all(full[:, 1:] == 0.0)

        # tar.gz 整包下载
        r = client.get("/api/batches/1/download")
        assert r.status_code == 200
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
            names = sorted(m.name for m in tf.getmembers() if m.isfile())
        assert "batch-1/manifest.json" in names
        assert sum(1 for n in names if n.endswith(".npy")) == 3

    def test_old_batch_available_after_new_publish(self, client):
        assert submit(client)["batch_id"] == 1
        assert submit(client)["batch_id"] == 2
        ids = [b["id"] for b in client.get("/api/batches").json()["batches"]]
        assert ids == [1, 2]
        for bid in (1, 2):
            m = client.get(f"/api/batches/{bid}/manifest.json").json()
            assert len(m["shards"]) == 3 and m["step"] == 0

    def test_shared_conflict_fails_and_reason_shown(self, client, tmp_path):
        # 旧检查点里共享来源位值不同
        w = weights_array(30)
        w[20:28] = weights_array(8, base=100.0)
        make_checkpoint(tmp_path / "ckpts", "ckpt-conflict", 9000, [10, 10, 10],
                        OLD_PARAMS, weights=w)
        # 先成功发布批次 1
        assert submit(client)["status"] == "done"
        job = submit(client, checkpoint="ckpt-conflict", template="tpl-a")
        assert job["status"] == "failed"
        assert "共享权重冲突" in job["error"]
        assert "tied.input_proj" in job["error"] and "tied.output_proj" in job["error"]
        # 网页继续提供上次成功的批次
        ids = [b["id"] for b in client.get("/api/batches").json()["batches"]]
        assert ids == [1]
        assert client.get("/api/batches/1/download").status_code == 200

    def test_invalid_template_fails_and_reason_shown(self, client, tmp_path):
        mpath = make_checkpoint(
            tmp_path / "ckpts", "tpl-bad", 0, [5, 5],
            [
                {"name": "a", "shape": [6], "start": 0, "end": 6},
                {"name": "b", "shape": [4], "start": 6, "end": 10},
            ],
        )
        data = json.loads(mpath.read_text())
        data["params"][1]["shape"] = [3]  # 与区间长度 4 不符
        mpath.write_text(json.dumps(data) + "\n")
        job = submit(client, template="tpl-bad")
        assert job["status"] == "failed" and "与区间长度" in job["error"]
        assert client.get("/api/batches").json()["batches"] == []

    def test_missing_shard_fails_and_previous_batch_served(self, client, tmp_path):
        assert submit(client)["status"] == "done"
        mpath = make_checkpoint(
            tmp_path / "ckpts", "bad", 1, [5, 5],
            [{"name": "p", "shape": [10], "start": 0, "end": 10}],
        )
        (mpath.parent / "shard_001.npy").unlink()
        job = submit(client, checkpoint="bad", template="tpl-a")
        assert job["status"] == "failed" and "缺少分片文件" in job["error"]
        ids = [b["id"] for b in client.get("/api/batches").json()["batches"]]
        assert ids == [1]
        assert client.get("/api/batches/1/download").status_code == 200

    def test_not_found_and_bad_requests(self, client):
        assert client.get("/api/batches/99").status_code == 404
        assert client.get("/api/batches/99/download").status_code == 404
        assert client.get("/api/batches/1/shards/../manifest.json").status_code in (404, 422)
        # 缺字段
        r = client.post("/api/convert", json={"checkpoint": "ckpt-a"})
        assert r.status_code == 422
        # 同一份清单不能同时作为旧检查点与模板
        r = client.post("/api/convert",
                        json={"checkpoint": "ckpt-a", "template": "ckpt-a"})
        assert r.status_code == 400
        # 越界路径
        r = client.post("/api/convert",
                        json={"checkpoint": "../etc", "template": "tpl-a"})
        assert r.status_code in (400, 404)
        # 不存在
        r = client.post("/api/convert",
                        json={"checkpoint": "nope", "template": "tpl-a"})
        assert r.status_code == 404
