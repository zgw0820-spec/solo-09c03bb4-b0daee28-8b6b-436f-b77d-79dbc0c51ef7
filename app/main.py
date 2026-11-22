"""FastAPI 网页:从挂载目录选检查点、填进程数,重分 Adam 检查点并发布批次。"""
from __future__ import annotations

import io
import itertools
import json
import os
import tarfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .manifest import ManifestError, load_and_validate
from .store import BatchStore

INDEX_HTML = Path(__file__).with_name("index.html").read_text(encoding="utf-8")

MAX_CHECKPOINTS_LISTED = 200
MAX_JOBS_KEPT = 100


class ConvertRequest(BaseModel):
    checkpoint: str = Field(..., description="挂载目录内检查点目录的相对路径")
    processes: int = Field(..., ge=1, description="目标进程数(每进程一片)")


def create_app(checkpoints_dir: str | Path | None = None,
               data_dir: str | Path | None = None) -> FastAPI:
    ckpts = Path(checkpoints_dir or os.environ.get("CHECKPOINTS_DIR", "/checkpoints"))
    store = BatchStore(data_dir or os.environ.get("DATA_DIR", "/data"))

    app = FastAPI(title="Adam 检查点重分片")
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()
    job_counter = itertools.count(1)

    # ---------- 网页 ----------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    # ---------- 源检查点 ----------

    @app.get("/api/checkpoints")
    def list_checkpoints() -> dict:
        out = []
        base = ckpts.resolve()
        if base.is_dir():
            for mp in sorted(base.rglob("manifest.json"))[:MAX_CHECKPOINTS_LISTED]:
                rel = str(mp.parent.relative_to(base))
                try:
                    m = load_and_validate(mp)
                    out.append({
                        "path": rel, "valid": True, "step": m.step,
                        "total_params": m.total_params, "shards": len(m.shards),
                    })
                except (ManifestError, OSError) as e:
                    out.append({"path": rel, "valid": False, "error": str(e)})
        return {"checkpoints_dir": str(base), "checkpoints": out}

    # ---------- 转换任务 ----------

    @app.post("/api/convert", status_code=202)
    def start_convert(req: ConvertRequest) -> dict:
        base = ckpts.resolve()
        target = (base / req.checkpoint).resolve()
        if not target.is_relative_to(base):
            raise HTTPException(400, "checkpoint 必须位于挂载目录内")
        if target.is_dir():
            target = target / "manifest.json"
        if not target.is_file():
            raise HTTPException(404, f"找不到清单: {req.checkpoint}")

        job_id = f"job-{next(job_counter)}"
        with jobs_lock:
            while len(jobs) >= MAX_JOBS_KEPT:  # 淘汰最旧的任务记录
                jobs.pop(next(iter(jobs)))
            jobs[job_id] = {
                "id": job_id, "status": "running", "checkpoint": req.checkpoint,
                "processes": req.processes, "batch_id": None, "error": None,
            }

        def run() -> None:
            try:
                meta = store.create_batch(target, req.processes, source=req.checkpoint)
                with jobs_lock:
                    jobs[job_id].update(status="done", batch_id=meta["id"])
            except Exception as e:  # 转换失败不影响既有批次
                with jobs_lock:
                    jobs[job_id].update(status="failed", error=str(e))

        threading.Thread(target=run, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        with jobs_lock:
            return {"jobs": list(reversed(list(jobs.values())))}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在(可能服务已重启)")
        return job

    # ---------- 批次 ----------

    @app.get("/api/batches")
    def list_batches() -> dict:
        return {"batches": store.list_batches()}

    @app.get("/api/batches/{batch_id}")
    def batch_detail(batch_id: int) -> dict:
        mpath = store.manifest_path(batch_id)
        if mpath is None:
            raise HTTPException(404, "批次不存在")
        return {"meta": store.get(batch_id),
                "manifest": json.loads(mpath.read_text(encoding="utf-8"))}

    @app.get("/api/batches/{batch_id}/manifest.json")
    def batch_manifest(batch_id: int) -> FileResponse:
        mpath = store.manifest_path(batch_id)
        if mpath is None:
            raise HTTPException(404, "批次不存在")
        return FileResponse(mpath, filename="manifest.json")

    @app.get("/api/batches/{batch_id}/shards/{name}")
    def batch_shard(batch_id: int, name: str) -> FileResponse:
        path = store.shard_path(batch_id, name)
        if path is None:
            raise HTTPException(404, "分片不存在")
        return FileResponse(path, filename=name)

    @app.get("/api/batches/{batch_id}/download")
    def batch_download(batch_id: int) -> StreamingResponse:
        if store.get(batch_id) is None:
            raise HTTPException(404, "批次不存在")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(store.batch_dir(batch_id), arcname=f"batch-{batch_id}")
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/gzip",
            headers={"Content-Disposition": f"attachment; filename=adam-batch-{batch_id}.tar.gz"},
        )

    return app


app = create_app()
