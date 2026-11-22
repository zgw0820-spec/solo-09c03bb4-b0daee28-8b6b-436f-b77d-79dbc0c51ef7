"""批次存储:原子发布、编号、注册表持久化与崩溃恢复。

目录布局(data_dir):
  batches/<id>/manifest.json   新清单(含 params 参数清单与新分片边界)
  batches/<id>/shard_*.npy     新分片(边界与模板一致)
  batches/.staging-<id>/       转换中的暂存目录(启动时清理)
  registry.json                {"next_id": N, "batches": {"<id>": meta}}

发布流程:迁移写入暂存目录 -> 全部文件齐备并 fsync -> os.rename 成
batches/<id> -> 更新注册表。任何一步中断都不会产生"半个批次",
网页始终只看到上次成功发布的批次;重启后旧批次仍凭编号可取。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from .manifest import ManifestError, load_and_validate
from .reshard import reshard


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class BatchStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.batches_dir = self.data_dir / "batches"
        self.registry_path = self.data_dir / "registry.json"
        self._lock = threading.Lock()
        self._batches: dict[str, dict] = {}
        self._next_id = 1
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self._recover()

    # ---------- 启动恢复 ----------

    def _recover(self) -> None:
        # 清理上次中断留下的暂存目录
        for d in self.batches_dir.iterdir():
            if d.is_dir() and d.name.startswith(".staging-"):
                shutil.rmtree(d, ignore_errors=True)

        if self.registry_path.is_file():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                self._next_id = int(data.get("next_id", 1))
                for bid, meta in data.get("batches", {}).items():
                    if self._batch_complete(self.batches_dir / bid):
                        self._batches[str(bid)] = meta
            except (json.JSONDecodeError, ValueError, OSError):
                # 注册表损坏则从目录重建
                self._batches = {}

        # 扫描注册表之外、已齐备的批次目录(例如注册表写入前崩溃)
        for d in sorted(self.batches_dir.iterdir()):
            if not (d.is_dir() and d.name.isdigit()) or d.name in self._batches:
                continue
            meta = self._meta_from_dir(d)
            if meta is not None:
                self._batches[d.name] = meta

        if self._batches:
            self._next_id = max(self._next_id, max(int(b) for b in self._batches) + 1)
        self._save_registry()

    def _batch_complete(self, bdir: Path) -> bool:
        try:
            load_and_validate(bdir / "manifest.json")
            return True
        except (ManifestError, OSError):
            return False

    def _meta_from_dir(self, bdir: Path) -> dict | None:
        if not self._batch_complete(bdir):
            return None
        m = load_and_validate(bdir / "manifest.json")
        created = datetime.fromtimestamp(
            (bdir / "manifest.json").stat().st_mtime, tz=timezone.utc
        ).isoformat()
        return {
            "id": int(bdir.name),
            "created_at": created,
            "step": m.step,
            "total_params": m.total_params,
            "shards": len(m.shards),
            "params": len(m.params),
            "source": "(重启后从目录恢复)",
            "template": "(重启后从目录恢复)",
        }

    def _save_registry(self) -> None:
        tmp = self.data_dir / "registry.json.tmp"
        tmp.write_text(
            json.dumps(
                {"next_id": self._next_id, "batches": self._batches},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.rename(tmp, self.registry_path)
        _fsync_dir(self.data_dir)

    # ---------- 查询 ----------

    def list_batches(self) -> list[dict]:
        with self._lock:
            return sorted(self._batches.values(), key=lambda m: m["id"])

    def get(self, batch_id: int) -> dict | None:
        with self._lock:
            return self._batches.get(str(batch_id))

    def batch_dir(self, batch_id: int) -> Path:
        return self.batches_dir / str(batch_id)

    def manifest_path(self, batch_id: int) -> Path | None:
        if self.get(batch_id) is None:
            return None
        return self.batch_dir(batch_id) / "manifest.json"

    def shard_path(self, batch_id: int, name: str) -> Path | None:
        """name 必须是该批次清单中登记的分片文件名。"""
        mpath = self.manifest_path(batch_id)
        if mpath is None:
            return None
        m = load_and_validate(mpath)
        for sh in m.shards:
            if sh.file == name:
                p = self.batch_dir(batch_id) / name
                return p if p.is_file() else None
        return None

    # ---------- 发布 ----------

    def create_batch(
        self,
        old_manifest: Path,
        template_manifest: Path,
        source: str,
        template: str,
    ) -> dict:
        """按模板迁移旧检查点并原子发布一个批次;失败清理暂存,不影响既有批次。

        编号在转换前即消耗掉,保证并发迁移各自独占暂存目录;
        失败的编号会留空,不影响旧批次凭编号取用。
        """
        with self._lock:
            batch_id = self._next_id
            self._next_id += 1
        staging = self.batches_dir / f".staging-{batch_id}"
        final = self.batches_dir / str(batch_id)
        try:
            out_manifest = reshard(old_manifest, template_manifest, staging)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        os.rename(staging, final)  # 同一文件系统内的原子发布
        _fsync_dir(self.batches_dir)

        meta = {
            "id": batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step": out_manifest["step"],
            "total_params": out_manifest["total_params"],
            "shards": len(out_manifest["shards"]),
            "params": len(out_manifest["params"]),
            "source": source,
            "template": template,
        }
        with self._lock:
            self._batches[str(batch_id)] = meta
            self._save_registry()
        return meta
