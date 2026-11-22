#!/usr/bin/env python3
"""生成网页演示用的检查点 / 模板样例。

用法: python3 scripts/make_sample.py [输出目录]
默认写到 sample_data/checkpoints/。

清单含 params(参数名、形状、全局下标区间),分片仍是 (N,3) float32 的 .npy,
三列依次为权重、Adam 一阶矩、Adam 二阶矩。

样例一览:
  ckpt-demo/          旧检查点(步 12000,30 下标,3 片),含一对位值完全相同的
                      共享来源参数 tied.input_proj / tied.output_proj
  ckpt-conflict/      旧检查点:同名参数 tied.output_proj 的位值与 ckpt-demo 不同,
                      配合 template-conflict 触发共享权重冲突
  ckpt-tiny/          小型旧检查点(步 30,10 下标,2 片)
  template-demo/      新模型模板:层顺序重排 + 共享权重(两份旧权原位值一致,继承成功)
                      + 旧清单没有的新参数(保留模板初值),分片边界与旧检查点不同
  template-conflict/  共享权重冲突模板:同一目标区间的两个名字分别命中位值不同的
                      旧权重,迁移必须失败并列出冲突参数名
  template-invalid/   无效清单:参数形状与区间长度不符,网页展示失败原因
"""
import json
import sys
from pathlib import Path

import numpy as np


def write_checkpoint(
    root: Path,
    name: str,
    step: int,
    shard_sizes: list[int],
    params: list[dict],
    weights: np.ndarray,
    moments: tuple[float, float] | None = None,
) -> Path:
    """按给定参数清单与权重写一份检查点,返回 manifest.json 路径。

    weights: 长度 total_params 的权重值(下标 i 对应全局下标 i)。
    分片按 shard_sizes 切,可与参数区间边界不对齐(区间可跨分片)。
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    total = int(sum(shard_sizes))
    if weights.shape != (total,):
        raise ValueError(f"{name}: 权重数组长度 {weights.shape} 与分片合计 {total} 不符")
    m1 = np.full(total, moments[0] if moments else 0.0, dtype="<f4")
    m2 = np.full(total, moments[1] if moments else 0.0, dtype="<f4")
    full = np.stack([weights.astype("<f4"), m1, m2], axis=1)

    shards = []
    pos = 0
    for i, n in enumerate(shard_sizes):
        fname = f"shard_{i:03d}.npy"
        np.save(d / fname, full[pos:pos + n])
        shards.append({"file": fname, "start": pos, "end": pos + n})
        pos += n

    manifest = {
        "step": step,
        "total_params": total,
        "shards": shards,
        "params": params,
    }
    mpath = d / "manifest.json"
    mpath.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"已生成 {d} (训练步 {step}, {total} 下标, {len(shard_sizes)} 片, {len(params)} 参数)")
    return mpath


def main(root: Path) -> None:
    # ---------- 旧检查点 ckpt-demo:30 下标,3 片 (10/10/10),区间跨分片 ----------
    # tied.input_proj [12,20) 与 tied.output_proj [20,28) 权重位值完全相同,
    # 供共享权重成功模板 template-demo 使用。
    old_weights = np.arange(30, dtype="<f4")
    old_weights[20:28] = old_weights[12:20]  # output_proj 复制 input_proj 的位值
    old_params = [
        {"name": "embed.token", "shape": [8], "start": 0, "end": 8},
        {"name": "embed.position", "shape": [4], "start": 8, "end": 12},
        {"name": "tied.input_proj", "shape": [2, 4], "start": 12, "end": 20},
        {"name": "tied.output_proj", "shape": [2, 4], "start": 20, "end": 28},
        {"name": "head.bias", "shape": [2], "start": 28, "end": 30},
    ]
    write_checkpoint(
        root, "ckpt-demo", step=12000, shard_sizes=[10, 10, 10],
        params=old_params, weights=old_weights, moments=(0.5, 0.25),
    )

    # ---------- 旧检查点 ckpt-conflict:output_proj 位值与 input_proj 不同 ----------
    # 与 ckpt-demo 同构,仅 tied.output_proj 一段换成另一组确定性位值,
    # 配合 template-conflict 让共享区间命中两份不一致的旧权重。
    conflict_weights = np.arange(30, dtype="<f4")
    conflict_weights[20:28] = 100.0 + np.arange(8, dtype="<f4")
    write_checkpoint(
        root, "ckpt-conflict", step=9000, shard_sizes=[10, 10, 10],
        params=list(old_params), weights=conflict_weights, moments=(0.9, 0.1),
    )

    # ---------- 小型旧检查点 ckpt-tiny:10 下标,2 片 (5/5) ----------
    tiny_params = [
        {"name": "a.weight", "shape": [6], "start": 0, "end": 6},
        {"name": "b.bias", "shape": [4], "start": 6, "end": 10},
    ]
    write_checkpoint(
        root, "ckpt-tiny", step=30, shard_sizes=[5, 5],
        params=tiny_params, weights=np.arange(10, dtype="<f4"),
    )

    # 模板初值统一取 1000+i,与继承来的旧权重(i<30)一眼可区分
    tpl_weights = (1000 + np.arange(30, dtype="<f4")).astype("<f4")

    # ---------- 模板 template-demo:层重排 + 共享成功 + 新参数保留初值 ----------
    # 分片边界 [0,12)/[12,24)/[24,30) 与旧检查点 [0,10)/[10,20)/[20,30) 不同,
    # 且多个参数区间跨分片。
    demo_params = [
        {"name": "head.bias", "shape": [2], "start": 0, "end": 2},        # 继承旧 [28,30)
        {"name": "embed.position", "shape": [4], "start": 2, "end": 6},   # 继承旧 [8,12)
        {"name": "embed.token", "shape": [8], "start": 6, "end": 14},     # 继承旧 [0,8)
        {"name": "new_block.norm", "shape": [2, 3], "start": 14, "end": 20},  # 新参数→模板初值
        {"name": "tied.input_proj", "shape": [2, 4], "start": 20, "end": 28},  # 共享,旧 [12,20)
        {"name": "tied.output_proj", "shape": [2, 4], "start": 20, "end": 28},  # 同区间共享,位值一致
        {"name": "extra.tail", "shape": [2], "start": 28, "end": 30},     # 新参数→模板初值
    ]
    write_checkpoint(
        root, "template-demo", step=0, shard_sizes=[12, 12, 6],
        params=demo_params, weights=tpl_weights,
    )

    # ---------- 模板 template-conflict:共享区间命中位值不同的两份旧权重 ----------
    # 用 ckpt-conflict 迁移时:input_proj 取旧 [12,20)(值 12..19),
    # output_proj 取旧 [20,28)(值 100..107),同一目标区间 [0,8) 位值不一致 → 冲突。
    conflict_params = [
        {"name": "tied.input_proj", "shape": [2, 4], "start": 0, "end": 8},
        {"name": "tied.output_proj", "shape": [2, 4], "start": 0, "end": 8},
        {"name": "embed.token", "shape": [8], "start": 8, "end": 16},
        {"name": "embed.position", "shape": [4], "start": 16, "end": 20},
        {"name": "head.bias", "shape": [2], "start": 20, "end": 22},
        {"name": "pad.fill", "shape": [8], "start": 22, "end": 30},
    ]
    write_checkpoint(
        root, "template-conflict", step=0, shard_sizes=[15, 15],
        params=conflict_params, weights=tpl_weights,
    )

    # ---------- 无效模板 template-invalid:形状与区间长度不符 ----------
    bad_mpath = write_checkpoint(
        root, "template-invalid", step=0, shard_sizes=[5, 5],
        params=[
            {"name": "a.weight", "shape": [6], "start": 0, "end": 6},
            {"name": "b.bias", "shape": [4], "start": 6, "end": 10},
        ],
        weights=tpl_weights[:10],
    )
    bad = json.loads(bad_mpath.read_text(encoding="utf-8"))
    bad["params"][1]["shape"] = [3]  # 区间长度 4,形状乘积 3 → 清单无效
    bad_mpath.write_text(json.dumps(bad, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"已生成无效清单 {bad_mpath}(b.bias 形状 [3] 与区间长度 4 不符)")


if __name__ == "__main__":
    root = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "sample_data" / "checkpoints"
    )
    main(root)
