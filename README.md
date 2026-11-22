# Adam 检查点重分片服务

训练换到卡数不同的机器后,Adam 检查点需要按新的进程数重新分片。本服务提供网页:
从挂载目录选择源检查点、填目标进程数,后台完成转换并发布批次,旧批次凭编号长期可取。

## 数据约定

源检查点目录内含 `manifest.json` 与若干 `.npy` 分片:

```json
{
  "step": 12000,
  "total_params": 1000,
  "shards": [
    {"file": "shard_000.npy", "start": 0, "end": 334},
    {"file": "shard_001.npy", "start": 334, "end": 667},
    {"file": "shard_002.npy", "start": 667, "end": 1000}
  ]
}
```

- `step` 训练步,`total_params` 参数总数,`[start, end)` 为该片的**全局下标区间**;
- 每个分片是 NumPy `.npy`,形状 `(end-start, 3)`,dtype `float32`,
  三列依次是**权重、Adam 一阶矩、Adam 二阶矩**;
- 区间必须从 0 开始连续覆盖到 `total_params`,不允许缺片、缺口或重叠,
  文件形状/dtype 必须与清单一致,否则该检查点标记为不可用。

## 转换规则

- 每个进程分一片,按全局下标**连续划分**,各片参数数量**最多相差 1**
  (前 `total % P` 片各多 1 个;进程数多于参数数时,末尾出现 0 行空片,仍是合法 `.npy`);
- 训练步与每个下标的三个原值原样保留(纯搬运,不做任何数值处理);
- 流式读写:读块 2048 行 + 写缓冲 2048 行,每行 12 字节,
  **分片读写缓冲合计 48 KiB ≤ 64 KiB**(`app/reshard.py` 中 `check_buffer_limits` 强制,
  默认 footprint 可用 `buffer_footprint_bytes()` 查看)。

## 可靠性与批次

- 转换先写入 `batches/.staging-<id>/` 暂存目录,清单与全部分片写齐并 fsync 后,
  通过 `os.rename` **原子发布**为 `batches/<id>/`,再落盘注册表;
- 缺片、区间缺口/重叠、文件与清单不符、写入失败、转换中断或服务重启:
  不产生半个批次,网页继续提供**上次成功的批次**;启动时自动清理暂存残留;
- 批次编号单调递增,注册表 `registry.json` 持久化;**发布新批次或重启服务后,
  旧批次仍凭编号取用**(注册表丢失时启动扫描批次目录自动重建);
- 新批次的清单与源清单同构,可直接作为下一次转换的输入(链式重分片)。

## 运行

### Docker Compose(推荐)

```bash
docker compose up --build
# 打开 http://localhost:8000
```

- `./sample_data/checkpoints` 只读挂载为 `/checkpoints`(网页里选择源检查点);
- 命名卷 `reshaper-data` 挂载为 `/data`,保存批次与注册表,重启不丢。

### 本地运行

```bash
pip install -r requirements-dev.txt
python3 scripts/make_sample.py            # 生成样例检查点
CHECKPOINTS_DIR=$PWD/sample_data/checkpoints DATA_DIR=$PWD/data \
  python3 -m uvicorn app.main:app --port 8000
```

### 重新生成样例

```bash
python3 scripts/make_sample.py            # ckpt-demo(3 片/1000 参数)与 ckpt-tiny(2 片/10 参数)
```

样例中下标 `i` 的三列取确定性值 `权重=i, 一阶矩=i+0.5, 二阶矩=i*0.25`,便于核对。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页 |
| GET | `/api/checkpoints` | 列出挂载目录内的检查点及校验结果 |
| POST | `/api/convert` | `{"checkpoint": "ckpt-demo", "processes": 4}` 发起转换,返回任务号 |
| GET | `/api/jobs` / `/api/jobs/{id}` | 任务状态(进行中/完成/失败原因) |
| GET | `/api/batches` | 已发布批次列表 |
| GET | `/api/batches/{id}` | 批次详情(含新清单) |
| GET | `/api/batches/{id}/manifest.json` | 下载新清单 |
| GET | `/api/batches/{id}/shards/{name}` | 下载单个分片 |
| GET | `/api/batches/{id}/download` | 整包 tar.gz |

## 测试

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -q
```

覆盖:均分划分(差 ≤1)、跨片转换逐值核对、链式转换、1 行缓冲的极端跨片、
缺片/缺口/重叠/形状与 dtype 不符、写入中断后旧批次可取、重启后凭编号取旧批次、
注册表丢失重建、API 全流程。
