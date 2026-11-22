# Adam 检查点参数迁移服务(层重排 / 共享权重)

新模型重排了层,还把原先独立的参数改成了共享权重。本服务提供网页:
从挂载目录选择**旧检查点**与**新模型的初始化模板**,后台按参数名与形状迁移权重、
按模板的分片边界发布新批次;清单无效、共享权重冲突或写入失败时网页说明原因、
本次不发布,旧批次仍可下载。

## 数据约定

检查点目录内含 `manifest.json` 与若干 `.npy` 分片:

```json
{
  "step": 12000,
  "total_params": 30,
  "shards": [
    {"file": "shard_000.npy", "start": 0, "end": 10},
    {"file": "shard_001.npy", "start": 10, "end": 20},
    {"file": "shard_002.npy", "start": 20, "end": 30}
  ],
  "params": [
    {"name": "embed.token", "shape": [8], "start": 0, "end": 8},
    {"name": "tied.input_proj", "shape": [2, 4], "start": 12, "end": 20},
    {"name": "tied.output_proj", "shape": [2, 4], "start": 20, "end": 28}
  ]
}
```

- `step` 训练步,`total_params` 去重后的全局下标总数;
- `[start, end)` 为**全局下标区间**,区间可以跨分片;
- `params` 是参数清单,每条含参数名、形状与区间:
  - 参数名在同一份清单内**唯一**;
  - `shape` 为非空正整数数组,其乘积必须**等于区间长度**;
  - **共享权重**的多个名字使用**完全相同的区间与形状**(区间在 `params` 中重复出现);
  - 去掉重复区间后,`params` 必须从 0 连续覆盖到 `total_params`,无缺口、无重叠;
- `shards` 同样从 0 连续覆盖 `[0, total_params)`、互不重叠(允许 0 行空片);
- 每个分片是 NumPy `.npy`,形状 `(end-start, 3)`,dtype `float32`,
  三列依次是**权重、Adam 一阶矩、Adam 二阶矩**;
- 以上任一不符,该检查点在网页上标记为不可用并说明原因。

## 迁移规则

1. 新批次**沿用模板的分片边界**(分片文件名与 `[start, end)` 与模板完全一致);
2. 每个目标参数区间按 **(参数名, 形状)** 在旧清单里找旧权重:
   - 找到同名且同形状的旧参数 → 继承该旧参数区间的权重(下标按区间顺序搬运,层可任意重排);
   - 找不到(新参数,或同名但形状不同)→ **保留模板初值**;
3. **共享权重**:同一目标区间有多个参数名时,若它们在旧检查点命中多份**不同区间**
   的旧权重,只有**原始位值完全一致**才继承;否则报**共享权重冲突**,
   错误信息列出冲突的目标区间与参数名,本次不发布;
4. 新批次的**训练步归零,两阶 Adam 矩(第 2、3 列)全部归零**;
5. 流式读写,mmap 映射不计入缓冲,驻留内存的读块/写缓冲
   **合计上界 48 KiB ≤ 64 KiB**(`app/reshard.py` 中 `check_buffer_limits` 强制,
   `buffer_footprint_bytes()` 可查看):两个读缓冲(旧检查点、模板各 1024 行)
   + 一个写缓冲(2048 行),按整行 12 字节保守计;实际读缓冲只复制权重一列(4 字节)。

## 可靠性与批次

- 迁移先写入 `batches/.staging-<id>/` 暂存目录,清单与全部分片写齐并 fsync 后,
  通过 `os.rename` **原子发布**为 `batches/<id>/`,再落盘注册表;
- 清单无效、共享冲突(均在创建输出目录**之前**校验)、分片写入失败、迁移中断或
  服务重启:不产生半个批次,网页继续提供**上次成功的批次**;启动时自动清理暂存残留;
- 批次编号单调递增,注册表 `registry.json` 持久化;**发布新批次或重启服务后,
  旧批次仍凭编号取用**(注册表丢失时启动扫描批次目录自动重建)。

## 运行

### Docker Compose(推荐)

```bash
docker compose up --build
# 打开 http://localhost:8000
```

- `./sample_data/checkpoints` 只读挂载为 `/checkpoints`(网页里选旧检查点与模板);
- 命名卷 `reshaper-data` 挂载为 `/data`,保存批次与注册表,重启不丢。

### 本地运行

```bash
pip install -r requirements-dev.txt
python3 scripts/make_sample.py            # 生成样例检查点/模板
CHECKPOINTS_DIR=$PWD/sample_data/checkpoints DATA_DIR=$PWD/data \
  python3 -m uvicorn app.main:app --port 8000
```

### 样例数据

`python3 scripts/make_sample.py` 在 `sample_data/checkpoints/` 生成:

| 目录 | 角色 | 说明 |
|---|---|---|
| `ckpt-demo` | 旧检查点 | 30 下标/3 片,含一对位值完全相同的共享来源参数 |
| `ckpt-conflict` | 旧检查点 | 同构,但 `tied.output_proj` 位值与 `tied.input_proj` 不同 |
| `ckpt-tiny` | 旧检查点 | 10 下标/2 片的小型样例 |
| `template-demo` | 新模型模板 | **层重排 + 共享权重成功 + 新参数保留初值**,分片边界与旧检查点不同 |
| `template-conflict` | 新模型模板 | 共享区间命中两份不一致旧权重 → **冲突失败** |
| `template-invalid` | 无效模板 | 参数形状与区间长度不符,网页展示无效原因 |

网页演示:
- `ckpt-demo` + `template-demo` → 成功,可下载新批次并核对继承/模板初值/Adam 归零;
- `ckpt-conflict` + `template-conflict` → 失败,任务里列出冲突参数名,旧批次不受影响;
- 任意检查点 + `template-invalid` → 失败并提示清单无效原因。

模板初值统一取 `1000 + 全局下标`,与继承来的旧权重(`< 30`)一眼可区分。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页 |
| GET | `/api/checkpoints` | 列出挂载目录内的检查点/模板及校验结果(含参数条数) |
| POST | `/api/convert` | `{"checkpoint": "ckpt-demo", "template": "template-demo"}` 发起迁移,返回任务号 |
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

覆盖:清单校验(参数名唯一、形状/区间一致、共享区间、缺口/重叠、分片 dtype/形状/截断/
路径穿越)、按 (名,形状) 继承与层重排、同名不同形状回退模板、共享权重一致继承与冲突
报错并列名、训练步与两阶 Adam 归零、跨分片参数与 1 行极端缓冲、0 行空片、
64 KiB 缓冲上限、写入中断后旧批次可取、冲突/无效清单不发布、重启后凭编号取旧批次、
注册表丢失重建、API 全流程。
