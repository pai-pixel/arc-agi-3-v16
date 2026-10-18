# V16 迭代日志（2026-10-17 到 2026-10-18）

目标：让 ARC-AGI-3 notebook `duck-qwen3-8-27b-nvfp4-dense-v16`
在 Kaggle GPU RTX Pro 6000 上跑通 27B 稠密模型（nvidia/Qwen3.8-27B-NVFP4），
替换原来为 180B MoE 老模型写的 serving 链路。

约定：以下时间均为北京时间（Asia/Shanghai）。"Failed Ns" 指内核运行 N 秒后失败。

## 背景

- 老模型：RadixArk/Qwen3.8-Flash-Next-NVFP4（180B MoE），Kaggle 镜像
  `keithtyser/qwen3-8-flash-next-nvfp4`，419 个文件，约 17GB。
- 新模型：nvidia/Qwen3.8-27B-NVFP4（27B 稠密），Kaggle 镜像
  `overseer66/qwen3-8-27b-nvfp4`，48 个文件，约 46.9GB，
  目录 `/kaggle/input/models/overseer66/qwen3-8-27b-nvfp4/pytorch/default/1/Qwen3.8-27B-NVFP4`。
- `serving_setup.py`（3109 行，来自 bundle `duck-qwen38-nvfp4-mtp-vllm-smoke-v1`）
  把老模型身份写死成模块常量：repo、revision、Kaggle 路径、3 个文件指纹、
  文件数 419、总字节数，以及 10+ 项老模型校验。
- HF revision（2026-10-17 查）：`482ca0f3832238542f8f5295dde86b5f22711d80`。

## Notebook 版本流水

| 版本 | 时间 | 结果 | 死因 / 说明 |
|------|------|------|-------------|
| V2 | 10-17 09:41 | Cancelled（0 秒） | 误用 T4 x2 加速器，未耗额度 |
| V3 | 10-17 09:46 | Failed 11 秒 | `KV_CACHE_DTYPE=fp8_e4m3` 被第 428 行校验拒绝（只认 auto/fp8），改 fp8 |
| V4 | 10-17 09:58 | Failed 11.3 秒 | `resolve_model_dir()` 只认老模型路径。用户拍板：全量补丁迭代修 |
| V5 | 10-17 11:13 | Failed 62.7 秒 | 补丁跑通（bundle 拷贝成功），自动发现按 MODEL_MANIFEST.json 找新模型得 0 个 |
| V6 | 10-17 11:43 | Failed 134.9 秒 | 8 常量全替换成功，卡 `source_identity()` 第一道（文件哈希） |
| V7 | 10-17 11:56 | Failed 27.7 秒 | 卡 `source_identity()` 第二道（manifest 哈希） |
| V8 | 10-17 12:48 | Failed 19.7 秒 | 补丁自身 SyntaxError（浏览器编辑劈行） |
| V9 | 10-17 13:21 | Failed 43.5 秒 | rglob 猜的路径被 `resolve_model_dir()` 判不存在 |
| V10 | 10-17 14:00 | Failed 119.3 秒 | 补丁内核验证路径 exists=True，但子进程 `resolve_model_dir()` 仍 FileNotFoundError（查的是 MODEL_MANIFEST.json 而非目录） |
| V11 | 10-17 14:44 | Failed 194.7 秒 | 4c 生效，卡 `patch_ple_layer()` 的 PATCH_IDENTITY 校验 |
| V12 | 10-17 16:57 | Failed 10.9 秒 | In[5] SyntaxError：4e 被压成单行、`\n` 变字面量，整 cell 未执行 |
| V13 | 10-17 18:00 | Failed 10.7 秒 | 同一 SyntaxError 未修好（4d 结尾与 4e 标题间是字面量 `\`+`n`） |
| V14 | 10-17 19:56 | Failed 35 秒 | 改用数据集文件加载，但路径写错（`exists=False`） |
| V15 | 10-17 20:03 | Failed 48.8 秒 | 同一 `exists=False` |
| V16 | 10-17 21:00 | Failed 72.6 秒 | 改 URL 下载，但提交环境无互联网，DNS 解析失败（gaierror）。竞赛+GPU 组合下平台明确拒绝开 Internet |
| V17 | 10-17 21:16 | Failed 约 75 秒 | 与 V16 同套 URL 代码，同样失败 |
| V18 | 10-17 23:35 | Failed 219.6 秒 | 诊断版：只打印 `/kaggle/input` 结构，查清数据集实际挂载在 `/kaggle/input/datasets/haungjiawei/` 下 |
| V19 | 10-18 00:40 | Failed 193.3 秒 | **补丁首次加载成功**（exists=True），4 个中和 + 8 常量全生效，卡 `runtime_environment()` 的 PLE 源码哈希校验（"The patched PLE source hash changed before import."） |
| V20 | 10-18 07:24 | Failed 216.4 秒 | 4f 生效，通过 runtime layer fast check，进到 vLLM 启动：APIServer 解析架构 Qwen3_5ForConditionalGeneration，随后 pydantic ValidationError（信息被截断为 "compressed-tenso…"），"vLLM exited before readiness" |
| V21 | 10-18 07:42 | Failed 194.8 秒 | 五个中和全生效，vLLM 引擎配置阶段崩溃，完整报错："Quantization method specified in the model config (compressed-tensors) does not match the quantization method specified in the `quantization` argument (modelopt_fp4)" |
| V22 | 10-18 07:52 | Running | 量化改成 `compressed-tensors`（用户授权），数据集 Version 6 |

## 数据集版本流水（haungjiawei/v16-model-patch）

| 版本 | 时间 | 大小 | 变更 |
|------|------|------|------|
| v1 | 10-17 19:42 | 14.74 kB | 4 个中和：source_identity / resolve_model_dir / patch_ple_layer / verify_model |
| v2 | 10-18 07:22 | 15.99 kB | + 4f：中和 runtime_environment() 的 PLE 哈希校验 |
| v3 | 10-18 07:34 | 16.69 kB | + 4g：`--quantization` 改成 `modelopt`（基于 V20 截断日志的推断，后证有误） |
| v4 | 10-18 07:38 | 17.14 kB | + 关闭 MTP（`TAAF_VLLM_MTP_TOKENS=0`，错误，用户纠正后撤回） |
| v5 | 10-18 07:40 | 16.74 kB | 撤回 MTP 关闭与量化改动，回到 5 个中和 |
| v6 | 10-18 07:49 | 17.35 kB | + 4g：`--quantization` 改成 `compressed-tensors`（用户授权） |

## 关键转折点

1. **2026-10-17 10:29**：用户拍板全量补丁 A 路线迭代修（而非重写 serving_setup）。
2. **2026-10-17 18:43**：用户要求大段代码走本地编辑+文件上传，不再粘 Monaco 编辑器。
3. **2026-10-17 23:45**：V18 诊断查清数据集挂载路径根因（`/kaggle/input/datasets/haungjiawei/`）。
4. **2026-10-18 00:40**：V19 补丁首次加载成功（exists=True），结束"路径写错"阶段。
5. **2026-10-18 07:38**：用户纠正——27B 自带 MTP，不要关；量化也不要动（当时）。
6. **2026-10-18 07:42**：V21 完整报错证明 Kaggle 镜像是定制版量化（compressed-tensors），
   用户授权改量化参数。
7. **2026-10-18 07:52**：V22 提交（Running），补丁迁入 GitHub。

## 未解 / 待验证

- vLLM 能否真正加载 compressed-tensors 格式的 27B 模型（待 V22 日志）。
- MTP speculative config（method="mtp"）在该模型上的实际行为（serving_setup 默认开启，用户要求不动）。
- `moe_backend` 默认为 None，未受影响（已审计）。
- vLLM 命令中的 `--tool-call-parser qwen3_coder`、`--reasoning-parser qwen3`、
  `--chat-template` 与官方模型卡示例一致，暂不动。
