# ARC-AGI-3 V16：Qwen3.8-27B-NVFP4 稠密模型 serving 补丁

ARC-AGI-3 竞赛 notebook `duck-qwen3-8-27b-nvfp4-dense-v16` 的模型适配补丁。
把为 180B MoE 老模型（RadixArk/Qwen3.8-Flash-Next-NVFP4）写的
`serving_setup.py`（3000+ 行）打补丁，使其能加载 27B 稠密新模型
（nvidia/Qwen3.8-27B-NVFP4，经 Kaggle 镜像 `overseer66/qwen3-8-27b-nvfp4`）。

## 文件结构

```
arc-agi-3-v16/
├── v16_model_patch.py        # 当前版本补丁（与数据集最新版一致）
├── patches/
│   ├── v1_v16_model_patch.py # 数据集 Version 1：4 个中和
│   ├── v2_v16_model_patch.py # 数据集 Version 2：+ PLE 哈希校验中和
│   ├── v3_v16_model_patch.py # 数据集 Version 3：+ 量化修正为 modelopt（推断有误）
│   ├── v4_v16_model_patch.py # 数据集 Version 4：+ 关闭 MTP（错误，已撤回）
│   ├── v5_v16_model_patch.py # 数据集 Version 5：撤回 MTP/量化改动
│   └── v6_v16_model_patch.py # 数据集 Version 6：量化修正为 compressed-tensors
├── README.md                 # 本文件
├── ITERATION_LOG.md          # 完整迭代日志（notebook V2 到 V22）
└── TROUBLESHOOTING.md        # 踩坑记录与解决方案
```

注：`patches/v1` 到 `v5` 为按变更记录重建的历史版本（非逐字节原文件），
`v6` 与当前 `v16_model_patch.py` 一致。

## 补丁做什么

1. **拷贝 bundle**：把只读的 TAAF bundle 拷到 `/kaggle/working/v16_bundle`（可写）。
2. **发现新模型**：按 Kaggle 标准挂载结构定位新模型目录，计算指纹
   （config sha256、文件数、总字节数）。
3. **替换 8 个常量**：`MODEL_HF_REPO`、`MODEL_HF_REVISION`、
   `MODEL_KAGGLE_PATH`、`MODEL_MANIFEST_SHA256`、`MODEL_CONFIG_SHA256`、
   `MODEL_FILE_COUNT`、`MODEL_TOTAL_BYTES`、`SERVED_MODEL_NAME`。
4. **中和 5 处老模型校验**：
   - 4b：`source_identity()` 整体跳过（文件/指纹身份校验）
   - 4c：`resolve_model_dir()` 整体跳过（只认老模型路径）
   - 4d：`patch_ple_layer()` 整体跳过（PLE 为老 MoE 模型专用）
   - 4e：`verify_model()` 整体跳过（10+ 项老模型校验）
   - 4f：`runtime_environment()` 的 PLE 源码哈希校验
5. **修正 `--quantization`**：Kaggle 镜像是定制版量化，
   config 里 `quant_method="compressed-tensors"`（非官方原版的 `modelopt`），
   vLLM 硬校验要求命令行参数与 config 一致。
6. **重定向 `BUNDLE_DIR`**：改 notebook 全局变量（不是环境变量），
   让 In[6] 的 setup cell 自动跑补丁版。

## 关键事实

- **MTP 不要关**：27B 模型自带原生 MTP（config 里 `mtp_num_hidden_layers=1`）。
- **量化必须改**：Kaggle 镜像 `overseer66/qwen3-8-27b-nvfp4` 是定制版量化，
  `quant_method="compressed-tensors"`，与 HF 官方 `nvidia/Qwen3.8-27B-NVFP4`
  的 `modelopt` 不一致。
- **数据集挂载路径**：`/kaggle/input/datasets/haungjiawei/v16-model-patch/`，
  不是 `/kaggle/input/v16-model-patch/`。
- **notebook 无互联网**：竞赛 + GPU 组合下平台拒绝开启 Internet，
  不能从 URL 下载，必须走数据集挂载。
- **大段代码不粘编辑器**：Monaco 编辑器粘贴大段代码不可靠
  （换行剥离、字面量 `\n`），一律本地编辑成文件、走数据集加载。

## 使用方式

notebook 的 "V16 vLLM 参数覆盖" cell 末尾用 3 行加载：

```python
from pathlib import Path
_patch_file = Path("/kaggle/input/datasets/haungjiawei/v16-model-patch/v16_model_patch.py")
print(f"[_patch_file] loading {_patch_file} exists={_patch_file.exists()}", flush=True)
exec(_patch_file.read_text())
```

更新流程：本地改补丁 → `py_compile` → push 到本仓库 →
用 GitHub raw URL 建 Kaggle 数据集新版本 →
notebook 挂载新版本数据集 → 提交新版本。

## 相关链接

- Notebook: https://www.kaggle.com/code/haungjiawei/duck-qwen3-8-27b-nvfp4-dense-v16
- 数据集: https://www.kaggle.com/datasets/haungjiawei/v16-model-patch
- 模型镜像: https://www.kaggle.com/models/overseer66/qwen3-8-27b-nvfp4
- HF 官方模型: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
