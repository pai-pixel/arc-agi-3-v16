# 踩坑记录（V16 调试，2026-10-17 到 2026-10-18）

## 1. Kaggle 配额数字读反

- **现象**：Settings 页 "Kaggle GPU 07:41 / 30 hrs" 被误读为"还剩 7 小时 41 分"。
- **真相**：格式是**已用 / 总额**，即已用 7 小时 41 分。
- **教训**：先跟对话框原文交叉验证再下结论；用户查过配额说"还剩挺多"后，
  不要再围绕配额反复打转。

## 2. 僵尸会话烧额度

- **现象**：2026-10-17 10:47 启动的 interactive draft 会话在 "Adding data" 阶段卡住，
  说"等它自己挂"后没设回头检查点，空转 3 小时 12 分钟，用户 13:55 亲自手动取消。
- **教训**：凡是说"等它自己结束"的运行必须设回头检查点；
  用户明确要求直接取消时不要犟。

## 3. Monaco 编辑器不可靠

- **现象**：经浏览器改 Kaggle cell 代码多次出事——fill 不触发输入事件、
  换行被剥离、大段粘贴压成单行、字面量 `\n`、fill 误删整个 cell
  （V12/V13 的 SyntaxError、V8 的劈行都是这么来的）。
- **对策**：大段代码一律本地编辑成文件，走数据集挂载 + `exec()` 加载；
  小范围唯一字符串可用 Find/Replace。

## 4. 数据集挂载路径写错（V14/V15 死因）

- **现象**：`Path("/kaggle/input/v16-model-patch/...").exists()` 一直 False。
- **真相**：用户自建数据集挂载在 `/kaggle/input/datasets/haungjiawei/` 下，
  不是 `/kaggle/input/<slug>/`。V18 诊断版打印目录结构后查清。
- **教训**：先打印实际目录结构，不要假设挂载路径。

## 5. 提交环境无互联网（V16/V17 死因）

- **现象**：改 URL 下载补丁，`gaierror [Errno -3]` DNS 解析失败。
- **真相**：该竞赛 + GPU RTX Pro 6000 组合下，平台明确拒绝开启 Internet
 （"Internet cannot be enabled for this competition with the current accelerator."）。
- **对策**：URL 下载路线不可行，只能走数据集挂载。

## 6. BUNDLE_DIR 是全局变量不是环境变量

- **现象**：设 `os.environ["TAAF_KAGGLE_BUNDLE_DIR"]` 不生效。
- **真相**：BUNDLE_DIR 是 In[4] 定义的 Python 全局变量；
  In[6] 的 `_command_env()` 里 `env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)`
  会覆盖环境变量同名值。必须直接改全局变量本身。

## 7. 误关 MTP（v4 错误，已撤回）

- **现象**：审计发现 serving_setup 默认 MTP 开启（DEFAULT_MTP_TOKENS=3，
  vLLM 命令加 speculative config method="mtp"），推断"稠密模型无 MTP"，
  在补丁里设 `TAAF_VLLM_MTP_TOKENS=0` 关掉。
- **真相**：用户纠正——27B 模型自带原生 MTP（config 里 `mtp_num_hidden_layers=1`），
  官方模型卡示例命令本身也没有 MTP flag，serving_setup 的 MTP 逻辑不动。
- **教训**：先查模型 config 实测，不要凭"稠密/MoE"类别推断。

## 8. 量化参数误判（v3 错误，v6 修正）

- **现象**：V20 日志截断为 "compressed-tenso…"，结合 HF 官方 config
 （`quant_method="modelopt"`），推断应改成 `modelopt`。
- **真相**：V21 完整报错证明 Kaggle 镜像 `overseer66/qwen3-8-27b-nvfp4`
  是**定制版量化**，其 config 里 `quant_method="compressed-tensors"`
  （LLM Compressor 格式），与 HF 官方原版不一致。
  vLLM 硬校验要求 `--quantization` 与 config 一致，否则拒绝启动。
- **教训**：截断日志不能作为改参数的依据；改之前先拿完整报错原文。
  用户授权后才改成 `compressed-tensors`。

## 9. 语音转写歧义

- "踢尸" = TPU；"NTP" = MTP；"还剩三十四个小时" = 30 小时。
- **教训**：歧义列候选、用证据核实，不擅自猜。

## 10. 换模型要同步复核耦合参数

- 换模型类别（MoE 到稠密）时，先列耦合参数清单：
  batch/并发、vLLM 参数、显存分配。本次 env 覆盖：
  `MAX_NUM_SEQS=32`、`KV_CACHE_MEMORY_BYTES=16GiB`、
  `MAX_NUM_BATCHED_TOKENS=32768`、`KV_CACHE_DTYPE=fp8`、
  `ENABLE_PREFIX_CACHING=1`。
- 大改动后记录原代码参数/函数变量、写好注释，方便下次迭代。
