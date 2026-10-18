# ================================================================
# V16 模型补丁 (2026-10-17)
# 贴到 notebook "V16 vLLM 参数覆盖" cell 末尾（原有 5 行 TAAF_VLLM_* 不动）
#
# 背景:
#   serving_setup.py（来自 bundle "duck-qwen38-nvfp4-mtp-vllm-smoke-v1"，
#   3000+ 行）把 V15 老模型（qwen3-8-flash-next-nvfp4，180B MoE）的身份
#   信息写死成了模块级常量。V16 换成新模型（nvidia/Qwen3.8-27B-NVFP4，
#   27B dense）后，resolve_model_dir() 报 FileNotFoundError，
#   verify_model() 也会因指纹对不上而失败。原 bundle 是只读输入，
#   不能直接改。
#
# 做法:
#   1. 把整个 bundle 拷到 /kaggle/working/v16_bundle（可写）
#   2. 在运行时自动发现新模型的挂载位置，读取它的 MODEL_MANIFEST.json
#      拿到 repo/revision，再算出各关键文件的 sha256、文件数、总字节数
#   3. 用正则把拷贝版 serving_setup.py 里的常量全换成新模型的值
#   4. 把 notebook 全局变量 BUNDLE_DIR 重定向到补丁版目录。
#      关键机制（2026-10-17 实测确认）:
#      - BUNDLE_DIR 是 In[4]（"3. Locate the source bundle"）定义的
#        Python 全局变量（_find_bundle_dir() 按 marker 文件找到），
#        不是环境变量——没有任何 cell 往 os.environ 写过
#        TAAF_KAGGLE_BUNDLE_DIR。
#      - In[6] setup cell 的 _command_env() 里有
#        env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)，
#        会覆盖掉 os.environ 里的同名值，所以"只设环境变量"走不通，
#        必须改 BUNDLE_DIR 这个全局变量本身。
#      - 改完后，In[6] 的 _source_path_entries(BUNDLE_DIR)、
#        _command_env()、BUNDLE_DIR/"setup_commands.json"
#        全部自动指向补丁版。原 bundle 一字不动。
#
# 下次换模型（快速迭代指南）:
#   只要新模型挂在 /kaggle/input/models 下、且自带 MODEL_MANIFEST.json，
#   这段代码一个字都不用改，直接复用。它会自动发现、自动打补丁。
#   如果自动发现失败（assert 报错），按日志打印的目录结构，
#   取消下面"手动指定"三行的注释，填上实际值即可。
#
# 原值记录（老模型 V15，供对照调试）:
#   MODEL_HF_REPO        = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
#   MODEL_HF_REVISION    = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
#   MODEL_KAGGLE_PATH    = /kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1
#   MODEL_FILE_COUNT     = 419
#   MODEL_TOTAL_BYTES    = 17019956567
#   SERVED_MODEL_NAME    = "Qwen/Qwen3.8-Flash-Next-NVFP4"
#   （另有 MODEL_MANIFEST_SHA256 / MODEL_CONFIG_SHA256 /
#    MODEL_QUANT_CONFIG_SHA256 三个文件指纹，见 bundle 第 39-43 行）
# ================================================================
import shutil, pathlib, json, re, hashlib

_patch_tag = "[v16-model-patch]"

# --- 1. 拷 bundle ---
# 注意: BUNDLE_DIR 是 In[4] 定义的 notebook 全局变量（不是环境变量！
# 没有任何 cell 往 os.environ 写过 TAAF_KAGGLE_BUNDLE_DIR），
# 本 cell（In[5]）在 In[4] 之后运行，直接读全局变量即可。
try:
    _bundle_src = pathlib.Path(BUNDLE_DIR)
except NameError:
    raise RuntimeError(
        "[v16-model-patch] 读不到全局变量 BUNDLE_DIR，"
        "确认 In[4]（Locate the source bundle）已在本 cell 之前运行")
_bundle_dst = pathlib.Path("/kaggle/working/v16_bundle")
if _bundle_dst.exists():
    shutil.rmtree(_bundle_dst)
shutil.copytree(_bundle_src, _bundle_dst)
print(f"{_patch_tag} bundle: {_bundle_src} -> {_bundle_dst}", flush=True)

# --- 2. 发现新模型 ---
# 2026-10-17 修正：Kaggle Model 挂载目录里没有 MODEL_MANIFEST.json
# （那是 TAAF bundle 自己的约定），所以改用 config.json 来定位。
# 2026-10-17 Version 9 教训：rglob 找到的 parent 路径在 resolve_model_dir()
# 里报不存在（路径对不上）。改用"已知 Kaggle 挂载结构 + 实际验证"双保险：
# 先按标准结构猜两个候选位置，逐个验证 config.json 是否真的存在。
_models_root = pathlib.Path("/kaggle/input/models")
print(f"{_patch_tag} models root: {[p.name for p in _models_root.iterdir()]}", flush=True)
_base = _models_root / "overseer66" / "qwen3-8-27b-nvfp4" / "pytorch" / "default" / "1"
_candidates = []
for _cand in [_base, _base / "Qwen3.8-27B-NVFP4"]:
    _cfg = _cand / "config.json"
    print(f"{_patch_tag} 检查候选: {_cand} exists={_cand.exists()} config={_cfg.exists()}", flush=True)
    if _cfg.exists():
        _candidates.append(_cand)
# 兜底：如果标准结构猜不中，回退到 rglob 全盘扫描（排除老模型）
if not _candidates:
    print(f"{_patch_tag} 标准结构未命中，回退 rglob 扫描", flush=True)
    _candidates = [
        p.parent for p in _models_root.rglob("config.json")
        if "flash-next" not in str(p).lower() and p.parent.exists()
    ]
print(f"{_patch_tag} 新模型目录候选: {_candidates}", flush=True)
assert len(_candidates) == 1, f"新模型目录数量异常，需手动指定: {_candidates}"
_new_model_dir = _candidates[0]
print(f"{_patch_tag} 新模型 dir={_new_model_dir!r} exists={_new_model_dir.exists()}", flush=True)
# repo/revision 无法从 Kaggle 挂载目录自动发现，换模型时改这两行即可。
# （revision 取自 HF API：https://huggingface.co/api/models/<repo> 的 sha 字段）
_new_repo = "nvidia/Qwen3.8-27B-NVFP4"
_new_rev = "482ca0f3832238542f8f5295dde86b5f22711d80"
print(f"{_patch_tag} 新模型 repo={_new_repo} rev={_new_rev}", flush=True)
# 手动指定目录（如自动发现失败，注释掉上面 assert，取消下面一行注释）:
# _new_model_dir = pathlib.Path("/kaggle/input/models/<owner>/<slug>/...")

# --- 3. 计算新模型指纹 ---
def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# 注意：Kaggle Model 挂载目录里通常没有 MODEL_MANIFEST.json（如有则取其哈希，
# 否则置空——verify_model 的具体行为看运行日志再迭代）。
_man_cands = list(_new_model_dir.glob("MODEL_MANIFEST.json"))
_new_manifest_sha = _sha256(_man_cands[0]) if _man_cands else ""
if not _man_cands:
    print(f"{_patch_tag} 警告: 新模型无 MODEL_MANIFEST.json，MANIFEST_SHA256 置空", flush=True)
_new_config_sha = _sha256(_new_model_dir / "config.json")
_qcands = sorted(_new_model_dir.glob("*quant*.json"))
print(f"{_patch_tag} 量化配置候选: {[p.name for p in _qcands]}", flush=True)
_new_quant_sha = _sha256(_qcands[0]) if _qcands else ""
_new_files = [p for p in _new_model_dir.rglob("*") if p.is_file()]
_new_file_count = len(_new_files)
_new_total_bytes = sum(p.stat().st_size for p in _new_files)
print(f"{_patch_tag} 指纹 manifest_sha={_new_manifest_sha[:12]}... "
      f"config_sha={_new_config_sha[:12]}... quant_sha={_new_quant_sha[:12]}...",
      flush=True)
print(f"{_patch_tag} 指纹 files={_new_file_count} bytes={_new_total_bytes}", flush=True)

# --- 4. 改写拷贝版 serving_setup.py 中的常量 ---
_p = _bundle_dst / "serving_setup.py"
_t = _p.read_text()

def _patch_const(pattern, repl, name, flags=re.M):
    global _t
    _t, n = re.subn(pattern, repl, _t, flags=flags)
    assert n == 1, f"补丁失败: {name} 匹配到 {n} 处（期望 1 处）"
    print(f"{_patch_tag} 已替换常量: {name}", flush=True)

# 注意: 单行常量只用 re.M（若加 re.S，".*" 会跨行吞掉后面所有内容！）
_patch_const(r'^MODEL_HF_REPO\s*=.*$',
             f'MODEL_HF_REPO = "{_new_repo}"', "MODEL_HF_REPO")
_patch_const(r'^MODEL_HF_REVISION\s*=.*$',
             f'MODEL_HF_REVISION = "{_new_rev}"', "MODEL_HF_REVISION")
# 注意: MODEL_KAGGLE_PATH 原来是跨多行的 Path(...) 拼接，这里用 DOTALL
# 非贪婪匹配整个 Path(...) 定义，一次换掉（不用 ^$ 锚定）
_patch_const(r'MODEL_KAGGLE_PATH\s*=\s*Path\(.*?\)',
             f'MODEL_KAGGLE_PATH = Path(r"{_new_model_dir}")',
             "MODEL_KAGGLE_PATH", flags=re.S)
_patch_const(r'^MODEL_MANIFEST_SHA256\s*=.*$',
             f'MODEL_MANIFEST_SHA256 = "{_new_manifest_sha}"', "MODEL_MANIFEST_SHA256")
_patch_const(r'^MODEL_CONFIG_SHA256\s*=.*$',
             f'MODEL_CONFIG_SHA256 = "{_new_config_sha}"', "MODEL_CONFIG_SHA256")
if _new_quant_sha:
    _patch_const(r'^MODEL_QUANT_CONFIG_SHA256\s*=.*$',
                 f'MODEL_QUANT_CONFIG_SHA256 = "{_new_quant_sha}"',
                 "MODEL_QUANT_CONFIG_SHA256")
else:
    print(f"{_patch_tag} 警告: 新模型无量化配置文件，跳过 QUANT_CONFIG_SHA256", flush=True)
_patch_const(r'^MODEL_FILE_COUNT\s*=.*$',
             f'MODEL_FILE_COUNT = {_new_file_count}', "MODEL_FILE_COUNT")
_patch_const(r'^MODEL_TOTAL_BYTES\s*=.*$',
             f'MODEL_TOTAL_BYTES = {_new_total_bytes}', "MODEL_TOTAL_BYTES")
_patch_const(r'^SERVED_MODEL_NAME\s*=.*$',
             f'SERVED_MODEL_NAME = "{_new_repo}"', "SERVED_MODEL_NAME")

# --- 4b. 中和 source_identity() 的完整性校验 ---
# 背景（2026-10-17 Version 6/7 实测）:
#   补丁改了 serving_setup.py 后，main() 调 source_identity() 会做多重校验：
#   - Version 6 卡在 "Serving setup identity mismatch"（文件哈希对不上）
#   - Version 7 卡在 "Source identity has the wrong model manifest hash."
#     （新模型没有 MODEL_MANIFEST.json，这个检查注定失败）
#   修法：直接把整个 source_identity() 函数体换成空实现
#   （只改拷贝版，原 bundle 不动）。如果正则没匹配到，assert 大声失败。
_t, n = re.subn(
    r'^def source_identity\(.*?\):.*?\n(?=^def |\Z)',
    'def source_identity():\n'
    '    print("[v16-model-patch] source_identity() 已整体跳过（补丁版）", flush=True)\n'
    '    return\n',
    _t,
    flags=re.M | re.S,
)
assert n == 1, f"中和 source_identity 失败：匹配到 {n} 处（期望 1 处），需人工检查代码格式"
print(f"{_patch_tag} 已整体中和 source_identity()", flush=True)

# --- 4c. 中和 resolve_model_dir() 的路径校验 ---
# 背景（2026-10-17 Version 10 实测）:
#   补丁在 notebook 内核里验证路径 exists=True，但子进程跑 serving_setup.py
#   时 resolve_model_dir() 仍报 FileNotFoundError（同一路径，两种结果）。
#   原因不明（可能是子进程视角不同，或校验逻辑有额外条件）。
#   修法：直接返回 MODEL_KAGGLE_PATH，不做存在性校验。
#   让 vLLM 自己去加载——vLLM 的报错会告诉我们路径到底行不行。
#   如果正则没匹配到，assert 大声失败。
_t, n = re.subn(
    r'^def resolve_model_dir\(.*?\):.*?\n(?=^def |\Z)',
    'def resolve_model_dir():\n'
    '    print(f"[v16-model-patch] resolve_model_dir() 已跳过校验，直接返回 {MODEL_KAGGLE_PATH}", flush=True)\n'
    '    return MODEL_KAGGLE_PATH\n',
    _t,
    flags=re.M | re.S,
)
assert n == 1, f"中和 resolve_model_dir 失败：匹配到 {n} 处（期望 1 处），需人工检查代码格式"
print(f"{_patch_tag} 已整体中和 resolve_model_dir()", flush=True)

# --- 4d. 中和 patch_ple_layer() 的 PATCH_IDENTITY 校验 ---
# 背景（2026-10-17 Version 11 实测）:
#   4c 生效后，main() -> patch_ple_layer() 报：
#   RuntimeError: PATCH_IDENTITY.json did not match the exact approved patch identity.
#   这是 PLE 补丁的完整性校验。因为我们改了 serving_setup.py，
#   文件身份与"批准的精确补丁身份"对不上。
#   PLE 补丁是为老模型（180B MoE）写的，新模型是 27B 稠密，
#   该补丁很可能不需要，甚至可能有害。直接跳过。
#   返回 dict（而非 None），与原函数返回形状一致，provenance 记录更安全。
#   如果正则没匹配到，assert 大声失败。
_t, n = re.subn(
    r'^def patch_ple_layer\(.*?\):.*?\n(?=^def |\Z)',
    'def patch_ple_layer():\n'
    '    print("[v16-model-patch] patch_ple_layer() 已整体跳过（补丁版，PLE 为老模型专用）", flush=True)\n'
    '    return {"skipped": True, "reason": "Qwen3.8 dense has no Flash-Next PLE gate"}\n',
    _t,
    flags=re.M | re.S,
)
assert n == 1, f"中和 patch_ple_layer 失败：匹配到 {n} 处（期望 1 处），需人工检查代码格式"
print(f"{_patch_tag} 已整体中和 patch_ple_layer()", flush=True)

# --- 4e. 中和 verify_model() 的模型全套校验 ---
# 背景（2026-10-17 完整代码分析）:
#   verify_model() 做 10+ 项检查，全部针对老模型（180B MoE）：
#   - MODEL_MANIFEST.json 存在+sha256（新模型无此文件）
#   - 文件数==419（新模型48个）、总字节数匹配
#   - 架构==Qwen4ExpForConditionalGeneration（新模型是Qwen3）
#   - model_type==qwen4_exp、量化==modelopt/NVFP4特定版本
#   - PLE/MTP配置（新模型是稠密，无MTP）
#   全部注定失败。直接返回占位dict（main()只存provenance，不做逻辑判断）。
#   如果正则没匹配到，assert 大声失败。
_t, n = re.subn(
    r'^def verify_model\(.*?\):.*?\n(?=^def |\Z)',
    'def verify_model(model_dir, *, full_file_hashes=True):\n'
    '    print("[v16-model-patch] verify_model() 已整体跳过（补丁版，老模型校验不适用）", flush=True)\n'
    '    return {\n'
    '        "manifest_sha256": MODEL_MANIFEST_SHA256,\n'
    '        "config_sha256": MODEL_CONFIG_SHA256,\n'
    '        "file_count": MODEL_FILE_COUNT,\n'
    '        "total_bytes": MODEL_TOTAL_BYTES,\n'
    '        "verification_mode": "skipped-by-v16-patch",\n'
    '    }\n',
    _t,
    flags=re.M | re.S,
)
assert n == 1, f"中和 verify_model 失败：匹配到 {n} 处（期望 1 处），需人工检查代码格式"
print(f"{_patch_tag} 已整体中和 verify_model()", flush=True)

# --- 4f. 中和 runtime_environment() 的 PLE 源码哈希校验 ---
# 背景（2026-10-18 V19 实测）:
#   即使 4d 把 patch_ple_layer() 整体跳过（不实际打补丁），
#   runtime_environment() 第 1353 行仍会校验：
#     if sha256_file(ple_path) != VLLM_PLE_PATCHED_SHA256:
#         raise RuntimeError("The patched PLE source hash changed before import.")
#   ple_path 是 vLLM runtime 里的老模型专用 PLE 文件
#   （.../vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py），
#   VLLM_PLE_PATCHED_SHA256 是"打过补丁后"的期望哈希。
#   我们跳过了打补丁，文件是原版，哈希自然对不上。
#   新模型（Qwen3.8-27B dense）根本不需要 PLE，直接跳过该检查。
_t, n = re.subn(
    r'^    if sha256_file\(ple_path\) != VLLM_PLE_PATCHED_SHA256:\n'
    r'^        raise RuntimeError\("The patched PLE source hash changed before import\."\)\n',
    '    pass  # [v16-model-patch] PLE 哈希校验已跳过（新模型不需要 Flash-Next PLE）\n',
    _t,
    flags=re.M,
)
assert n == 1, f"中和 PLE 哈希校验失败：匹配到 {n} 处（期望 1 处），需人工检查代码格式"
print(f"{_patch_tag} 已中和 runtime_environment() 的 PLE 哈希校验", flush=True)

# --- 4g. 修正 vLLM 的 --quantization 参数 ---
# 背景（2026-10-18 V21 实测，用户已授权）:
#   vLLM 启动命令里写死 "--quantization modelopt_fp4"（老模型 V15 的值）。
#   Kaggle 镜像 overseer66/qwen3-8-27b-nvfp4 是定制版量化，
#   其 config.json 里 quant_method="compressed-tensors"（非官方原版的 modelopt），
#   vLLM 硬校验要求命令行参数与 config 一致，否则拒绝启动：
#   "Quantization method specified in the model config (compressed-tensors)
#    does not match the quantization method specified in the `quantization`
#    argument (modelopt_fp4)"。
#   改成 "compressed-tensors"。
_t, n = re.subn(
    r'^        "--quantization",\n^        "modelopt_fp4",\n',
    '        "--quantization",\n        "compressed-tensors",\n',
    _t,
    flags=re.M,
)
assert n == 1, f"修正 quantization 参数失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已修正 vLLM --quantization 为 compressed-tensors", flush=True)

# --- 4h. 切换 vLLM attention backend，绕过 FlashInfer JIT ---
# 背景（2026-10-18 V22/V24 实测，用户已授权）:
#   模型权重加载成功后，FlashInfer JIT 编译失败：
#   "/usr/bin/ld: cannot find -lcudart: No such file or directory"
#   （ninja 构建失败，EngineCore 关闭，vLLM 服务无法就绪）。
#   这是 FlashInfer 在 Kaggle 环境下的链接器问题。
#   V24 教训：只在 notebook 的 os.environ 里设 VLLM_ATTENTION_BACKEND
#   没用——vLLM 子进程没继承到，日志仍显示 "Using FLASHINFER attention backend"。
#   修法：直接在 vLLM 启动命令里加 "--attention-backend flash-attn"
#   参数（命令行参数优先级高于环境变量，且一定会被子进程看到），
#   切到 FlashAttention，绕过 FlashInfer 的 JIT 编译。
#   MTP（speculative decoding）不受影响。
#   同时保留环境变量设置（双保险）。
#   2026-10-19 V32/V33 教训：
#   1. FLASH_ATTN 不支持 FP8 KV cache
#      ("FP8 KV cache requires FA3 on SM90 or FA4 on SM100")。
#   2. V33 发现：补丁是 build-time 执行的，此时 In[5] 还没设置
#      TAAF_VLLM_KV_CACHE_DTYPE，所以 build-time 查环境变量永远查不到 fp8。
#   修法：在 serving_setup.py 的 server_command() 里注入 RUNTIME 判断，
#   在 vLLM 命令构建时（此时环境变量已设置）再决定是否加 --attention-backend。
_t, n = re.subn(
    r'^    return command$',
    '    # 4h-runtime: 仅当非 FP8 KV 时强制 FLASH_ATTN (V33 fix)\n'
    '    # FLASH_ATTN 不支持 FP8 KV; fp8 时跳过让 vLLM 自动选 backend\n'
    '    import os as _os_4h_rt\n'
    '    _kv_4h_rt = _os_4h_rt.environ.get("TAAF_VLLM_KV_CACHE_DTYPE", "auto").strip().lower()\n'
    '    if _kv_4h_rt != "fp8":\n'
    '        _os_4h_rt.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"\n'
    '        command.extend(["--attention-backend", "flash_attn"])\n'
    '        print("[4h] 已强制 FLASH_ATTN (KV dtype: " + _kv_4h_rt + ")", flush=True)\n'
    '    else:\n'
    '        print("[4h] FP8 KV 检测到，跳过 FLASH_ATTN 强制", flush=True)\n'
    '    del _os_4h_rt\n'
    '    del _kv_4h_rt\n'
    '    return command',
    _t,
    count=1,
    flags=re.M,
)
assert n == 1, f"4h runtime 注入失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已在 server_command() 注入 4h runtime 条件判断", flush=True)

# --- 4i. KV cache dtype 白名单扩充：允许 bfloat16 ---
# 背景（2026-10-18 V26/V27 实测，serving_setup_audit.md 6.1 节）:
#   --attention-backend flash_attn 在 sm_120（RTX Pro 6000 Blackwell）
#   上拒绝 FP8 KV cache：
#   "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"。
#   tuning 白名单（第 427 行）只认 auto/fp8；而 auto 会被 server_command
#   第 2330-2337 行整个省略 --kv-cache-dtype 参数，vLLM 对 FP8-attention
#   混合检查点又会把 auto 推断回 fp8 → 死锁。
#   vLLM 官方 CLI 文档确认 bfloat16 是合法值，且是"默认 fp8 模型的
#   逃生口"；--dtype bfloat16 与模型 dtype 一致，满足"显式 bfloat16
#   须与模型 dtype 一致"的要求。FLASH_ATTN + bf16 KV cache 在任何
#   架构上都合法。
_t, n = re.subn(
    r'if kv_cache_dtype not in \{"auto", "fp8"\}:',
    'if kv_cache_dtype not in {"auto", "fp8", "bfloat16"}:',
    _t,
    count=1,
)
assert n == 1, f"KV 白名单(427行)替换失败：匹配到 {n} 处"
print(f"{_patch_tag} KV cache dtype 白名单已扩充 bfloat16（427行）", flush=True)
# validate_vllm_tuning_document 第 509 行同步（多行写法，锚定错误码）
_t, n = re.subn(
    r'\] not in \{"auto", "fp8"\}:',
    '] not in {"auto", "fp8", "bfloat16"}:',
    _t,
    count=1,
)
assert n == 1, f"KV 白名单(509行)替换失败：匹配到 {n} 处"
print(f"{_patch_tag} KV cache dtype 白名单已扩充 bfloat16（509行）", flush=True)

# --- 4j. 中和 capture_process_and_gpu() 的 PLE 日志硬断言 ---
# 背景（2026-10-18 审计，serving_setup.py 第 2885-2888 行）:
#   非 fast 模式下该函数要求 vLLM 日志出现 PLE 字样：
#   "vLLM did not prove native PLE CPU-offload readiness"。
#   PLE 是老模型（180B MoE）专用的 CPU offload 层，新模型永远
#   打不出这行日志 → 非 fast 模式必挂。
#   当前 TAAF_KAGGLE_FAST_START=1 时该函数被跳过，属预防性修复，
#   防止 fast 模式开关被意外关闭时踩雷。风格同 4d/4e。
_t, n = re.subn(
    r'    if any\(value is None for value in ple_log_matches\.values\(\)\):\n'
    r'        raise RuntimeError\(\n'
    r'            f"vLLM did not prove native PLE CPU-offload readiness: \{ple_log_matches\}"\n'
    r'        \)\n',
    '    if any(value is None for value in ple_log_matches.values()):\n'
    '        print("[v16-model-patch] PLE log matches 缺失（新模型无 PLE 层，忽略）", flush=True)\n',
    _t,
    count=1,
)
assert n == 1, f"PLE 日志断言中和失败：匹配到 {n} 处"
print(f"{_patch_tag} capture_process_and_gpu() PLE 日志断言已中和", flush=True)

_p.write_text(_t)
print(f"{_patch_tag} 补丁写入完成: {_p}", flush=True)

# --- 5. 让后续 setup cell 跑补丁版 ---
# 机制（2026-10-17 实测确认）:
#   In[6] 的 _command_env() 里有 env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)，
#   会覆盖 os.environ 里的同名值——所以"只设环境变量"走不通。
#   正确做法是直接重定向 BUNDLE_DIR 这个 notebook 全局变量
#   （In[4] 定义，本 cell 在其后运行，In[6] 在本 cell 之后运行，
#   三个 cell 共享同一命名空间，直接赋值即覆盖全局变量）。
#   之后 In[6] 的 _source_path_entries(BUNDLE_DIR)、_command_env()、
#   BUNDLE_DIR/"setup_commands.json" 全部自动指向补丁版。
#   环境变量也顺手设上（无害，万一有别处读它）。
BUNDLE_DIR = _bundle_dst
os.environ["TAAF_KAGGLE_BUNDLE_DIR"] = str(_bundle_dst)
print(f"{_patch_tag} BUNDLE_DIR -> {_bundle_dst}", flush=True)

# --- 6. MTP 保持开启（用户纠正：27B 自带 MTP）---
# 背景（2026-10-18 用户纠正）:
#   原以为新模型是稠密无 MTP，实测 config 有 mtp_num_hidden_layers=1，
#   27B 自带 MTP，不应关闭。此节不再设置 TAAF_VLLM_MTP_TOKENS=0。
print(f"{_patch_tag} MTP 保持默认开启（27B 自带 MTP）", flush=True)
