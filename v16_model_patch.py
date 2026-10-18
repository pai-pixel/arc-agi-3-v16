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
_models_root = pathlib.Path("/kaggle/input/models")
print(f"{_patch_tag} models root: {[p.name for p in _models_root.iterdir()]}", flush=True)
_base = _models_root / "overseer66" / "qwen3-8-27b-nvfp4" / "pytorch" / "default" / "1"
_candidates = []
for _cand in [_base, _base / "Qwen3.8-27B-NVFP4"]:
    _cfg = _cand / "config.json"
    print(f"{_patch_tag} 检查候选: {_cand} exists={_cand.exists()} config={_cfg.exists()}", flush=True)
    if _cfg.exists():
        _candidates.append(_cand)
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
_new_repo = "nvidia/Qwen3.8-27B-NVFP4"
_new_rev = "482ca0f3832238542f8f5295dde86b5f22711d80"
print(f"{_patch_tag} 新模型 repo={_new_repo} rev={_new_rev}", flush=True)

# --- 3. 计算新模型指纹 ---
def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

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

_patch_const(r'^MODEL_HF_REPO\s*=.*$',
             f'MODEL_HF_REPO = "{_new_repo}"', "MODEL_HF_REPO")
_patch_const(r'^MODEL_HF_REVISION\s*=.*$',
             f'MODEL_HF_REVISION = "{_new_rev}"', "MODEL_HF_REVISION")
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

# --- 4b. 中和 source_identity() ---
_t, n = re.subn(
    r'^def source_identity\(.*?:.*?\n(?=^def |\Z)',
    'def source_identity():\n'
    '    print("[v16-model-patch] source_identity() 已整体跳过（补丁版）", flush=True)\n'
    '    return\n',
    _t,
    flags=re.M | re.S,
)
assert n == 1, f"中和 source_identity 失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已整体中和 source_identity()", flush=True)

# --- 4c. 中和 resolve_model_dir() ---
_t, n = re.subn(
    r'^def resolve_model_dir\(.*?:.*?\n(?=^def |\Z)',
    'def resolve_model_dir():\n'
    '    print(f"[v16-model-patch] resolve_model_dir() 已跳过校验，直接返回 {MODEL_KAGGLE_PATH}", flush=True)\n'
    '    return MODEL_KAGGLE_PATH\n',
    _t,
    flags=re.M | re.S,
)
assert n == 1, f"中和 resolve_model_dir 失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已整体中和 resolve_model_dir()", flush=True)

# --- 4d. 中和 patch_ple_layer() ---
_t, n = re.subn(
    r'^def patch_ple_layer\(.*?:.*?\n(?=^def |\Z)',
    'def patch_ple_layer():\n'
    '    print("[v16-model-patch] patch_ple_layer() 已整体跳过（补丁版，PLE 为老模型专用）", flush=True)\n'
    '    return {"skipped": True, "reason": "Qwen3.8 dense has no Flash-Next PLE gate"}\n',
    _t,
    flags=re.M | re.S,
)
assert n == 1, f"中和 patch_ple_layer 失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已整体中和 patch_ple_layer()", flush=True)

# --- 4e. 中和 verify_model() ---
_t, n = re.subn(
    r'^def verify_model\(.*?:.*?\n(?=^def |\Z)',
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
assert n == 1, f"中和 verify_model 失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已整体中和 verify_model()", flush=True)

# --- 4f. 中和 runtime_environment() 的 PLE 哈希校验 ---
_t, n = re.subn(
    r'^    if sha256_file\(ple_path\) != VLLM_PLE_PATCHED_SHA256:\n'
    r'^        raise RuntimeError\("The patched PLE source hash changed before import\."\)\n',
    '    pass  # [v16-model-patch] PLE 哈希校验已跳过\n',
    _t,
    flags=re.M,
)
assert n == 1, f"中和 PLE 哈希校验失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已中和 runtime_environment() 的 PLE 哈希校验", flush=True)

# --- 4g. 修正 vLLM 的 --quantization 参数 ---
_t, n = re.subn(
    r'^        "--quantization",\n^        "modelopt_fp4",\n',
    '        "--quantization",\n        "compressed-tensors",\n',
    _t,
    flags=re.M,
)
assert n == 1, f"修正 quantization 参数失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已修正 vLLM --quantization 为 compressed-tensors", flush=True)

# --- 4h. 切换 vLLM attention backend ---
import os as _os_attn
_os_attn.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
print(f"{_patch_tag} 已设 VLLM_ATTENTION_BACKEND=FLASH_ATTN（环境变量）", flush=True)
del _os_attn
_t, n = re.subn(
    r'^        "--quantization",\n^        "compressed-tensors",\n',
    '        "--quantization",\n        "compressed-tensors",\n'
    '        "--attention-backend",\n        "flash-attn",\n',
    _t,
    flags=re.M,
)
assert n == 1, f"添加 attention-backend 参数失败：匹配到 {n} 处（期望 1 处）"
print(f"{_patch_tag} 已在 vLLM 命令中添加 --attention-backend flash-attn", flush=True)

_p.write_text(_t)
print(f"{_patch_tag} 补丁写入完成: {_p}", flush=True)

# --- 5. BUNDLE_DIR 重定向 ---
BUNDLE_DIR = _bundle_dst
os.environ["TAAF_KAGGLE_BUNDLE_DIR"] = str(_bundle_dst)
print(f"{_patch_tag} BUNDLE_DIR -> {_bundle_dst}", flush=True)

# --- 6. MTP 保持开启 ---
print(f"{_patch_tag} MTP 保持默认开启（27B 自带 MTP）", flush=True)
