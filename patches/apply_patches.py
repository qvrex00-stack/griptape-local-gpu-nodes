"""
diffusers Qwen3 패치 적용 스크립트
install.py에서 자동 실행됨
"""

import os
import sys


PATCH = '''from transformers import Qwen2TokenizerFast
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
except Exception:
    from transformers.modeling_utils import PreTrainedModel as Qwen3ForCausalLM'''

OLD_PATTERNS = [
    'from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM',
    '''from transformers import Qwen2TokenizerFast
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
except ImportError:
    from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM''',
]

TARGET_FILES = [
    "pipeline_flux2_klein.py",
    "pipeline_flux2_klein_inpaint.py",
    "pipeline_flux2_klein_kv.py",
]


def apply_patches(venv_path: str) -> bool:
    site_packages = os.path.join(venv_path, "Lib", "site-packages")
    flux2_dir = os.path.join(site_packages, "diffusers", "pipelines", "flux2")

    if not os.path.exists(flux2_dir):
        print(f"[Patch] diffusers flux2 dir not found: {flux2_dir}")
        return False

    patched = 0
    for fname in TARGET_FILES:
        fpath = os.path.join(flux2_dir, fname)
        if not os.path.exists(fpath):
            print(f"[Patch] File not found: {fname}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        if PATCH in content:
            print(f"[Patch] Already patched: {fname}")
            patched += 1
            continue

        replaced = False
        for old in OLD_PATTERNS:
            if old in content:
                content = content.replace(old, PATCH)
                replaced = True
                break

        if replaced:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[Patch] Applied: {fname}")
            patched += 1
        else:
            print(f"[Patch] Could not find pattern in: {fname}")

    return patched > 0


if __name__ == "__main__":
    venv = sys.argv[1] if len(sys.argv) > 1 else None
    if not venv:
        print("Usage: python apply_patches.py <venv_path>")
        sys.exit(1)
    success = apply_patches(venv)
    sys.exit(0 if success else 1)
