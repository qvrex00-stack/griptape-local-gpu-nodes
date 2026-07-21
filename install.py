"""
Griptape Local GPU Nodes - Auto Install Script

Usage:
    python install.py --griptape-dir "C:/Foundry/Griptape"

Or with environment variable:
    set GRIPTAPE_DIR=C:/Foundry/Griptape
    python install.py
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None, check=True, env=None):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, env=env or os.environ.copy())
    if check and result.returncode != 0:
        print(f"  ERROR: Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def find_uv():
    """uv 실행 파일 경로 찾기"""
    # 일반 PATH에서 찾기
    uv = shutil.which("uv")
    if uv:
        return uv

    # Windows 일반 설치 경로
    if platform.system() == "Windows":
        candidates = [
            Path.home() / ".cargo" / "bin" / "uv.exe",
            Path.home() / "AppData" / "Local" / "uv" / "bin" / "uv.exe",
            Path.home() / ".local" / "bin" / "uv.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    else:
        candidates = [
            Path.home() / ".cargo" / "bin" / "uv",
            Path.home() / ".local" / "bin" / "uv",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    return None


def install_uv():
    """uv 자동 설치"""
    print("  uv not found. Installing uv...")
    if platform.system() == "Windows":
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command",
             "irm https://astral.sh/uv/install.ps1 | iex"],
            capture_output=False
        )
    else:
        result = subprocess.run(
            ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            capture_output=False
        )

    if result.returncode != 0:
        print("  ERROR: uv installation failed.")
        print("  Please install manually: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

    # PATH 업데이트 후 재탐색
    if platform.system() == "Windows":
        os.environ["PATH"] = (
            str(Path.home() / "AppData" / "Local" / "uv" / "bin") + ";" +
            str(Path.home() / ".cargo" / "bin") + ";" +
            os.environ.get("PATH", "")
        )
    else:
        os.environ["PATH"] = (
            str(Path.home() / ".local" / "bin") + ":" +
            str(Path.home() / ".cargo" / "bin") + ":" +
            os.environ.get("PATH", "")
        )

    uv = find_uv()
    if not uv:
        print("  ERROR: uv installed but not found in PATH.")
        print("  Please restart your terminal and run install.py again.")
        sys.exit(1)

    print(f"  uv installed: {uv}")
    return uv


def get_uv():
    """uv 경로 반환 (없으면 설치)"""
    uv = find_uv()
    if not uv:
        uv = install_uv()
    else:
        print(f"  uv found: {uv}")
    return uv


def detect_gpu():
    """GPU 종류 감지 후 적절한 PyTorch index URL 반환"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpu_info = lines[0] if lines else ""
            print(f"  Detected NVIDIA GPU: {gpu_info}")

            parts = gpu_info.split(",")
            if len(parts) >= 3:
                cc = parts[2].strip().replace(".", "")
                try:
                    cc_num = int(cc)
                    if cc_num >= 120:    # Blackwell (RTX 5000 series)
                        return "nightly_cu128"
                    elif cc_num >= 89:   # Ada Lovelace (RTX 4000 series)
                        return "stable_cu121"
                    elif cc_num >= 80:   # Ampere (RTX 3000 series)
                        return "stable_cu121"
                    elif cc_num >= 75:   # Turing (RTX 2000 series)
                        return "stable_cu118"
                    else:               # Older GPUs
                        return "stable_cu118"
                except ValueError:
                    pass
    except Exception:
        pass

    print("  No NVIDIA GPU detected or nvidia-smi not found. Will use CPU.")
    return "cpu"


def get_torch_install_cmd(uv, python_path, gpu_type):
    """GPU 타입에 맞는 PyTorch 설치 명령 반환"""
    base = [uv, "pip", "install", "--python", python_path]
    pre = ["--pre"] if "nightly" in gpu_type else []

    if gpu_type == "nightly_cu128":
        return base + pre + ["--upgrade", "--force-reinstall",
                              "torch", "torchvision", "torchaudio",
                              "--index-url", "https://download.pytorch.org/whl/nightly/cu128"]
    elif gpu_type in ("stable_cu121",):
        return base + ["torch", "torchvision", "torchaudio",
                       "--index-url", "https://download.pytorch.org/whl/cu121"]
    elif gpu_type == "stable_cu118":
        return base + ["torch", "torchvision", "torchaudio",
                       "--index-url", "https://download.pytorch.org/whl/cu118"]
    else:  # cpu
        return base + ["torch", "torchvision", "torchaudio"]


def install(griptape_dir: str):
    griptape_dir = Path(griptape_dir).resolve()
    repo_dir = Path(__file__).parent.resolve()

    print(f"\nGriptape directory: {griptape_dir}")
    print(f"Repo directory:     {repo_dir}")
    print(f"Platform:           {platform.system()} {platform.machine()}")

    if not griptape_dir.exists():
        print(f"ERROR: Griptape directory not found: {griptape_dir}")
        sys.exit(1)

    standard_lib = griptape_dir / "libraries" / "griptape-nodes-library-standard"
    if not standard_lib.exists():
        print(f"ERROR: Standard library not found: {standard_lib}")
        print("Please make sure Griptape Nodes is installed first.")
        sys.exit(1)

    # ── Step 0: uv 확인/설치 ──────────────────────────────────────
    print("\n[0/6] Checking prerequisites...")
    uv = get_uv()

    # ── Step 1: GPU 감지 ──────────────────────────────────────────
    print("\n[1/6] Detecting GPU...")
    gpu_type = detect_gpu()
    print(f"  GPU type: {gpu_type}")

    # ── Step 2: model_server 폴더 설정 ───────────────────────────
    print("\n[2/6] Setting up model server...")
    server_dst = griptape_dir / "model_server"
    server_src = repo_dir / "model_server"

    if not server_dst.exists():
        shutil.copytree(server_src, server_dst, ignore=shutil.ignore_patterns(".venv"))
        print(f"  Copied to: {server_dst}")
    else:
        shutil.copy2(server_src / "server.py", server_dst / "server.py")
        shutil.copy2(server_src / "requirements.txt", server_dst / "requirements.txt")
        print(f"  Updated: {server_dst}")

    # ── Step 3: model_server .venv 생성 및 패키지 설치 ───────────
    print("\n[3/6] Setting up model server virtual environment...")
    server_venv = server_dst / ".venv"
    if platform.system() == "Windows":
        python_path = str(server_venv / "Scripts" / "python.exe")
    else:
        python_path = str(server_venv / "bin" / "python")

    if not server_venv.exists():
        print("  Creating virtual environment with Python 3.12...")
        run([uv, "venv", str(server_venv), "--python", "3.12"])
    else:
        print("  Virtual environment already exists, skipping creation.")

    print(f"  Installing PyTorch ({gpu_type})...")
    torch_cmd = get_torch_install_cmd(uv, python_path, gpu_type)
    run(torch_cmd)

    print("  Installing other packages...")
    run([uv, "pip", "install", "--python", python_path,
         "-r", str(server_dst / "requirements.txt")])

    # ── Step 4: diffusers Qwen3 패치 ─────────────────────────────
    print("\n[4/6] Applying diffusers patches...")
    patch_script = repo_dir / "patches" / "apply_patches.py"

    run([python_path, str(patch_script), str(server_venv)], check=False)

    std_venv = standard_lib / ".venv"
    if platform.system() == "Windows":
        std_python = str(std_venv / "Scripts" / "python.exe")
    else:
        std_python = str(std_venv / "bin" / "python")

    if std_venv.exists() and os.path.exists(std_python):
        print("  Patching standard library venv...")
        run([std_python, str(patch_script), str(std_venv)], check=False)

    # ── Step 5: 노드 파일 복사 ────────────────────────────────────
    print("\n[5/6] Installing node files...")
    nodes_src = repo_dir / "nodes"
    image_dst = standard_lib / "griptape_nodes_library" / "image"

    for node_file in ["local_image_generation.py", "local_image_inpaint.py"]:
        shutil.copy2(nodes_src / node_file, image_dst / node_file)
        print(f"  Copied: {node_file} -> standard library")

    lama_lib = griptape_dir / "libraries" / "griptape-nodes-library-lama"
    lama_image_dst = lama_lib / "griptape_nodes_library" / "image"
    lama_image_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nodes_src / "local_object_remover.py", lama_image_dst / "local_object_remover.py")
    print("  Copied: local_object_remover.py -> lama library")

    lama_json_src = repo_dir / "model_server" / "griptape-nodes-library-lama.json"
    if lama_json_src.exists():
        shutil.copy2(lama_json_src, lama_lib / "griptape_nodes_library.json")
        print("  Copied: griptape_nodes_library.json -> lama library")

    # ── Step 6: advanced 파일 복사 ────────────────────────────────
    print("\n[6/6] Installing advanced library file...")
    adv_src = repo_dir / "advanced" / "griptape_nodes_library_advanced.py"
    adv_dst = standard_lib / "griptape_nodes_library" / "griptape_nodes_library_advanced.py"
    shutil.copy2(adv_src, adv_dst)
    print("  Copied: griptape_nodes_library_advanced.py")

    # ── 완료 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start Griptape Nodes")
    print("  2. In Settings > Libraries, register:")
    print(f"     {lama_lib}")
    print("  3. Restart Griptape - model server will auto-start")
    print("\nModel server URL: http://127.0.0.1:8088")
    print(f"GPU type detected: {gpu_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install Griptape Local GPU Nodes")
    parser.add_argument(
        "--griptape-dir",
        default=os.environ.get("GRIPTAPE_DIR", ""),
        help="Path to Griptape installation directory (e.g. C:/Foundry/Griptape)"
    )
    args = parser.parse_args()

    if not args.griptape_dir:
        print("ERROR: Please specify --griptape-dir or set GRIPTAPE_DIR environment variable")
        print('Example: python install.py --griptape-dir "C:/Foundry/Griptape"')
        sys.exit(1)

    install(args.griptape_dir)
