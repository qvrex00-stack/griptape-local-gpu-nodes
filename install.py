"""
Griptape Local GPU Nodes - 자동 설치 스크립트

사용법:
    python install.py --griptape-dir "C:/Foundry/Griptape"

또는 환경변수 사용:
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


def run(cmd, cwd=None, check=True):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print(f"  ERROR: Command failed with code {result.returncode}")
        sys.exit(1)
    return result


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

            # Compute Capability 감지
            parts = gpu_info.split(",")
            if len(parts) >= 3:
                cc = parts[2].strip().replace(".", "")
                cc_num = int(cc)
                if cc_num >= 120:  # Blackwell (RTX 5000)
                    return "nightly_cu128"
                elif cc_num >= 89:  # Ada (RTX 4000)
                    return "stable_cu121"
                elif cc_num >= 80:  # Ampere (RTX 3000)
                    return "stable_cu121"
                else:
                    return "stable_cu118"
    except Exception:
        pass

    print("  No NVIDIA GPU detected or nvidia-smi not found, using CPU.")
    return "cpu"


def get_torch_install_cmd(python_path, gpu_type):
    """GPU 타입에 맞는 PyTorch 설치 명령 반환"""
    base = ["uv", "pip", "install", "--python", python_path, "--pre" if "nightly" in gpu_type else ""]
    base = [x for x in base if x]  # 빈 문자열 제거

    if gpu_type == "nightly_cu128":
        return base + ["--upgrade", "--force-reinstall", "torch", "torchvision", "torchaudio",
                       "--index-url", "https://download.pytorch.org/whl/nightly/cu128"]
    elif gpu_type == "stable_cu121":
        return base + ["torch", "torchvision", "torchaudio",
                       "--index-url", "https://download.pytorch.org/whl/cu121"]
    elif gpu_type == "stable_cu118":
        return base + ["torch", "torchvision", "torchaudio",
                       "--index-url", "https://download.pytorch.org/whl/cu118"]
    else:
        return base + ["torch", "torchvision", "torchaudio"]


def install(griptape_dir: str):
    griptape_dir = Path(griptape_dir).resolve()
    repo_dir = Path(__file__).parent.resolve()

    print(f"\nGriptape directory: {griptape_dir}")
    print(f"Repo directory: {repo_dir}")

    if not griptape_dir.exists():
        print(f"ERROR: Griptape directory not found: {griptape_dir}")
        sys.exit(1)

    standard_lib = griptape_dir / "libraries" / "griptape-nodes-library-standard"
    if not standard_lib.exists():
        print(f"ERROR: Standard library not found: {standard_lib}")
        sys.exit(1)

    # ── Step 1: GPU 감지 ──────────────────────────────────────────
    print("\n[1/6] Detecting GPU...")
    gpu_type = detect_gpu()
    print(f"  GPU type: {gpu_type}")

    # ── Step 2: model_server 폴더 복사 ───────────────────────────
    print("\n[2/6] Setting up model server...")
    server_dst = griptape_dir / "model_server"
    server_src = repo_dir / "model_server"

    if not server_dst.exists():
        shutil.copytree(server_src, server_dst)
        print(f"  Copied to: {server_dst}")
    else:
        # server.py만 업데이트
        shutil.copy2(server_src / "server.py", server_dst / "server.py")
        shutil.copy2(server_src / "requirements.txt", server_dst / "requirements.txt")
        print(f"  Updated: {server_dst}")

    # ── Step 3: model_server .venv 생성 및 패키지 설치 ───────────
    print("\n[3/6] Setting up model server virtual environment...")
    server_venv = server_dst / ".venv"
    python_path = str(server_venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python"))

    if not server_venv.exists():
        run(["uv", "venv", str(server_venv), "--python", "3.12"])

    # PyTorch 설치
    torch_cmd = get_torch_install_cmd(python_path, gpu_type)
    print(f"  Installing PyTorch ({gpu_type})...")
    run(torch_cmd)

    # 나머지 패키지
    print("  Installing other packages...")
    run(["uv", "pip", "install", "--python", python_path,
         "-r", str(server_dst / "requirements.txt")])

    # ── Step 4: diffusers Qwen3 패치 ─────────────────────────────
    print("\n[4/6] Applying diffusers patches...")
    patch_script = repo_dir / "patches" / "apply_patches.py"

    # model_server venv 패치
    run([python_path, str(patch_script), str(server_venv)], check=False)

    # standard library venv 패치
    std_venv = standard_lib / ".venv"
    std_python = str(std_venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python"))
    if std_venv.exists() and os.path.exists(std_python):
        run([std_python, str(patch_script), str(std_venv)], check=False)

    # ── Step 5: 노드 파일 복사 ────────────────────────────────────
    print("\n[5/6] Installing node files...")
    nodes_src = repo_dir / "nodes"
    image_dst = standard_lib / "griptape_nodes_library" / "image"

    for node_file in ["local_image_generation.py", "local_image_inpaint.py"]:
        shutil.copy2(nodes_src / node_file, image_dst / node_file)
        print(f"  Copied: {node_file} -> standard library")

    # lama 라이브러리 설정
    lama_lib = griptape_dir / "libraries" / "griptape-nodes-library-lama"
    lama_image_dst = lama_lib / "griptape_nodes_library" / "image"
    lama_image_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nodes_src / "local_object_remover.py", lama_image_dst / "local_object_remover.py")
    print(f"  Copied: local_object_remover.py -> lama library")

    # lama library json 복사
    lama_json_src = repo_dir / "model_server" / "griptape-nodes-library-lama.json"
    if lama_json_src.exists():
        shutil.copy2(lama_json_src, lama_lib / "griptape_nodes_library.json")

    # ── Step 6: advanced 파일 복사 ────────────────────────────────
    print("\n[6/6] Installing advanced library file...")
    adv_src = repo_dir / "advanced" / "griptape_nodes_library_advanced.py"
    adv_dst = standard_lib / "griptape_nodes_library" / "griptape_nodes_library_advanced.py"
    shutil.copy2(adv_src, adv_dst)
    print(f"  Copied: griptape_nodes_library_advanced.py")

    # ── 완료 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Start Griptape Nodes")
    print(f"  2. In Settings > Libraries, register:")
    print(f"     {lama_lib}")
    print(f"  3. Restart Griptape - model server will start automatically")
    print(f"\nModel server URL: http://127.0.0.1:8088")
    print(f"GPU type detected: {gpu_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install Griptape Local GPU Nodes")
    parser.add_argument(
        "--griptape-dir",
        default=os.environ.get("GRIPTAPE_DIR", ""),
        help="Path to Griptape installation directory"
    )
    args = parser.parse_args()

    if not args.griptape_dir:
        print("ERROR: Please specify --griptape-dir or set GRIPTAPE_DIR environment variable")
        print("Example: python install.py --griptape-dir \"C:/Foundry/Griptape\"")
        sys.exit(1)

    install(args.griptape_dir)
