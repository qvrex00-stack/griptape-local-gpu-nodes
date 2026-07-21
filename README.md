# Griptape Local GPU Nodes

FLUX.2-klein 및 LaMa를 Griptape Nodes에서 로컬 GPU로 실행하기 위한 노드 패키지입니다.

## 포함된 노드

| 노드 | 설명 |
|------|------|
| **Local Image Generation** | FLUX.2-klein으로 이미지 생성 (Txt2Img / Img2Img) |
| **Local Image Inpaint** | FLUX.2-klein으로 인페인팅 (마스크 영역 교체) |
| **Local Object Remover** | LaMa AI로 오브젝트 제거 (프롬프트 불필요) |

## 아키텍처

```
Griptape Desktop
└── Node (HTTP 요청)
        ↓
    Model Server (FastAPI, port 8088)
        ├── FLUX.2-klein (GPU)
        └── LaMa (GPU/CPU)
```

Griptape와 모델이 별도 프로세스로 실행되어 torch 버전 충돌 없이 안정적으로 동작합니다.

---

## 요구사항

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.12
- **GPU**: NVIDIA GPU 권장 (CPU도 동작하나 느림)
  - RTX 5000 series (Blackwell): CUDA 12.8 Nightly PyTorch 자동 설치
  - RTX 4000/3000 series: CUDA 12.1 stable PyTorch 자동 설치
  - RTX 2000 이하: CUDA 11.8 stable PyTorch 자동 설치
- **VRAM**: 최소 8GB (FLUX Klein 4B 기준)
- **Griptape Nodes Desktop**: 설치되어 있어야 함
- **uv**: [설치 링크](https://docs.astral.sh/uv/getting-started/installation/)

---

## 설치

### 1. 이 레포 클론

```bash
git clone https://github.com/qvrex00-stack/griptape-local-gpu-nodes.git
cd griptape-local-gpu-nodes
```

### 2. 설치 스크립트 실행

**Windows:**
```powershell
python install.py --griptape-dir "C:\Foundry\Griptape"
```

**Linux/macOS:**
```bash
python install.py --griptape-dir "/path/to/Griptape"
```

또는 환경변수 사용:
```bash
export GRIPTAPE_DIR="/path/to/Griptape"
python install.py
```

설치 스크립트가 자동으로:
1. GPU 감지 및 적절한 PyTorch 버전 설치
2. Model Server 가상환경 생성 및 패키지 설치
3. diffusers Qwen3 패치 적용
4. 노드 파일 복사
5. Griptape 시작 시 서버 자동 실행 설정

### 3. Griptape 설정

1. **Griptape Nodes** 실행
2. **Settings → Libraries → Add Library** 에서 다음 경로 등록:
   ```
   <griptape_dir>/libraries/griptape-nodes-library-lama
   ```
3. **Griptape 재시작** → 모델 서버가 자동으로 시작됩니다

---

## 수동 서버 실행

자동 시작이 안 될 경우 수동으로 실행:

**Windows:**
```powershell
& "<griptape_dir>\model_server\.venv\Scripts\python.exe" "<griptape_dir>\model_server\server.py"
```

**Linux/macOS:**
```bash
<griptape_dir>/model_server/.venv/bin/python <griptape_dir>/model_server/server.py
```

서버 상태 확인:
```
http://127.0.0.1:8088/health
```

---

## GPU 설정 (환경변수)

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `GRIPTAPE_MODEL_SERVER_DIR` | 자동 감지 | 모델 서버 디렉토리 경로 |

---

## 트러블슈팅

### 서버가 시작되지 않는 경우
1. `uv`가 설치되어 있는지 확인
2. `install.py`를 다시 실행
3. 수동으로 서버 실행 후 오류 메시지 확인

### GPU를 사용하지 않는 경우
```
http://127.0.0.1:8088/health
```
에서 `cuda_available: false` 이면 PyTorch가 CPU 버전입니다.

```powershell
# CUDA 버전 재설치 (RTX 5000 series 기준)
uv pip install --python "<model_server>/.venv/Scripts/python.exe" --pre --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### diffusers 오류
```
Could not import module 'Qwen3ForCausalLM'
```
패치를 다시 적용하세요:
```bash
python patches/apply_patches.py "<model_server>/.venv"
```

---

## GPU 메모리 관리

- 요청 없이 **5분**이 지나면 자동으로 GPU 메모리 해제
- 다음 요청 시 자동으로 GPU에 다시 로드
- `/health` 엔드포인트에서 현재 상태 확인 가능

---

## 모델 서버 엔드포인트

| 엔드포인트 | 방법 | 설명 |
|-----------|------|------|
| `/health` | GET | 서버 상태 확인 |
| `/generate` | POST | FLUX Klein 이미지 생성 |
| `/inpaint` | POST | FLUX Klein 인페인팅 |
| `/remove` | POST | LaMa 오브젝트 제거 |

---

## 라이선스

MIT License
