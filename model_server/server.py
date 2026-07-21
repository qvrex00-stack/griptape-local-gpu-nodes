"""
Griptape Local Model Server
FLUX.2-klein + LaMa local inference server

Usage:
    python server.py
"""

import io
import logging
import random
import os
import base64
import time
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_server")

# Pipeline cache
_pipelines = {}
_pipeline_times = {}
_lama_model = None
_lama_last_used = 0
_cache_lock = threading.Lock()

UNLOAD_TIMEOUT = 300  # 5 minutes


def unload_idle_pipelines():
    """Background thread: offload idle pipelines from GPU after timeout"""
    while True:
        time.sleep(60)
        now = time.time()
        with _cache_lock:
            import torch
            for key in list(_pipeline_times.keys()):
                if now - _pipeline_times[key] > UNLOAD_TIMEOUT:
                    if key in _pipelines:
                        try:
                            _pipelines[key].to("cpu")
                            torch.cuda.empty_cache()
                            logger.info(f"[Auto-unload] Offloaded to CPU: {key}")
                        except Exception as e:
                            logger.warning(f"Could not offload {key}: {e}")
            global _lama_model, _lama_last_used
            if _lama_model is not None and now - _lama_last_used > UNLOAD_TIMEOUT:
                try:
                    _lama_model.model.to("cpu")
                    torch.cuda.empty_cache()
                    logger.info("[Auto-unload] LaMa offloaded to CPU")
                except Exception:
                    pass


class GenerateRequest(BaseModel):
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 8
    guidance_scale: float = 0.0
    seed: int = -1
    lora_path: str = ""
    lora_scale: float = 0.8


class InpaintRequest(BaseModel):
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    prompt: str
    image_b64: str
    mask_b64: str
    num_inference_steps: int = 8
    guidance_scale: float = 0.0
    seed: int = -1
    lora_path: str = ""
    lora_scale: float = 0.8


class RemoveRequest(BaseModel):
    image_b64: str
    mask_b64: str


class ImageResponse(BaseModel):
    image_b64: str
    seed_used: int
    width: int
    height: int


def b64_to_pil(b64_str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def pil_to_b64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def get_pipeline(model_id, pipeline_class, torch_dtype, device):
    import torch
    cache_key = f"{model_id}_{pipeline_class.__name__}_{torch_dtype}_{device}"
    with _cache_lock:
        if cache_key not in _pipelines:
            logger.info(f"Loading {pipeline_class.__name__} from {model_id}...")
            pipe = pipeline_class.from_pretrained(model_id, torch_dtype=torch_dtype)
            pipe.to(device)
            try:
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
            except Exception:
                pass
            _pipelines[cache_key] = pipe
            logger.info(f"Pipeline loaded and cached: {cache_key}")
        else:
            pipe = _pipelines[cache_key]
            try:
                pipe.to(device)
            except Exception:
                pass
        _pipeline_times[cache_key] = time.time()
    return _pipelines[cache_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    logger.info("=" * 50)
    logger.info("Griptape Local Model Server Starting...")
    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"Auto-unload timeout: {UNLOAD_TIMEOUT}s ({UNLOAD_TIMEOUT//60} min)")
    logger.info("=" * 50)
    threading.Thread(target=unload_idle_pipelines, daemon=True).start()
    yield
    logger.info("Server shutting down...")


app = FastAPI(
    title="Griptape Local Model Server",
    description="FLUX.2-klein + LaMa local inference server",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    import torch
    loaded = [{"key": k, "idle_sec": int(time.time() - ts)} for k, ts in _pipeline_times.items()]
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_pipelines": loaded,
        "unload_timeout_sec": UNLOAD_TIMEOUT,
    }


@app.post("/generate", response_model=ImageResponse)
def generate(req: GenerateRequest):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model_lower = req.model_id.lower()
    is_klein = "klein" in model_lower or "flux.2" in model_lower
    torch_dtype = torch.float16 if is_klein else torch.bfloat16
    try:
        if is_klein:
            from diffusers import Flux2KleinPipeline
            pipeline_class = Flux2KleinPipeline
        else:
            from diffusers import FluxPipeline
            pipeline_class = FluxPipeline
        pipe = get_pipeline(req.model_id, pipeline_class, torch_dtype, device)
        if req.lora_path and os.path.exists(req.lora_path):
            pipe.load_lora_weights(req.lora_path, adapter_name="lora")
            pipe.set_adapters("lora", adapter_weights=float(req.lora_scale))
        result = pipe(
            prompt=req.prompt, width=req.width, height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=0.0 if is_klein else req.guidance_scale,
            generator=generator,
        )
        img = result.images[0]
        logger.info(f"Generation complete: {img.width}x{img.height}")
        return ImageResponse(image_b64=pil_to_b64(img), seed_used=seed, width=img.width, height=img.height)
    except Exception as e:
        logger.error(f"Generate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inpaint", response_model=ImageResponse)
def inpaint(req: InpaintRequest):
    import torch
    import numpy as np
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pil_image = b64_to_pil(req.image_b64).convert("RGB")
    pil_mask = b64_to_pil(req.mask_b64).convert("L")
    orig_w, orig_h = pil_image.size
    width = (orig_w // 32) * 32 or 32
    height = (orig_h // 32) * 32 or 32
    if width != orig_w or height != orig_h:
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    pil_mask = pil_mask.resize((width, height), Image.Resampling.NEAREST)
    mask_np = (np.array(pil_mask) > 127).astype(np.uint8) * 255
    pil_mask_binary = Image.fromarray(mask_np, mode="L")
    try:
        from diffusers import Flux2KleinInpaintPipeline
        pipe = get_pipeline(req.model_id, Flux2KleinInpaintPipeline, torch.bfloat16, device)
        try:
            pil_mask_binary = pipe.mask_processor.blur(pil_mask_binary, blur_factor=12)
        except Exception:
            pass
        result = pipe(
            prompt=req.prompt, image=pil_image, mask_image=pil_mask_binary,
            height=height, width=width, num_inference_steps=req.num_inference_steps,
            guidance_scale=0.0, strength=1.0, generator=generator,
        )
        img = result.images[0]
        logger.info(f"Inpaint complete: {img.width}x{img.height}")
        return ImageResponse(image_b64=pil_to_b64(img), seed_used=seed, width=img.width, height=img.height)
    except Exception as e:
        logger.error(f"Inpaint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remove", response_model=ImageResponse)
def remove(req: RemoveRequest):
    global _lama_model, _lama_last_used
    import torch
    import numpy as np
    from PIL import Image
    pil_image = b64_to_pil(req.image_b64).convert("RGB")
    pil_mask = b64_to_pil(req.mask_b64).convert("L")
    if pil_mask.size != pil_image.size:
        pil_mask = pil_mask.resize(pil_image.size, Image.Resampling.NEAREST)
    mask_np = (np.array(pil_mask) > 127).astype(np.uint8) * 255
    pil_mask_binary = Image.fromarray(mask_np, mode="L")
    try:
        from simple_lama_inpainting import SimpleLama
        with _cache_lock:
            if _lama_model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading LaMa on {device}...")
                _lama_model = SimpleLama(device=torch.device(device))
                logger.info("LaMa loaded.")
            else:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    _lama_model.model.to(device)
                    _lama_model.device = torch.device(device)
                except Exception:
                    pass
            _lama_last_used = time.time()
        result = _lama_model(pil_image, pil_mask_binary)
        logger.info(f"LaMa complete: {result.size}")
        return ImageResponse(image_b64=pil_to_b64(result), seed_used=0, width=result.width, height=result.height)
    except Exception as e:
        logger.error(f"Remove failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8088, log_level="info")
