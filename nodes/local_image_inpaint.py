from __future__ import annotations

import base64
import io
import logging
import random
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes")

__all__ = ["LocalImageInpaint"]

SERVER_URL = "http://127.0.0.1:8088"

MODEL_OPTIONS = [
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9B",
]


class LocalImageInpaint(BaseNode):
    """Inpaint images locally using FLUX.2-klein via local model server."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Image Nodes"
        self.description = "Inpaint images locally using FLUX.2-klein (via local model server)."

        self.add_parameter(ParameterImage(name="input_image", tooltip="Source image to inpaint", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterImage(name="input_mask", tooltip="White mask = inpaint area, black = keep", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterString(name="model_id_or_path", default_value=MODEL_OPTIONS[0], tooltip="FLUX.2-klein model ID", allow_output=False, traits={Options(choices=MODEL_OPTIONS)}, ui_options={"display_name": "Model Path/ID"}))
        self.add_parameter(ParameterString(name="prompt", tooltip="What to generate in the masked area", multiline=True, placeholder_text="A red glowing orb...", allow_output=False))
        self.add_parameter(ParameterInt(name="num_inference_steps", default_value=8, tooltip="Inference steps", allow_output=False, min_val=1, max_val=50))
        self.add_parameter(ParameterBool(name="randomize_seed", default_value=True, tooltip="Randomize seed", allow_output=False))
        self.add_parameter(ParameterInt(name="seed", default_value=42, tooltip="Random seed", allow_output=False))

        with ParameterGroup(name="LoRA Config", ui_options={"collapsed": True}) as lora_group:
            ParameterString(name="lora_path", default_value="", tooltip="Optional LoRA weights path", allow_output=False)
            ParameterFloat(name="lora_scale", default_value=0.8, tooltip="LoRA scale", allow_output=False, min_val=0.0, max_val=2.0)
        self.add_node_element(lora_group)

        self.add_parameter(ParameterImage(name="output", tooltip="Inpainted image", allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY}, settable=False, pulse_on_run=True))
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="local_inpainted.png")
        self._output_file.add_parameter()

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output"] = None
        import requests
        from PIL import Image

        model_id = self.get_parameter_value("model_id_or_path") or MODEL_OPTIONS[0]
        prompt = self.get_parameter_value("prompt") or ""
        steps = self.get_parameter_value("num_inference_steps") or 8
        randomize = self.get_parameter_value("randomize_seed") or False
        seed = self.get_parameter_value("seed") or 42
        lora_path = self.get_parameter_value("lora_path") or ""
        lora_scale = self.get_parameter_value("lora_scale") or 0.8
        image_param = self.get_parameter_value("input_image")
        mask_param = self.get_parameter_value("input_mask")

        if not image_param or not mask_param:
            raise ValueError("input_image and input_mask are required.")
        if not prompt:
            raise ValueError("Prompt is required.")
        if randomize:
            seed = random.randint(0, 2**32 - 1)
            self.set_parameter_value("seed", seed)

        try:
            requests.get(f"{SERVER_URL}/health", timeout=3)
        except Exception:
            raise RuntimeError("Model server is not running! See README.md for instructions.")

        def resolve_pil(param_val):
            image_path = None
            if isinstance(param_val, str): image_path = param_val
            elif hasattr(param_val, "value"): image_path = param_val.value
            elif isinstance(param_val, dict) and "value" in param_val: image_path = param_val["value"]
            if image_path:
                try:
                    resolved = File(image_path).resolve()
                    path = resolved.location if hasattr(resolved, "location") else str(resolved)
                except Exception:
                    path = image_path
                return Image.open(path)
            raise ValueError("Could not extract image.")

        pil_image = resolve_pil(image_param).convert("RGB")
        pil_mask = resolve_pil(mask_param).convert("L")

        def pil_to_b64(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        payload = {"model_id": model_id, "prompt": prompt, "image_b64": pil_to_b64(pil_image), "mask_b64": pil_to_b64(pil_mask), "num_inference_steps": steps, "guidance_scale": 0.0, "seed": seed, "lora_path": lora_path, "lora_scale": lora_scale}

        try:
            resp = requests.post(f"{SERVER_URL}/inpaint", json=payload, timeout=300)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Server request failed: {e}")

        result = resp.json()
        img_bytes = base64.b64decode(result["image_b64"])
        dest = self._output_file.build_file()
        saved = dest.write_bytes(img_bytes)
        url_artifact = ImageUrlArtifact(saved.location)
        self.set_parameter_value("output", url_artifact)
        self.publish_update_to_parameter("output", url_artifact)
        logger.info(f"Inpaint success: {saved.name}")
