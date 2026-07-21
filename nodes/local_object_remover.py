from __future__ import annotations

import base64
import io
import logging
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.files.file import File

logger = logging.getLogger("griptape_nodes")

__all__ = ["LocalObjectRemover"]

SERVER_URL = "http://127.0.0.1:8088"


class LocalObjectRemover(BaseNode):
    """Remove objects using LaMa via local model server - no prompt needed."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Image Nodes"
        self.description = "Remove objects using LaMa AI (via local model server). White mask = remove area."

        self.add_parameter(ParameterImage(name="input_image", tooltip="Source image", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterImage(name="input_mask", tooltip="White mask over object to remove", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterImage(name="output", tooltip="Image with object removed", allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY}, settable=False, pulse_on_run=True))
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="local_removed.png")
        self._output_file.add_parameter()

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output"] = None
        import requests
        from PIL import Image

        image_param = self.get_parameter_value("input_image")
        mask_param = self.get_parameter_value("input_mask")

        if not image_param or not mask_param:
            raise ValueError("input_image and input_mask are required.")

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

        payload = {"image_b64": pil_to_b64(pil_image), "mask_b64": pil_to_b64(pil_mask)}

        try:
            resp = requests.post(f"{SERVER_URL}/remove", json=payload, timeout=120)
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
        logger.info(f"Remove success: {saved.name}")
