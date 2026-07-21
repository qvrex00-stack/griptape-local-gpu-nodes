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

__all__ = ["LocalImageGeneration"]

SERVER_URL = "http://127.0.0.1:8088"

MODEL_OPTIONS = [
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9B",
]


class LocalImageGeneration(BaseNode):
    """Generate images locally using FLUX.2-klein via local model server."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Image Nodes"
        self.description = "Generate images locally using FLUX.2-klein (via local model server)."

        self.add_parameter(
            ParameterImage(
                name="input_image",
                tooltip="Optional source image for img2img",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Input Image (Optional)"}
            )
        )

        self.add_parameter(
            ParameterString(
                name="model_id_or_path",
                default_value=MODEL_OPTIONS[0],
                tooltip="FLUX.2-klein model ID",
                allow_output=False,
                traits={Options(choices=MODEL_OPTIONS)},
                ui_options={"display_name": "Model Path/ID"}
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the image to generate",
                multiline=True,
                placeholder_text="A cinematic close-up of a futuristic spaceship...",
                allow_output=False
            )
        )

        self.add_parameter(
            ParameterInt(
                name="num_inference_steps",
                default_value=8,
                tooltip="Number of inference steps (4-12 recommended for FLUX Klein)",
                allow_output=False,
                min_val=1,
                max_val=50
            )
        )

        self.add_parameter(
            ParameterInt(
                name="width",
                default_value=1024,
                tooltip="Output width (multiples of 32)",
                allow_output=False,
                min_val=256,
                max_val=1536,
                step=32
            )
        )

        self.add_parameter(
            ParameterInt(
                name="height",
                default_value=1024,
                tooltip="Output height (multiples of 32)",
                allow_output=False,
                min_val=256,
                max_val=1536,
                step=32
            )
        )

        self.add_parameter(
            ParameterBool(
                name="randomize_seed",
                default_value=True,
                tooltip="Randomize seed on each run",
                allow_output=False
            )
        )

        self.add_parameter(
            ParameterInt(
                name="seed",
                default_value=42,
                tooltip="Random seed",
                allow_output=False
            )
        )

        with ParameterGroup(name="LoRA Config", ui_options={"collapsed": True}) as lora_group:
            ParameterString(
                name="lora_path",
                default_value="",
                tooltip="Optional LoRA weights path",
                allow_output=False
            )
            ParameterFloat(
                name="lora_scale",
                default_value=0.8,
                tooltip="LoRA scale",
                allow_output=False,
                min_val=0.0,
                max_val=2.0
            )
        self.add_node_element(lora_group)

        self.add_parameter(
            ParameterImage(
                name="output",
                tooltip="Generated image",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="local_generated.png"
        )
        self._output_file.add_parameter()

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output"] = None

        import requests

        model_id = self.get_parameter_value("model_id_or_path") or MODEL_OPTIONS[0]
        prompt = self.get_parameter_value("prompt") or ""
        steps = self.get_parameter_value("num_inference_steps") or 8
        width = self.get_parameter_value("width") or 1024
        height = self.get_parameter_value("height") or 1024
        randomize = self.get_parameter_value("randomize_seed") or False
        seed = self.get_parameter_value("seed") or 42
        lora_path = self.get_parameter_value("lora_path") or ""
        lora_scale = self.get_parameter_value("lora_scale") or 0.8

        if randomize:
            seed = random.randint(0, 2**32 - 1)
            self.set_parameter_value("seed", seed)

        if not prompt:
            raise ValueError("Prompt is required.")

        try:
            health = requests.get(f"{SERVER_URL}/health", timeout=3)
            logger.info(f"Server status: {health.json()}")
        except Exception:
            raise RuntimeError(
                "Model server is not running!\n"
                "Please start the model server first.\n"
                "See README.md for instructions."
            )

        payload = {
            "model_id": model_id,
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": 0.0,
            "seed": seed,
            "lora_path": lora_path,
            "lora_scale": lora_scale,
        }

        logger.info(f"Sending generate request: {model_id}, {width}x{height}, steps={steps}")
        try:
            resp = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=600)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Server request failed: {e}")

        result = resp.json()
        seed_used = result["seed_used"]
        self.set_parameter_value("seed", seed_used)

        img_bytes = base64.b64decode(result["image_b64"])
        dest = self._output_file.build_file()
        saved = dest.write_bytes(img_bytes)

        url_artifact = ImageUrlArtifact(saved.location)
        self.set_parameter_value("output", url_artifact)
        self.publish_update_to_parameter("output", url_artifact)
        logger.info(f"Generation success: {saved.name} (seed={seed_used})")
