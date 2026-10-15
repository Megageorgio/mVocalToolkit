"""Validate a local speech preview before queueing an engine job."""
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, Field


class SpeechPreviewRequest(BaseModel):
    # Output paths and unknown worker kwargs are never forwarded by this API.
    model_config = {"extra": "ignore", "allow_inf_nan": False}
    kind: Literal["qwen_tts_voice", "voxcpm_voice"]
    model_dir: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4000)
    speaker: str = ""
    language: str = "Auto"
    reference: str = ""
    base_model: str = ""
    device: str = "auto"
    seed: int = Field(42, strict=True, ge=-(2**63), le=2**64-1)
    steps: int = Field(10, strict=True, ge=1, le=1000)
    max_tokens: int = Field(1024, strict=True, ge=1, le=8192)
    temperature: float = Field(.9, gt=0, le=2)
    guidance: float = Field(2., gt=0, le=10)


def checked_speech_preview(payload: dict) -> dict:
    request = SpeechPreviewRequest.model_validate(payload)
    if not request.text.strip(): raise ValueError("Enter text to speak")
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?|mps", request.device):
        raise ValueError("Choose auto, cpu, cuda:N or mps")
    root = Path(request.model_dir).expanduser().resolve()
    if not request.model_dir.strip() or not root.is_dir(): raise ValueError("Select a local exported speech model folder")
    request.model_dir = str(root)
    if request.kind == "voxcpm_voice":
        if request.reference:
            path = Path(request.reference).expanduser().resolve()
            if not path.is_file(): raise ValueError("Reference audio does not exist")
            request.reference = str(path)
        lora = root / "lora_config.json"
        if lora.is_file():
            config = json.loads(lora.read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict): raise ValueError("LoRA configuration must be an object")
            if not isinstance(config.get("lora_config"), dict): raise ValueError("LoRA configuration must contain lora_config settings")
            base = request.base_model or config.get("base_model", "")
            if not isinstance(base, str) or not base.strip() or not Path(base).expanduser().is_dir():
                raise ValueError("Choose the local VoxCPM base model used by this LoRA")
            request.base_model = str(Path(base).expanduser().resolve())
    return request.model_dump()
