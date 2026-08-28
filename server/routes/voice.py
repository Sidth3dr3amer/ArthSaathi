"""
Voice service mount.

`TestVoice/backend.py` already implements the working ASR -> LLM -> TTS pipeline.
It is mounted rather than rebuilt.

**The mount is lazy, deliberately.** That module loads `WhisperModel("small")` at
*import* time, which downloads and holds roughly 500 MB. Importing it eagerly
would mean every API start, and every test collection, paid that cost -- and CI
would need the model. So it is imported on first use, and its absence degrades to
a 503 explaining what to install rather than preventing the whole API from
booting.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ml.src.common import config

router = APIRouter(prefix="/voice", tags=["voice"])

VOICE_MODULE_PATH = config.PROJECT_ROOT / "TestVoice" / "backend.py"

_voice_app: Any = None
_load_error: str | None = None


def load_voice_app() -> Any:
    """
    Import `TestVoice/backend.py` on first use.

    Loaded by path rather than as a package because `TestVoice` is not one, and
    making it one would mean editing a working service this task should not touch.
    """
    global _voice_app, _load_error
    if _voice_app is not None or _load_error is not None:
        return _voice_app

    if not VOICE_MODULE_PATH.exists():
        _load_error = f"voice service not found at {VOICE_MODULE_PATH}"
        return None

    try:
        spec = importlib.util.spec_from_file_location("testvoice_backend", VOICE_MODULE_PATH)
        module = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
        sys.modules["testvoice_backend"] = module
        spec.loader.exec_module(module)                          # type: ignore[union-attr]
        _voice_app = module.app
    except Exception as exc:
        _load_error = repr(exc)
        return None

    return _voice_app


@router.get("/status")
def voice_status() -> dict[str, Any]:
    """Whether the voice service can be loaded, without loading it."""
    return {
        "module_present": VOICE_MODULE_PATH.exists(),
        "path": str(VOICE_MODULE_PATH),
        "loaded": _voice_app is not None,
        "load_error": _load_error,
        "note": (
            "The voice service loads a ~500 MB Whisper model on first request, "
            "so it is imported lazily rather than at API start."
        ),
        "requires": ["faster-whisper", "edge-tts", "GROQ_API_KEY"],
        "groq_configured": bool(config.GROQ_API_KEY),
    }


@router.post("/warmup")
def warmup() -> dict[str, Any]:
    """
    Force the Whisper load now rather than on a user's first request.

    Worth calling before a demo: the first transcription would otherwise take the
    model download on the critical path.
    """
    app = load_voice_app()
    if app is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"voice service unavailable: {_load_error}. "
                "Install faster-whisper and edge-tts, and set GROQ_API_KEY."
            ),
        )
    return {"loaded": True, "routes": [r.path for r in app.routes if hasattr(r, "path")]}
