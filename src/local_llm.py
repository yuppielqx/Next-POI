"""Unified LLM inference wrapper for OpenAI models and local fallbacks."""
import os
import threading
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from transformers import BitsAndBytesConfig, pipeline
except Exception:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None
    pipeline = None

from src.utils import logger

_pipelines: dict = {}
_openai_client = None
_lock = threading.Lock()  # prevent multiple threads from loading the same model simultaneously
_PROJECT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a project-local .env file without overriding shell env."""
    if not path.exists():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            os.environ.setdefault(key, _strip_env_value(value))
    except Exception as e:
        logger.warning(f"Failed to load environment file {path}: {e}")


_load_dotenv(_PROJECT_ENV_PATH)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style content arrays (list of {type, text}) to plain strings."""
    normalized = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(
                part["text"] for part in content if part.get("type") == "text"
            )
        normalized.append({"role": m["role"], "content": content})
    return normalized


def _get_pipeline(model_name: str):
    if torch is None or pipeline is None:
        raise RuntimeError(
            "Local model dependencies are missing. Install torch and transformers, "
            "or use a gpt-* model through the OpenAI API."
        )

    if model_name in _pipelines:
        return _pipelines[model_name]

    with _lock:
        # Double-checked locking: another thread may have loaded it while we waited
        if model_name in _pipelines:
            return _pipelines[model_name]

        pipe = None
        device = 0 if torch.cuda.is_available() else -1

        # Attempt 1: 4-bit quantization with device_map (needs accelerate + bitsandbytes)
        try:
            logger.info(f"Loading model {model_name} with 4-bit quantization…")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            pipe = pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"quantization_config": bnb_config},
                device_map="auto",
            )
        except Exception as e:
            logger.warning(f"4-bit + device_map failed ({e}); trying bf16 with device_map…")

        # Attempt 2: bf16 with device_map (needs accelerate)
        if pipe is None:
            try:
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
            except Exception as e:
                logger.warning(f"bf16 + device_map failed ({e}); trying bf16 on cuda:{device}…")

        # Attempt 3: bf16 on explicit device (no accelerate needed)
        if pipe is None:
            pipe = pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=device,
            )

        _pipelines[model_name] = pipe
        logger.info(f"Model {model_name} loaded.")

    return _pipelines[model_name]


def _is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-")


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set. Add it to {_PROJECT_ENV_PATH} or export it in your shell."
        )

    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    _openai_client = OpenAI(**client_kwargs)
    return _openai_client


def chat_completion(
    model: str,
    messages: list[dict],
    max_new_tokens: int = 512,
) -> str | None:
    """
    Run local inference. Returns assistant response text, or None on failure.
    Messages follow the OpenAI format: [{"role": ..., "content": ...}].
    Content may be a string or an OpenAI-style content array.
    """
    try:
        normalized_messages = _normalize_messages(messages)
        if _is_openai_model(model):
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=normalized_messages,
                max_tokens=max_new_tokens,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text:
                return text.strip()
            return None

        pipe = _get_pipeline(model)
        result = pipe(
            normalized_messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        return result[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        logger.warning(f"local_llm ({model}) error: {e}")
        return None
