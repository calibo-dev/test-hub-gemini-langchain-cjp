from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

from dotenv import load_dotenv

from latest_ai_development.config.settings import get_stages_config

load_dotenv()

if (
    not os.getenv("LANGSMITH_API_KEY", "").strip()
    and not os.getenv("LANGCHAIN_API_KEY", "").strip()
):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - langchain normally installs langsmith
    traceable = None

logger = logging.getLogger(__name__)

_LANGSMITH_INITIALIZED = False


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def get_tracing_backend() -> str:
    """Return the normalized tracing backend name."""
    configured_backend = os.getenv("TRACING_BACKEND")
    if configured_backend is None and _truthy(os.getenv("LANGSMITH_TRACING")):
        value = "LANGSMITH"
    else:
        value = (configured_backend or "NONE").strip().upper() or "NONE"

    if value in {"OFF", "DISABLED"}:
        value = "NONE"

    logger.info("Tracing backend: %s", value)
    return value


def _first_env_value(*env_names: str) -> str | None:
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def _resolve_langsmith_api_key() -> str | None:
    secret_name = _first_env_value("LANGSMITH_API_KEY_SECRET", "LANGCHAIN_API_KEY_SECRET")
    if not secret_name:
        return None

    try:
        from latest_ai_development.secrets_manager import SecretsManager
    except ModuleNotFoundError as exc:
        logger.warning(
            "LangSmith API key resolution skipped because secrets manager is unavailable "
            "| error=%s",
            exc,
        )
        return None

    secret_value = SecretsManager().get_secret(secret_name)
    return secret_value.strip() if secret_value else None


def initialize_langsmith_tracing() -> None:
    """Enable LangSmith tracing for LangChain when TRACING_BACKEND=LANGSMITH."""
    global _LANGSMITH_INITIALIZED

    tracing_backend = get_tracing_backend()
    if tracing_backend != "LANGSMITH":
        if tracing_backend == "NONE":
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    if _LANGSMITH_INITIALIZED:
        return

    try:
        api_key = _resolve_langsmith_api_key()
        if not api_key:
            raise ValueError(
                "TRACING_BACKEND is LANGSMITH but LANGSMITH_API_KEY_SECRET is not "
                "configured."
            )

        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        logger.info(
            "LangSmith tracing enabled | project=%s",
            os.getenv("LANGSMITH_PROJECT") or "not set",
        )
        _LANGSMITH_INITIALIZED = True
    except Exception as exc:
        os.environ["TRACING_BACKEND"] = "NONE"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning(
            "LangSmith tracing initialization failed; falling back to no tracing | error=%s",
            exc,
        )


def trace_workflow() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a lazy LangSmith decorator for the workflow root run."""
    return _lazy_traceable(lambda: _workflow_trace_spec())


def trace_stage(
    stage_name: str,
    *,
    component_name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a lazy LangSmith decorator for a workflow stage."""
    return _lazy_traceable(lambda: _stage_trace_spec(stage_name, component_name))


def trace_stage_execution(
    stage_name: str,
    component_name: str | None,
    invoke: Callable[[dict[str, Any]], Any],
    inputs: dict[str, Any],
) -> Any:
    """Execute a stage under a dynamic trace span."""
    traced_invoke = trace_stage(stage_name, component_name=component_name)(invoke)
    return traced_invoke(inputs)


def _lazy_traceable(
    spec_factory: Callable[[], dict[str, Any]],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            traced_func = _traceable_from_spec(spec_factory())(func)
            return traced_func(*args, **kwargs)

        return wrapper

    return decorator


def _traceable_from_spec(
    spec: dict[str, Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if traceable is None:
        return lambda func: func

    if get_tracing_backend() != "LANGSMITH":
        return lambda func: func

    if not spec.get("enabled", True):
        return lambda func: func

    return traceable(
        name=spec.get("name") or "Trace",
        run_type=spec.get("run_type", "chain"),
        tags=list(spec.get("tags", [])),
        metadata=dict(spec.get("metadata", {})),
        process_inputs=_build_field_processor(spec.get("input_fields")),
        process_outputs=_build_field_processor(spec.get("output_fields")),
    )


def _workflow_trace_spec() -> dict[str, Any]:
    config_spec = _get_named_tracing_spec("workflow")
    return _normalize_trace_spec(
        _default_workflow_trace_spec(),
        config_spec,
        fallback_name="Research Reporting Workflow",
    )


def _stage_trace_spec(stage_name: str, component_name: str | None) -> dict[str, Any]:
    config_spec = _get_named_tracing_spec(stage_name)
    default_spec = _default_stage_trace_spec(stage_name, component_name)
    return _normalize_trace_spec(
        default_spec,
        config_spec,
        fallback_name=default_spec["name"],
    )


def _default_workflow_trace_spec() -> dict[str, Any]:
    return {
        "enabled": True,
        "name": "Research Reporting Workflow",
        "run_type": "chain",
        "tags": ["workflow", "multi-agent"],
        "metadata": {
            "workflow_name": "research_reporting",
            "workflow_version": "1.0",
        },
        "input_fields": {
            "flow_run_id": {"path": "flow_run_id"},
            "topic": {"path": "inputs.topic"},
            "current_year": {"path": "inputs.current_year"},
            "knowledge_context_present": {
                "path": "inputs.knowledge_context",
                "transform": "bool",
            },
        },
        "output_fields": {
            "flow_run_id": {"path": "flow_run_id"},
            "topic": {"path": "topic"},
            "keys": {"path": ".", "transform": "keys"},
            "has_report": {"path": "report", "transform": "bool"},
            "report_chars": {"path": "report", "transform": "len"},
            "research_notes_chars": {"path": "research_notes", "transform": "len"},
        },
    }


def _default_stage_trace_spec(stage_name: str, component_name: str | None) -> dict[str, Any]:
    readable_stage_name = _default_stage_name(stage_name)
    metadata = {
        "component_type": "stage",
        "stage": stage_name,
    }
    if component_name:
        metadata["component_name"] = component_name

    return {
        "enabled": True,
        "name": readable_stage_name,
        "run_type": "chain",
        "tags": ["stage", stage_name],
        "metadata": metadata,
        "input_fields": {
            "flow_run_id": {"path": "inputs.flow_run_id"},
            "topic": {"path": "inputs.topic"},
            "current_year": {"path": "inputs.current_year"},
            "input_keys": {"path": "inputs", "transform": "keys"},
        },
        "output_fields": {
            "topic": {"path": "topic"},
            "output_keys": {"path": ".", "transform": "keys"},
            "report_chars": {"path": "report", "transform": "len"},
            "research_notes_chars": {"path": "research_notes", "transform": "len"},
        },
    }


def _get_tracing_config() -> dict[str, Any]:
    stages_cfg = get_stages_config()
    tracing_cfg = stages_cfg.get("tracing", {})
    return tracing_cfg if isinstance(tracing_cfg, dict) else {}


def _get_named_tracing_spec(section_name: str) -> dict[str, Any]:
    tracing_cfg = _get_tracing_config()

    direct_section = tracing_cfg.get(section_name)
    if isinstance(direct_section, dict):
        return direct_section

    sections = tracing_cfg.get("stages", {})
    if isinstance(sections, dict):
        section = sections.get(section_name, {})
        return section if isinstance(section, dict) else {}

    return {}


def _default_stage_name(stage_name: str) -> str:
    return f"{stage_name.replace('_', ' ').title()} Stage"


def _normalize_trace_spec(
    base_spec: dict[str, Any],
    override_spec: dict[str, Any],
    *,
    fallback_name: str,
) -> dict[str, Any]:
    normalized = dict(base_spec)
    normalized.setdefault("name", fallback_name)

    if "name" in override_spec:
        normalized["name"] = override_spec["name"]

    if "enabled" in override_spec:
        normalized["enabled"] = override_spec["enabled"]

    if "run_type" in override_spec:
        normalized["run_type"] = override_spec["run_type"]

    if "tags" in override_spec:
        normalized["tags"] = list(override_spec["tags"])

    if "metadata" in override_spec:
        base_metadata = normalized.get("metadata", {})
        override_metadata = override_spec["metadata"]
        if isinstance(base_metadata, Mapping) and isinstance(override_metadata, Mapping):
            normalized["metadata"] = _merge_dicts(base_metadata, override_metadata)
        else:
            normalized["metadata"] = override_metadata

    if "input_fields" in override_spec:
        normalized["input_fields"] = override_spec["input_fields"]

    if "output_fields" in override_spec:
        normalized["output_fields"] = override_spec["output_fields"]

    for key, value in override_spec.items():
        if key in {
            "name",
            "enabled",
            "run_type",
            "tags",
            "metadata",
            "input_fields",
            "output_fields",
        }:
            continue
        normalized[key] = value

    return normalized


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(payload: Any, path: str | None) -> Any:
    if path is None or path == "":
        return payload

    current: Any = payload
    for segment in path.split("."):
        if segment in {"", "."}:
            continue

        if isinstance(current, Mapping):
            if segment not in current:
                return None
            current = current[segment]
            continue

        if hasattr(current, segment):
            current = getattr(current, segment)
            continue

        return None

    return current


def _apply_transform(value: Any, transform: str | None) -> Any:
    if transform in (None, "", "identity"):
        return value

    if transform in {"bool", "exists"}:
        return bool(value)

    if transform == "len":
        if value is None:
            return 0
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            return len(str(value))

    if transform == "str":
        return "" if value is None else str(value)

    if transform == "type":
        return type(value).__name__

    if transform == "keys":
        if isinstance(value, Mapping):
            return sorted(value.keys())
        keys = getattr(value, "keys", None)
        if callable(keys):
            return sorted(list(keys()))
        return []

    raise ValueError(f"Unsupported trace transform: {transform}")


def _normalize_field_spec(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"path": value}

    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(f"Trace field spec must be a string or mapping, got {type(value).__name__}")


def _build_field_processor(field_specs: Any) -> Callable[[Any], dict[str, Any]]:
    specs = field_specs if isinstance(field_specs, Mapping) else {}
    normalized_specs = {
        name: _normalize_field_spec(spec)
        for name, spec in specs.items()
    }

    def processor(payload: Any) -> dict[str, Any]:
        if not normalized_specs:
            if isinstance(payload, Mapping):
                return {"keys": sorted(payload.keys())}
            return {"output_type": type(payload).__name__}

        processed: dict[str, Any] = {}
        for field_name, spec in normalized_specs.items():
            raw_value = _resolve_path(payload, spec.get("path"))
            default_value = spec.get("default")
            value = raw_value if raw_value is not None else default_value
            processed[field_name] = _apply_transform(value, spec.get("transform"))
        return processed

    return processor


__all__ = [
    "get_tracing_backend",
    "initialize_langsmith_tracing",
    "trace_stage",
    "trace_stage_execution",
    "trace_workflow",
]
