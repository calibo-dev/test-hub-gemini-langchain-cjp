from __future__ import annotations

import logging
from typing import Any

from latest_ai_development.config.settings import (
    get_stages_config,
    get_workflow_config,
)
from latest_ai_development.stages.stage_registry import STAGE_REGISTRY

logger = logging.getLogger(__name__)

_RESERVED_STAGE_CONFIG_KEYS = {"tracing"}


def validate_configuration() -> None:
    """
    Validate overall configuration integrity.

    Checks:
    - workflow.yaml stage references
    - stages.yaml prompt definitions
    - stages.yaml model.primary/model.fallback configuration
    - optional tracing metadata alignment
    """

    validate_stage_configuration()
    validate_model_configuration()


def validate_stage_configuration() -> None:
    """
    Validate consistency between:
      - workflow.yaml
      - stages.yaml
      - stage_registry
    """

    workflow_cfg = get_workflow_config()
    stages_cfg = get_stages_config()

    workflow_stages = set(workflow_cfg.get("stages", []))
    registry_stages: set[str] = set(STAGE_REGISTRY.keys())
    config_stages: set[str] = {
        stage_name
        for stage_name in stages_cfg.keys()
        if stage_name not in _RESERVED_STAGE_CONFIG_KEYS
    }

    # 1. workflow.yaml must reference registered stages
    unknown_registry = workflow_stages - registry_stages
    if unknown_registry:
        raise ValueError(
            "workflow.yaml references unknown stages not in STAGE_REGISTRY: "
            f"{sorted(unknown_registry)}"
        )

    # 2. workflow.yaml must have stage config
    missing_stage_config = workflow_stages - config_stages
    if missing_stage_config:
        raise ValueError(
            f"Missing stage configuration in stages.yaml for: {sorted(missing_stage_config)}"
        )

    # 3. warn if unused configs exist
    unused_configs = config_stages - workflow_stages
    if unused_configs:
        logger.warning(
            "[Warning] stages.yaml contains configs not used in workflow.yaml: "
            f"{sorted(unused_configs)}"
        )

    validate_tracing_configuration(workflow_stages, stages_cfg.get("tracing", {}))


def validate_tracing_configuration(
    workflow_stages: set[str],
    tracing_cfg: object,
) -> None:
    """
    Validate optional tracing metadata.

    Missing tracing entries only warn because generated stages can fall back to
    dynamic default trace names.
    """

    if not isinstance(tracing_cfg, dict):
        return

    traced_stage_names: set[str] = set()

    direct_stage_cfgs = tracing_cfg.get("stages", {})
    if isinstance(direct_stage_cfgs, dict):
        traced_stage_names.update(
            stage_name
            for stage_name, stage_cfg in direct_stage_cfgs.items()
            if isinstance(stage_cfg, dict)
        )

    traced_stage_names.update(
        stage_name
        for stage_name, stage_cfg in tracing_cfg.items()
        if stage_name not in {"workflow", "stages"} and isinstance(stage_cfg, dict)
    )

    if not traced_stage_names:
        return

    missing_trace_config = workflow_stages - traced_stage_names
    if missing_trace_config:
        logger.warning(
            "[Warning] tracing config is missing entries for workflow stages: "
            f"{sorted(missing_trace_config)}"
        )

    unused_trace_config = traced_stage_names - workflow_stages
    if unused_trace_config:
        logger.warning(
            "[Warning] tracing config contains entries not used in workflow.yaml: "
            f"{sorted(unused_trace_config)}"
        )


def validate_model_configuration() -> None:
    """Validate stage-owned model configuration."""

    workflow_cfg = get_workflow_config()
    stages_cfg = get_stages_config()

    for stage_name in workflow_cfg.get("stages", []) or []:
        stage_cfg = stages_cfg.get(stage_name)
        if not isinstance(stage_cfg, dict):
            continue

        model_cfg = stage_cfg.get("model")
        if not isinstance(model_cfg, dict):
            raise ValueError(f"stages.yaml stage '{stage_name}' must define model.primary")

        _validate_model_section(stage_name, model_cfg, "primary", required=True)
        _validate_model_section(stage_name, model_cfg, "fallback", required=False)


def _validate_model_section(
    stage_name: str,
    model_cfg: dict[str, Any],
    section: str,
    *,
    required: bool,
) -> None:
    section_cfg = model_cfg.get(section)
    if section_cfg is None:
        if required:
            raise ValueError(f"stages.yaml stage '{stage_name}' must define model.{section}")
        return

    if not isinstance(section_cfg, dict):
        raise ValueError(f"stages.yaml stage '{stage_name}' model.{section} must be a mapping")

    provider = section_cfg.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(
            f"stages.yaml stage '{stage_name}' model.{section}.provider is required"
        )

    model_id = section_cfg.get("modelId")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(
            f"stages.yaml stage '{stage_name}' model.{section}.modelId is required"
        )

    generation_defaults = section_cfg.get("generationDefaults")
    if generation_defaults is not None and not isinstance(generation_defaults, dict):
        raise ValueError(
            f"stages.yaml stage '{stage_name}' model.{section}.generationDefaults "
            "must be a mapping"
        )
