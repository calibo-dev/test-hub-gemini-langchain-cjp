from __future__ import annotations

from typing import Set

from latest_ai_development.config.settings import (
    get_workflow_config,
    get_stages_config,
    get_models_config,
    get_settings,
)

from latest_ai_development.stages.stage_registry import STAGE_REGISTRY


def validate_configuration() -> None:
    """
    Validate overall configuration integrity.

    Checks:
    - workflow.yaml stage references
    - stages.yaml prompt definitions
    - models.yaml provider configuration
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
    registry_stages: Set[str] = set(STAGE_REGISTRY.keys())
    config_stages: Set[str] = set(stages_cfg.keys())

    # 1. workflow.yaml must reference registered stages
    unknown_registry = workflow_stages - registry_stages
    if unknown_registry:
        raise ValueError(
            f"workflow.yaml references unknown stages not in STAGE_REGISTRY: {sorted(unknown_registry)}"
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
        print(
            f"[Warning] stages.yaml contains configs not used in workflow.yaml: {sorted(unused_configs)}"
        )


def validate_model_configuration() -> None:
    """
    Validate models.yaml and environment provider configuration.
    """

    settings = get_settings()
    models_cfg = get_models_config()

    default_provider = models_cfg.get("default_provider")
    providers = models_cfg.get("providers", {})

    if not providers:
        raise ValueError("models.yaml must define at least one provider")

    if default_provider not in providers:
        raise ValueError(
            f"default_provider '{default_provider}' is not defined in models.yaml providers"
        )

    env_provider = settings.provider.lower()

    if env_provider not in providers:
        raise ValueError(
            f"Environment PROVIDER '{env_provider}' not defined in models.yaml providers"
        )

    provider_cfg = providers.get(env_provider)

    if not provider_cfg.get("chat_model"):
        raise ValueError(
            f"Provider '{env_provider}' must define 'chat_model' in models.yaml"
        )