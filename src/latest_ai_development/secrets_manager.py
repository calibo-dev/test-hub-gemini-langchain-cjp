from __future__ import annotations

import base64
import json
import os
from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def get_secret(secret_name: str, region_name: str | None = None) -> str:
    if not secret_name:
        return ""

    resolved_region = (
        region_name
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

    client = boto3.client("secretsmanager", region_name=resolved_region)
    response = client.get_secret_value(SecretId=secret_name)

    if "SecretString" in response and response["SecretString"]:
        return response["SecretString"]

    if "SecretBinary" in response and response["SecretBinary"]:
        return base64.b64decode(response["SecretBinary"]).decode("utf-8")

    return ""


def _extract_value_from_secret(secret_value: str, preferred_key: str) -> str:
    if not secret_value:
        return ""

    try:
        parsed = json.loads(secret_value)
    except json.JSONDecodeError:
        return secret_value.strip()

    if isinstance(parsed, dict):
        if preferred_key in parsed:
            return str(parsed[preferred_key]).strip()

        for key in ("OPENAI_API_KEY", "api_key", "key", "value"):
            if key in parsed:
                return str(parsed[key]).strip()

    return secret_value.strip()


@lru_cache(maxsize=8)
def resolve_secret_or_env(
    env_var_name: str,
    secret_name_env_var: str | None = None,
    preferred_secret_key: str | None = None,
) -> str:
    direct_value = os.getenv(env_var_name, "").strip()
    if direct_value:
        return direct_value

    if not secret_name_env_var:
        return ""

    secret_name = os.getenv(secret_name_env_var, "").strip()
    if not secret_name:
        return ""

    try:
        secret_value = get_secret(secret_name)
    except (BotoCoreError, ClientError):
        return ""

    return _extract_value_from_secret(secret_value, preferred_secret_key or env_var_name)


def resolve_openai_api_key() -> str:
    return resolve_secret_or_env(
        env_var_name="OPENAI_API_KEY",
        secret_name_env_var="OPENAI_API_KEY_SECRET",
        preferred_secret_key="OPENAI_API_KEY",
    )