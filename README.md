# NOTE

- This document is provided as a **starter template only**.
- Customers must review, update, and validate this content to ensure it meets their **functional, security, compliance, and operational requirements** before deployment.

# LangChain Country Reference Workflow

A production-ready LangChain FastAPI workflow for **safe, metadata-driven SQL queries** against the `ak_country` Snowflake table. The system:

1. Retrieves the latest schema metadata using `CountryAgenticTool`.
2. Translates the user's natural language request into a candidate SQL query.
3. Validates the query for read-only use and schema compliance.
4. Executes the query via `CountryAgenticTool` when safe.
5. Synthesizes a final `response` that is returned through the `/ask` API without persisting any workflow artifacts.

![Python](https://img.shields.io/badge/Python-black?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-black?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-black)
![Snowflake](https://img.shields.io/badge/Snowflake-black?logo=snowflake)
![Docker](https://img.shields.io/badge/Docker-black?logo=docker)

## 1. Project Overview

This repository contains a LangChain orchestration template that exposes the workflow via a FastAPI `/ask` endpoint. Each request runs through five sequential stages: metadata retrieval, SQL generation, validation, safe execution, and response generation. No workflow output file is written to disk because `execution.save_output` is set to `false`—the canonical output is the `response` field returned by `/ask`.

### Workflow Topology

- `metadata_fetching_agent`: Discovers column names, types, and schema notes from `ak_country` using `CountryAgenticTool`.
- `sql_generator_agent`: Builds candidate SQL based on the user query and retrieved metadata, flagging out-of-scope requests.
- `validator_agent`: Ensures the SQL is read-only, uses valid columns, and propagates an `out_of_scope` flag when needed.
- `query_execution_agent`: Executes validated SQL (when in scope) via `CountryAgenticTool` and captures raw results.
- `response_generation_agent`: Crafts the final user-facing answer from the query results or out-of-scope signal.

## 2. Tech Stack

| Technology / Framework | Detected Version |
| --- | --- |
| Python | `>=3.11,<3.15` |
| LangChain | `>=0.3.20` |
| langchain-openai | `>=0.3.8` |
| langchain-ollama | `>=0.2.3` |
| langchain-aws | `>=1.0` |
| FastAPI | `>=0.115.8` |
| Uvicorn | `>=0.34.0` |
| pandas | `>=2.0.0` |
| snowflake-connector-python | `>=3.10.0` |
| boto3 | `>=1.37.0` |
| azure-identity | `>=1.25.1` |
| azure-keyvault-secrets | `>=4.10.0` |
| PyYAML | `>=6.0.2` |
| pydantic | `>=2.10.6` |
| pydantic-settings | `>=2.7.1` |

## 3. Project Structure Explanation

```bash
├── src/
│   └── latest_ai_development/
│       ├── config/
│       │   ├── workflow.yaml          # Stage order and execution configuration
│       │   ├── stages.yaml            # Stage prompts, instructions, tool flags, and model config
│       │   ├── validators.py          # Startup configuration validation
│       │   └── settings.py            # Runtime environment settings
│       ├── agents/                    # Tool-calling agent builders (metadata/query execution)
│       ├── chains/                    # LCEL chains for SQL generation, validation, and response generation
│       ├── prompts/                   # Prompt builders for each stage
│       ├── stages/                    # Stage classes and registry mapping canonical stage keys
│       ├── tools/                     # Tool registry and integrations (CountryAgenticTool, custom tools)
│       ├── workflow.py                # Orchestration controller running sequential stages with LangSmith tracing
│       └── main.py                    # FastAPI entrypoint exposing /health and /ask
├── knowledge/                          # Optional knowledge files referenced via the request
├── .env.example                        # Environment variable guidance
├── pyproject.toml                     # Dependencies (includes Snowflake connector and pandas)
├── uv.lock                            # Locked dependency graph
├── AGENTS.md                          # Workflow and agent documentation
└── README.md                          # Project overview, structure, usage, and tracing guidance
```

## 4. Key Files and Configuration

- **`src/latest_ai_development/main.py`**  
  Initializes FastAPI, loads settings via `get_settings()`, validates configuration, and exposes `/ask` and `/health`. `/ask` accepts `user_query` and optional `knowledge_context`, runs the workflow, and returns the final `response`.

- **`src/latest_ai_development/workflow.py`**  
  Implements `LatestAiDevelopmentWorkflow` with sequential stages (`metadata_fetching_agent` → `sql_generator_agent` → `validator_agent` → `query_execution_agent` → `response_generation_agent`). Tracing is wired through `trace_workflow` and `trace_stage_execution` hooks from `latest_ai_development.tracing`.

- **`src/latest_ai_development/config/workflow.yaml`**  
  Defines the new stage order and ensures `execution.save_output` stays `false` so no artifacts are persisted.

- **`src/latest_ai_development/config/stages.yaml`**  
  Stores per-stage system prompts, instructions, tool usage flags, temperature overrides, and model configuration for every stage in the contract.

- **`src/latest_ai_development/stages/`**  
  Contains the canonical stage classes (one per design agent) that validate inputs, invoke agents/chains, and return consistent output keys.

- **`src/latest_ai_development/agents/`** and **`src/latest_ai_development/chains/`**  
  Provide builder functions for tool-calling agents (metadata and query execution) plus JSON/text chain runners for SQL generation, validation, and response synthesis.

- **`src/latest_ai_development/tools/`**  
  Hosts `CountryAgenticTool`, `custom_tool`, and `tool_registry.py` which exposes stage-scoped tools while respecting supported runtime tool wiring.

## 5. Setup & Installation

### Prerequisites

- Python in the supported range: `>=3.11,<3.15`
- `uv` for dependency management
- Snowflake credentials stored via `CountryAgenticTool` environment variables (see `.env.example`)
- Stage model configuration defined in `src/latest_ai_development/config/stages.yaml`
- Optional contextual knowledge USD via the `knowledge_context` field

### Secret Management & Environment Variables

- `OPENAI_API_KEY_SECRET`
- `ANTHROPICAI_API_KEY_SECRET`
- `GEMINIAI_API_KEY_SECRET`
- `BEDROCK_AI_API_KEY_SECRET`
- `LANGCHAIN_API_KEY_SECRET`
- `LANGSMITH_API_KEY_SECRET`
- `TRACING_BACKEND=LANGSMITH` to enable LangSmith tracing
- `LANGSMITH_PROJECT` for grouping traces
- `LANGSMITH_ENDPOINT` (optional override)
- `CLOUD_PROVIDER` (AWS or AZURE)
- `AWS_REGION` when using AWS secrets
- `AZURE_KEY_VAULT_URL` when using Azure
- `COUNTRYAGENTICTOOL_PAT_SECRET`, `COUNTRYAGENTICTOOL_ACCOUNT`, `COUNTRYAGENTICTOOL_USER`, `COUNTRYAGENTICTOOL_WAREHOUSE`, `COUNTRYAGENTICTOOL_DATABASE`, `COUNTRYAGENTICTOOL_SCHEMA`, `COUNTRYAGENTICTOOL_HOST`

<!-- BEGIN GENERATED TOOL: rdbms-snowflake:COUNTRYAGENTICTOOL -->
## Snowflake RDBMS Tool Configuration: `CountryAgenticTool`

This generated tool uses tool-specific Snowflake configuration with the prefix `COUNTRYAGENTICTOOL`.

The generated tool applies non-secret values from `supported_tools[].configuration` directly
as generated defaults. Runtime environment variables with the prefix `COUNTRYAGENTICTOOL` override
those non-secret defaults when provided. The PAT secret reference is always supplied only by
`COUNTRYAGENTICTOOL_PAT_SECRET`.

Generated default keys present: `account, database, host, schema, user, warehouse`.

Generated SQL execution is unrestricted by this tool. SQL is sent to Snowflake as provided,
and Snowflake permissions determine what succeeds.

```env
COUNTRYAGENTICTOOL_ACCOUNT=
COUNTRYAGENTICTOOL_USER=
COUNTRYAGENTICTOOL_PAT_SECRET=
COUNTRYAGENTICTOOL_WAREHOUSE=
COUNTRYAGENTICTOOL_DATABASE=
COUNTRYAGENTICTOOL_SCHEMA=
COUNTRYAGENTICTOOL_HOST=
```

`COUNTRYAGENTICTOOL_PAT_SECRET` is required at runtime and must contain the SecretsManager secret
name for this tool's Snowflake PAT. This value is intentionally not read from
`supported_tools[].configuration`. Do not store PAT secret names, PAT values, passwords,
tokens, private keys, or connection strings in source code or payloads.

The generated tool can discover authorized Snowflake metadata at runtime when called without
a query. Use metadata scopes such as `databases`, `schemas`, `tables`, and `columns` to
inspect allowed objects, then call the same tool with the required SQL query.

Static table schema is not required in the supported tool payload. The generated tool relies
on Snowflake metadata visible to the configured user.
<!-- END GENERATED TOOL: rdbms-snowflake:COUNTRYAGENTICTOOL -->

### Install Dependencies

```bash
uv sync
```

### Run the Workflow

```python
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow
from datetime import datetime

inputs = {
    "user_query": "List countries in the Caribbean region with their ISO alpha-3 codes.",
    "current_year": int(datetime.now().year),
    "knowledge_context": "",
}

LatestAiDevelopmentWorkflow().kickoff(inputs=inputs)
```

### Run via FastAPI

```bash
POST /ask
{"user_query":"Describe regions covered in the ak_country table","knowledge_context":""}
```

### Stage Model Fallback

Each stage defines `model.primary` and an optional `model.fallback` in `src/latest_ai_development/config/stages.yaml`. The runtime wraps fallback models when available using LangChain middleware, mirroring the template's previous behavior.

## 6. Development Guidelines

- Keep orchestration logic in `workflow.py`. The workflow executes stages sequentially, maintains context, and never mixes stage roles with canonical stage keys.
- Update prompts in `src/latest_ai_development/prompts/prompt_builder.py` when you need to adjust stage instructions or include new structured sections.
- `execution.save_output` must remain `false` so the workflow does not write reports or artifacts to disk—the API response is the source of truth.
- Add or update canonical stage files under `src/latest_ai_development/stages/` whenever the workflow shape changes.

## 7. Security & Networking (HTTP / HTTPS)

- HTTP support is available by default; secure deployments should configure TLS/HTTPS.
- LangSmith tracing is enabled by setting `TRACING_BACKEND=LANGSMITH` and providing `LANGSMITH_API_KEY_SECRET`. If tracing is disabled, leave `TRACING_BACKEND` unset or set it to `NONE`.
- The workflow also honors `LANGCHAIN_API_KEY_SECRET` for providers that integrate with LangSmith.
- Keep all PAT secrets and credential names out of source code and rely on the secrets manager hooks in `latest_ai_development.tools.snowflake_connection`.
