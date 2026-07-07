# NOTE

- This document is provided as a **starter template only**.
- Customers must review, update, and validate this content to ensure it meets their **functional, security, compliance, and operational requirements** before deployment.

# LangChain Python Template

A production-oriented LangChain template for **two-agent sequential orchestration** with FastAPI integration, provider-based model selection, and markdown report generation.

![Python](https://img.shields.io/badge/Python-black?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-black?logo=fastapi)
![Uvicorn](https://img.shields.io/badge/Uvicorn-black?logo=uvicorn)
![LangChain](https://img.shields.io/badge/LangChain-black)
![Docker](https://img.shields.io/badge/Docker-black?logo=docker)
![Helm](https://img.shields.io/badge/Helm-black?logo=helm)

## 1. Project Overview

Based on the available code, this project is a **Python multi-agent “research → report” service** built with **LangChain** and exposed via a **FastAPI** API. It orchestrates a sequential workflow (research agent → reporting agent) to generate a markdown report file (`report.md`) from a topic input.

Primary stack:
- **Python** application using a **src/**-layout package (`latest_ai_development`)
- **LangChain** for agent orchestration, prompt construction, and model interaction
- **FastAPI + Uvicorn** for HTTP endpoints (`/health`, `/ask`)
- Optional integrations visible in code: **AWS Secrets Manager (boto3)**, **Azure Key Vault**, and LLM endpoints configured via environment variables.

### Template Origin

This project was initialized using **LangChain**.

```bash
langchain template new langgraph-agent-template
```

## 2. Tech Stack

### Detected Stack

| Technology / Framework | Detected Version |
| --- | --- |
| Python | `>=3.11,<3.13` |
| LangChain | `>=0.3.20` |
| langchain-openai | `>=0.3.8` |
| langchain-community | `>=0.3.19` |
| langchain-ollama | `>=0.2.3` |
| FastAPI | `>=0.115.8` |
| Uvicorn | `>=0.34.0` |
| boto3 | `>=1.37.0` |
| azure-identity | `>=1.25.1` |
| azure-keyvault-secrets | `>=4.10.0` |
| Pydantic | `>=2.10.6` |
| pydantic-settings | `>=2.7.1` |
| PyYAML | `>=6.0.2` |
| Hatchling (build backend) | Used |
| Docker | Project-specific runtime/build configuration |
| Helm Chart | `version: 0.1.0` (chart), `appVersion: "1.0"` |

- **Primary orchestration framework:** `LangChain`
- **Package configuration:** `pyproject.toml`
- **Lockfile(s) found:** `uv.lock`
- **Runtime version hint:** `requires-python = ">=3.11,<3.13"`

## 3. Project Structure Explanation

```bash
├── src/
│   └── latest_ai_development/
│       ├── config/
│       │   ├── agents.yaml            # Agent role configuration (roles/goals/backstories/instructions)
│       │   ├── tasks.yaml             # Task configuration (descriptions/expected outputs/agent mapping)
│       │   ├── models.yaml            # Provider-specific model defaults
│       │   └── settings.py            # Runtime settings loader for env/config paths
│       ├── tools/
│       │   ├── __init__.py            # Tool exports
│       │   └── custom_tool.py         # Example LangChain-compatible tool implementation
│       ├── __init__.py                # Package exports and version metadata
│       ├── workflow.py                # Main LangChain two-agent orchestration logic
│       ├── prompts.py                 # Prompt construction for both agents
│       ├── main.py                    # FastAPI application and CLI entrypoints
│       └── secrets_manager.py         # AWS Secrets Manager / Azure Key Vault integration
├── knowledge/
│   └── user_preference.txt            # Example user/context preference file
├── output/
│   └── report.md                      # Final generated markdown report
├── helm_chart/                        # Kubernetes deployment templates
│   ├── Chart.yaml                     # Helm chart metadata
│   ├── values.yaml                    # Deployment configuration values
│   └── templates/                     # Kubernetes manifests
├── Dockerfile                         # Container build and runtime definition
├── Jenkinsfile                        # Jenkins pipeline entrypoint
├── Jenkinsfile.ci                     # CI pipeline definition
├── Jenkinsfile.deploy                 # Deployment/testing pipeline definition
├── pyproject.toml                     # Project metadata, dependencies, and script entrypoints
├── uv.lock                            # Dependency lockfile
├── .dockerignore                      # Docker ignore rules
├── .gitignore                         # Git ignore rules
├── AGENTS.md                          # Agent/orchestration documentation
└── README.md                          # Project overview, setup, and usage guide
```

Conventions used (based on the available code):
- **src/ layout**: application code lives under `src/latest_ai_development/`.
- **YAML-driven configuration**: agent roles, task behavior, and model/provider defaults are externalized into `config/*.yaml`.
- **Generated artifacts**: `report.md` is the final markdown report produced by the two-agent orchestration.

## 4. Key Files and Configuration

- **`pyproject.toml`**  
  Declares project metadata, Python version constraints, dependencies, and console entrypoints. Incorrect edits can break installation, dependency resolution, or runtime commands.

- **`uv.lock`**  
  Stores the locked dependency graph for reproducible environments. Changing or removing it can make dependency resolution inconsistent across environments.

- **`src/latest_ai_development/main.py`**  
  Defines the FastAPI app, health endpoint, `/ask` endpoint, and CLI entry functions (`run`, `train`, `replay`, `test`, `run_with_trigger`). Incorrect edits can break API routes, server startup, or command execution.

- **`src/latest_ai_development/workflow.py`**  
  Implements the main LangChain two-agent sequential orchestration. Incorrect edits can break agent coordination, topic handoff, provider/model selection, or markdown report generation.

- **`src/latest_ai_development/prompts.py`**  
  Defines the prompt construction used by the researcher and reporting analyst agents. Incorrect edits can degrade output quality, break formatting expectations, or weaken agent/task alignment.

- **`src/latest_ai_development/config/agents.yaml`**  
  Defines agent role configuration, including role, goal, backstory, and behavioral instructions for the researcher and reporting analyst.

- **`src/latest_ai_development/config/tasks.yaml`**  
  Defines task configuration, including descriptions, expected outputs, and the sequential handoff between the two agents.

- **`src/latest_ai_development/config/models.yaml`**  
  Defines model/provider defaults used by the orchestration layer, including OpenAI and Ollama settings.

- **`src/latest_ai_development/config/settings.py`**  
  Resolves runtime configuration from environment variables, including provider selection, host/port, context path, output path, and config file locations.

- **`src/latest_ai_development/secrets_manager.py`**  
  Provides secret resolution support for AWS Secrets Manager and Azure Key Vault. Incorrect edits can cause runtime authentication or configuration failures.

- **`src/latest_ai_development/tools/custom_tool.py`**  
  Provides a starter example for extending the system with LangChain-compatible tools. Incorrect edits can break tool argument validation or invocation behavior.

- **`Dockerfile`**  
  Defines the container build and runtime startup process. Incorrect edits can prevent the image from building or the application from starting correctly.

- **`helm_chart/Chart.yaml`**  
  Helm chart metadata. Incorrect values can break chart packaging or installation.

- **`helm_chart/values.yaml`**  
  Deployment configuration defaults. Incorrect changes can break environment-specific deployments.

- **`helm_chart/templates/*`**  
  Kubernetes manifests for deployment resources. Incorrect edits can prevent workloads, services, or ingress resources from being created correctly.

- **`Jenkinsfile`**, **`Jenkinsfile.ci`**, **`Jenkinsfile.deploy`**  
  CI/CD pipeline definitions. Incorrect edits can break build, validation, publishing, or deployment flows.

- **`AGENTS.md`**  
  Project documentation for the two-agent orchestration. Stale edits may mislead users or maintainers, though they do not directly change runtime behavior.


If you want a slightly more detailed version, use this:

```md
## 5. Setup & Installation

A typical local setup for this LangChain template looks like:

### Prerequisites

- Python in the supported range: `>=3.11,<3.13`
- `uv` for dependency management
- A configured LLM provider:
  - `OPENAI` with `API_KEY_SECRET` or `OPENAI_API_KEY_SECRET`
  - `ANTHROPICAI` with `API_KEY_SECRET` or `ANTHROPICAI_API_KEY_SECRET`
  - `GEMINIAI` with `API_KEY_SECRET` or `GEMINIAI_API_KEY_SECRET`
  - or `OLLAMA` with a reachable `OLLAMA_BASE_URL`

### Secret Management

- Set `API_KEY_SECRET` to resolve hosted provider API keys from the configured secret backend.
- `OPENAI_API_KEY_SECRET` remains supported as a backward-compatible fallback when `API_KEY_SECRET` is not set.
- `ANTHROPICAI_API_KEY_SECRET` remains supported as a backward-compatible fallback when `API_KEY_SECRET` is not set.
- `GEMINIAI_API_KEY_SECRET` remains supported as a backward-compatible fallback when `API_KEY_SECRET` is not set.
- Set `CLOUD_PROVIDER=AWS` to retrieve secrets from AWS Secrets Manager.
- Set `CLOUD_PROVIDER=AZURE` to retrieve secrets from Azure Key Vault.
- When using AWS, configure `AWS_REGION` as needed.
- When using Azure, configure `AZURE_KEY_VAULT_URL` and an Azure identity supported by `DefaultAzureCredential`.
- Set `TRACING_BACKEND=LANGSMITH` to enable LangSmith tracing.
- Set `LANGSMITH_API_KEY_SECRET` to resolve the LangSmith API key through the configured secret backend.
- Leave `TRACING_BACKEND` unset, or set `TRACING_BACKEND=NONE`, to run without tracing.

### Install Dependencies

```bash
uv sync
```
**Start Server**
```bash
PYTHONPATH=src uv run python -m latest_ai_development.main
```

## 6. Development Guidelines

- **Keep orchestration logic in `workflow.py`, not in `main.py`.** `main.py` should remain a lightweight API and CLI entry layer, while `workflow.py` should own agent coordination, provider selection, and output generation.
- **Prefer YAML-driven changes for behavioral adjustments.** Agent roles, task behavior, and model/provider defaults should be maintained in `src/latest_ai_development/config/*.yaml`. Reserve code changes for orchestration, integrations, and runtime behavior.
- **Follow the `src/` layout consistently.** Imports and local execution rely on `PYTHONPATH=src`, and this convention should remain aligned across development, testing, and containerized execution.
- **Handle generated artifacts carefully.** `report.md` is the final output produced by the reporting agent. Changing output paths, filenames, or write logic in `workflow.py` or `settings.py` can affect downstream usage and automation.
- **Treat deployment and CI files as controlled assets.** Helm and Jenkins files appear to be standardized delivery templates; incorrect changes can break packaging, deployment, or CI/CD flows.
- **Avoid committing runtime artifacts.** Local caches, compiled Python files, generated reports, and other temporary outputs should be excluded from source control or cleaned regularly.
  
## 7. Security & Networking (HTTP / HTTPS)

- HTTP support is available out of the box.
- For secure deployments, customers are expected to enable and configure TLS/HTTPS after creating the template repository under Helm charts.
- While HTTP is supported, we strongly recommend using HTTPS for all production deployments.
