# Agents Documentation

## Overview

This project implements a LangChain-based two-agent system for researching and reporting on various topics. It uses two specialized agents in a sequential process to gather information and generate a comprehensive markdown report.

## Architecture

The system uses LangChain with FastAPI integration and follows a sequential multi-agent orchestration pattern for research and report generation.

- **LLM Provider Selection**: Controlled by `PROVIDER` environment variable
  - `PROVIDER=OPENAI`: Uses OpenAI chat models (requires `OPENAI_API_KEY` or `OPENAI_API_KEY_SECRET`)
  - `PROVIDER=OLLAMA` (default): Uses Ollama with `llama3.2`
- **Coordination Pattern**: Sequential multi-agent orchestration
- **Framework**: LangChain with FastAPI integration
- **Python Version**: 3.11 - 3.12

## Agents

### 1. Researcher Agent

**Role**: `{topic} Senior Data Researcher`

**Goal**: Uncover relevant, accurate, and practical developments in the specified topic.

**Backstory**:  
A seasoned researcher with a knack for uncovering the latest developments in any given topic. Known for the ability to find the most relevant information and present it in a clear and concise manner.

**Runtime Behavior**:
- Uses the provider and model configuration resolved at runtime
- Focuses on gathering relevant findings for the requested topic
- Produces structured research notes for downstream use
- Does not generate the final polished report

**Primary Responsibility**: Research Task
- Conducts focused research on the specified topic
- Uses available context and knowledge inputs when provided
- Identifies relevant findings, trends, and supporting details
- Outputs structured research notes for the reporting agent

### 2. Reporting Analyst Agent

**Role**: `{topic} Reporting Analyst`

**Goal**: Create detailed reports based on research findings.

**Backstory**:  
A meticulous analyst with a keen eye for detail. Known for the ability to turn complex information into clear and concise reports, making it easy for others to understand and act on the information provided.

**Runtime Behavior**:
- Uses the provider and model configuration resolved at runtime
- Consumes the research output from the previous agent
- Expands findings into a clear markdown report
- Saves final output to `report.md`

**Primary Responsibility**: Reporting Task
- Reviews the research context produced by the researcher agent
- Expands major findings into clear sections
- Produces a complete markdown report with actionable structure
- Outputs the final report to `report.md`


## Workflow

The agents work in a sequential process:

```text
1. Researcher Agent
   └─> Investigates the requested topic
       └─> Produces structured research findings
           └─> Hands off context to the reporting agent

2. Reporting Analyst Agent
   └─> Consumes the researcher’s findings
       └─> Organizes them into a structured markdown report
           └─> Writes the final output to `report.md`
```

## Configuration Files

### `agents.yaml`

Located at: `src/latest_ai_development/config/agents.yaml`

Defines agent configuration for the orchestration, including:

- role
- goal
- backstory
- behavioral instructions

### `tasks.yaml`

Located at: `src/latest_ai_development/config/tasks.yaml`

Defines task configuration for agent coordination, including:

- task descriptions
- expected outputs
- execution order across agents

### `models.yaml`

Located at: `src/latest_ai_development/config/models.yaml`

Defines provider-specific model defaults, including:

- chat model name
- temperature
- max token configuration

### `settings.py`

Located at: `src/latest_ai_development/config/settings.py`

Loads runtime settings from environment variables and resolves:

- provider
- host / port
- context path
- output path
- config file paths

## Usage

### Running the Agent Orchestration

The system can be executed in multiple ways.

#### 1. Direct Execution

```python
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow
from datetime import datetime

inputs = {
    "topic": "Your Topic Here",
    "current_year": int(datetime.now().year),
    "save_output": True,
}

LatestAiDevelopmentWorkflow().kickoff(inputs=inputs)
```
#### 2. FastAPI Endpoint
```bash
POST /ask
{
  "topic": "Your Topic Here"
}
```
#### 3. Command Line
```bash
# Run the workflow
latest_ai_development

# Train-style repeated execution
train <n_iterations> <filename>

# Replay from payload file or topic
replay <task_id>

# Test the workflow
test <n_iterations> <eval_llm>

# Run with trigger payload
run_with_trigger '<json_payload>'
```

## Environment Variables

- `PROVIDER`: Selects the LLM provider (`OPENAI` or `OLLAMA`, default: `OLLAMA`)
- `OPENAI_API_KEY`: API key for the OpenAI provider
- `OPENAI_API_KEY_SECRET`: AWS Secrets Manager secret name used to resolve the OpenAI API key
- `OLLAMA_BASE_URL`: Base URL for the Ollama API
- `PORT`: Server port (default: `8080`)
- `API_HOST`: Server host (default: `0.0.0.0`)
- `AWS_REGION`: AWS region for Secrets Manager (default: `us-east-1`)
- `CONTEXT`: Path used for knowledge and context files (**mandatory**)
- `REPORT_OUTPUT_FILE`: Output report filename (default: `report.md`)

## Integration Features

### FastAPI Server

- Health check endpoint: `GET /health`
- Workflow execution endpoint: `POST /ask`
- Root path: `/testing`

### AWS Secrets Manager

The system includes secret resolution support for secure credential management:

- Retrieves secrets from AWS Secrets Manager
- Supports both full secret retrieval and specific key extraction
- Uses configurable AWS region

### Custom Tools

The system supports custom tool development using LangChain-compatible tools.  
Example template available at: `src/latest_ai_development/tools/custom_tool.py`

Custom tools can be implemented as:

- callable utilities
- LangChain `@tool` functions
- other LangChain-compatible helpers invoked from the orchestration

## Output

The final output is a markdown report (`report.md`) containing:

- the main findings identified during research
- structured sections organized by the reporting agent
- relevant insights, details, and practical takeaways
- formatting without markdown code fences for clean presentation


## Deployment

The project includes:
- Docker support (Dockerfile, .dockerignore)
- Kubernetes Helm charts (helm_chart/)
- Jenkins CI/CD pipelines (Jenkinsfile, Jenkinsfile.ci, Jenkinsfile.deploy)


## Dependencies

Key dependencies include:

- `langchain` - Core LangChain framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community integrations
- `langchain-ollama` - Ollama integration
- `fastapi` - API server
- `uvicorn` - ASGI server
- `boto3` - AWS integration
- `pydantic` - Data validation
- `pydantic-settings` - Environment-based settings management
- `PyYAML` - YAML configuration loading

## Extending the System

### Adding New Roles

1. Define the role configuration in `agents.yaml`
2. Add or update role-specific instructions in `prompts.py`
3. Add or update orchestration logic in `workflow.py`

### Adding New Tasks

1. Define the task configuration in `tasks.yaml`
2. Update task sequencing in `workflow.py`
3. Define how outputs are passed between agents

### Creating Custom Tools

1. Create a LangChain-compatible tool in `tools/custom_tool.py`
2. Define an input schema using Pydantic if needed
3. Implement the callable or `@tool`
4. Connect the tool within the orchestration where required

## Restrictions and Guidelines

### Environment Variables
- **PORT**: MUST be read from environment variable. Do not hardcode port values.
- **CONTEXT**: Context path is MANDATORY for running the agent. Must be set as an environment variable.

### Dockerfile
- The Dockerfile start command MUST NOT be modified
- Any changes made to the Dockerfile MUST adhere to the existing start command
- Maintain compatibility with the current container startup process

### Protected Files
The following files and directories are PROTECTED and should NOT be modified:
- `Jenkinsfile`
- `Jenkinsfile.ci`
- `Jenkinsfile.deploy`
- `helm_chart/` (entire directory and all contents)

These files are managed by the DevOps team and any changes could break the CI/CD pipeline or deployment process.

## Notes

- The system uses a sequential process by default
- Hierarchical process is available as an alternative
- All agents share the same LLM configuration
- Knowledge sources can be added to enhance agent capabilities
- The system supports training, testing, and replay functionality for iterative improvement