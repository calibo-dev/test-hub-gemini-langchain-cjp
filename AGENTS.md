# Workflow and Agent Documentation

## Overview

This project implements a FastAPI-based LangChain workflow that safely answers natural language questions about the `ak_country` Snowflake table. The workflow runs five sequential stages to inspect metadata, translate English queries into SQL, validate those queries, execute safe SQL, and emit a user-ready response. No artifacts are persisted; the `/ask` HTTP POST returns the final `response`.

## Workflow Topology

1. **MetadataFetchingAgent (`metadata_fetching_agent`)**
   - **Purpose**: Use `CountryAgenticTool` to gather column names, types, and schema constraints for `ak_country` so downstream stages can rely on accurate structure data.
   - **Inputs**: None beyond the request context (`current_year`, `knowledge_context`).
   - **Outputs**: `metadata` (structured column catalog, types, constraints).
   - **Tools**: `CountryAgenticTool`.

2. **SQLGeneratorAgent (`sql_generator_agent`)**
   - **Purpose**: Convert the user's `user_query` plus `metadata` into a candidate SQL SELECT statement that only touches validated columns. Flag unsupported requests via `out_of_scope`.
   - **Inputs**: `user_query`, `metadata`, `current_year`, `knowledge_context`.
   - **Outputs**: `sql_query`, `out_of_scope`.
   - **Tools**: None (pure LLM chain).

3. **ValidatorAgent (`validator_agent`)**
   - **Purpose**: Ensure the candidate SQL remains read-only, references only valid columns, and respects ak_country scope. If validation fails, clear the SQL and set `out_of_scope=true`.
   - **Inputs**: `sql_query`, `out_of_scope`.
   - **Outputs**: `validated_query`, `out_of_scope`.
   - **Tools**: None.

4. **QueryExecutionAgent (`query_execution_agent`)**
   - **Purpose**: Execute the validated SQL through `CountryAgenticTool` when `out_of_scope=false`. Capture raw rows or error details. Skip execution when the request is out of scope.
   - **Inputs**: `validated_query`, `out_of_scope`, `knowledge_context`.
   - **Outputs**: `results`, `out_of_scope`.
   - **Tools**: `CountryAgenticTool`.

5. **ResponseGenerationAgent (`response_generation_agent`)**
   - **Purpose**: Craft the final natural language `response` from query `results` or from the `out_of_scope` indicator.
   - **Inputs**: `results`, `out_of_scope`, `user_query`, `current_year`, `knowledge_context`.
   - **Outputs**: `response`.
   - **Tools**: None.

## Stage Execution Details

- Each stage is wired through `stage_registry.py` using its canonical stage key (e.g., `metadata_fetching_agent`).
- Prompts are built in `src/latest_ai_development/prompts/prompt_builder.py`, which also enforces the security guidelines and supplies format instructions for structured parsers.
- Tool-calling stages (metadata and query execution) use agent builders in `src/latest_ai_development/agents/` that bind stage-specific tools via `tool_registry.get_tools(stage_name)`.
- Non-tool stages compose LCEL chains in `src/latest_ai_development/chains/`, leveraging `JsonOutputParser` or `StrOutputParser` plus LangChain's Runnable API.
- `LatestAiDevelopmentWorkflow` orchestrates the stages sequentially, tracing every stage via `trace_stage_execution` and wrapping the workflow with `@trace_workflow`.

## Tools

- `CountryAgenticTool`  
  - Handles Snowflake credentials via `latest_ai_development.tools.snowflake_connection`.  
  - Supports metadata inspection and SQL execution.  
  - Wired to `metadata_fetching_agent` and `query_execution_agent` through `tool_registry`.

## FastAPI Interface

### POST /ask

- Payload:
  ```json
  {
    "user_query": "List all Caribbean countries with their ISO alpha-3 codes.",
    "knowledge_context": ""
  }
  ```
- Response:
  ```json
  {
    "user_query": "List all Caribbean countries with their ISO alpha-3 codes.",
    "response": "Caribbean countries include ..."
  }
  ```
- The API returns only the canonical `response` from the final stage. No files are written to disk.

## Tracing and Observability

- LangSmith tracing is enabled via `TRACING_BACKEND=LANGSMITH`. Provide the API key via `LANGSMITH_API_KEY_SECRET` and optionally group traces with `LANGSMITH_PROJECT` and `LANGSMITH_ENDPOINT`.
- `LANGCHAIN_API_KEY_SECRET` is also honored when wiring LangSmith.
- Each stage logs start/completion along with the `flow_run_id`.
