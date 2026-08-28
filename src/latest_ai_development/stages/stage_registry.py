from latest_ai_development.stages.metadata_fetching_agent_stage import MetadataFetchingAgentStage
from latest_ai_development.stages.response_generation_agent_stage import ResponseGenerationAgentStage
from latest_ai_development.stages.query_execution_agent_stage import QueryExecutionAgentStage
from latest_ai_development.stages.sql_generator_agent_stage import SqlGeneratorAgentStage
from latest_ai_development.stages.validator_agent_stage import ValidatorAgentStage


STAGE_REGISTRY = {
    "metadata_fetching_agent": MetadataFetchingAgentStage,
    "sql_generator_agent": SqlGeneratorAgentStage,
    "validator_agent": ValidatorAgentStage,
    "query_execution_agent": QueryExecutionAgentStage,
    "response_generation_agent": ResponseGenerationAgentStage,
}