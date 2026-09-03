# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from collections.abc import Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias

import regex as re
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        if any(k not in field_names for k in data):
            logger.debug(
                "The following fields were present in the request but ignored: %s",
                data.keys() - field_names,
            )
        return result


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None


class VllmRequestMetrics(OpenAIBaseModel):
    """Detailed execution metrics for a completed vLLM request."""

    request_id: str
    num_engine_requests: int
    arrival_time_unix_seconds: float
    finished_time_unix_seconds: float
    queue_time_seconds: float
    time_to_first_token_seconds: float
    prefill_time_seconds: float
    decode_time_seconds: float
    inference_time_seconds: float
    e2e_latency_seconds: float
    mean_time_per_output_token_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int
    recomputed_prompt_tokens: int
    local_computed_prompt_tokens: int
    local_cache_hit_prompt_tokens: int
    external_kv_transfer_prompt_tokens: int
    num_preemptions: int
    is_corrupted: bool
    requested_max_tokens: int | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


def build_vllm_request_metrics(
    request_id: str,
    request_stats: Sequence["RequestStateStats | None"],
    *,
    requested_max_tokens: int | None = None,
    finish_reason: str | None = None,
    stop_reason: int | str | None = None,
) -> VllmRequestMetrics | None:
    """Build response metrics from one or more completed engine requests."""

    stats = [stat for stat in request_stats if stat is not None]
    if not stats:
        return None

    def duration(end: float, start: float) -> float:
        if end <= 0.0 or start <= 0.0:
            return 0.0
        return max(0.0, end - start)

    arrival_time = min(stat.arrival_time for stat in stats)
    finished_time = max(stat.finished_time for stat in stats)
    queue_time = max(duration(stat.scheduled_ts, stat.queued_ts) for stat in stats)
    prefill_time = max(
        duration(stat.first_token_ts, stat.scheduled_ts) for stat in stats
    )
    decode_time = max(
        duration(stat.last_token_ts, stat.first_token_ts) for stat in stats
    )
    inference_time = max(
        duration(stat.last_token_ts, stat.scheduled_ts) for stat in stats
    )

    prompt_tokens = sum(stat.num_prompt_tokens for stat in stats)
    completion_tokens = sum(stat.num_generation_tokens for stat in stats)
    cached_tokens = sum(stat.num_cached_tokens for stat in stats)
    recomputed_tokens = sum(stat.num_recomputed_tokens for stat in stats)
    external_tokens = sum(stat.num_external_computed_tokens for stat in stats)
    generated_after_first = sum(
        max(0, stat.num_generation_tokens - 1) for stat in stats
    )
    mean_time_per_output_token = (
        sum(
            duration(stat.last_token_ts, stat.first_token_ts) for stat in stats
        )
        / generated_after_first
        if generated_after_first
        else 0.0
    )

    return VllmRequestMetrics(
        request_id=request_id,
        num_engine_requests=len(stats),
        arrival_time_unix_seconds=arrival_time,
        finished_time_unix_seconds=finished_time,
        queue_time_seconds=queue_time,
        time_to_first_token_seconds=max(
            stat.first_token_latency for stat in stats
        ),
        prefill_time_seconds=prefill_time,
        decode_time_seconds=decode_time,
        inference_time_seconds=inference_time,
        e2e_latency_seconds=max(0.0, finished_time - arrival_time),
        mean_time_per_output_token_seconds=mean_time_per_output_token,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_prompt_tokens=cached_tokens,
        recomputed_prompt_tokens=recomputed_tokens,
        local_computed_prompt_tokens=prompt_tokens - cached_tokens,
        local_cache_hit_prompt_tokens=(
            cached_tokens + recomputed_tokens - external_tokens
        ),
        external_kv_transfer_prompt_tokens=external_tokens,
        num_preemptions=sum(stat.num_preemptions for stat in stats),
        is_corrupted=any(stat.is_corrupted for stat in stats),
        requested_max_tokens=requested_max_tokens,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: UsageInfo | None = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: str | None = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class LegacyStructuralTag(OpenAIBaseModel):
    begin: str
    # schema is the field, but that causes conflicts with pydantic so
    # instead use structural_tag_schema with an alias
    structural_tag_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    end: str


class LegacyStructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    structures: list[LegacyStructuralTag]
    triggers: list[str]


class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    format: Any


AnyStructuralTagResponseFormat: TypeAlias = (
    LegacyStructuralTagResponseFormat | StructuralTagResponseFormat
)


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None


AnyResponseFormat: TypeAlias = (
    ResponseFormat | StructuralTagResponseFormat | LegacyStructuralTagResponseFormat
)


class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None = False
    continuous_usage_stats: bool | None = False


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    defer_loading: bool | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


# extra="forbid" is a workaround to have kwargs as a field,
# see https://github.com/pydantic/pydantic/issues/3125
class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


LogitsProcessors = list[str | LogitsProcessorConstructor]


def get_logits_processors(
    processors: LogitsProcessors | None, pattern: str | None
) -> list[Any] | None:
    if processors and pattern:
        logits_processors = []
        for processor in processors:
            qualname = processor if isinstance(processor, str) else processor.qualname
            if not re.match(pattern, qualname):
                raise ValueError(
                    f"Logits processor '{qualname}' is not allowed by this "
                    "server. See --logits-processor-pattern engine argument "
                    "for more information."
                )
            try:
                logits_processor = resolve_obj_by_qualname(qualname)
            except Exception as e:
                raise ValueError(
                    f"Logits processor '{qualname}' could not be resolved: {e}"
                ) from e
            if isinstance(processor, LogitsProcessorConstructor):
                logits_processor = logits_processor(
                    *processor.args or [], **processor.kwargs or {}
                )
            logits_processors.append(logits_processor)
        return logits_processors
    elif processors:
        raise ValueError(
            "The `logits_processors` argument is not supported by this "
            "server. See --logits-processor-pattern engine argument "
            "for more information."
        )
    return None


class FunctionCall(OpenAIBaseModel):
    # Internal field to preserve native tool call ID from tool parser.
    # Excluded from serialization to maintain OpenAI API compatibility
    # (function object should only contain 'name' and 'arguments').
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: str | None = None


class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class GenerationError(Exception):
    """raised when finish_reason indicates internal server error (500)"""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message)
        self.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
