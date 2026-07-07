# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy metadata and helpers for agent-aware KV cache management."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

if TYPE_CHECKING:
    from vllm.v1.request import Request

PromptPart = Literal["fixed", "dynamic", "output"]
KVBlockStatus = Literal["gpu", "cpu", "loading", "offloading", "reserved"]
KVPoolClass = Literal["shared", "reserved"]
AgentKVEvictionPolicy = Literal["lru", "cachettl", "kvflow", "tokencake"]
ProgramType = Literal["dag", "react"]
NextOpType = Literal["tool", "llm", "done"]


@dataclass
class KVBlockPolicyMetadata:
    workflow_id: str | None = None
    program_id: str | None = None
    agent_id: str | None = None
    prompt_part: PromptPart = "output"
    steps_to_execution: float | None = None
    ttl_deadline: float | None = None
    critical: bool = False
    pool_class: KVPoolClass = "shared"
    status: KVBlockStatus = "gpu"

    def reset_for_allocation(self, pool_class: KVPoolClass) -> None:
        self.workflow_id = None
        self.program_id = None
        self.agent_id = None
        self.prompt_part = "output"
        self.steps_to_execution = None
        self.ttl_deadline = None
        self.critical = False
        self.pool_class = pool_class
        self.status = "gpu"


@dataclass
class KVRequestPolicyMetadata:
    # Common agent/workflow identifiers.
    workflow_id: str | None = None
    program_id: str | None = None
    program_type: ProgramType | None = None
    agent_id: str | None = None
    op_id: str | None = None
    fixed_prefix_len: int | None = None

    # KVFlow-specific metadata.
    agent_steps_to_execution: dict[str, float] = field(default_factory=dict)
    next_agent_ids: list[str] = field(default_factory=list)

    # CacheTTL-specific metadata.
    cachettl_should_pin: bool | None = None
    cachettl_ttl_seconds: float | None = None
    cachettl_is_last_step: bool | None = None

    critical: bool = False

    # Tokencake-specific metadata.
    session_id: str | None = None
    next_op_type: NextOpType | None = None
    next_tool_type: str | None = None
    predicted_tool_time: float | None = None
    multi_turn_kv_reuse: bool = False
    static_priority: float | None = None

    @classmethod
    def from_extra_args(
        cls, extra_args: dict[str, Any] | None
    ) -> "KVRequestPolicyMetadata":
        if not extra_args:
            return cls()
        raw = extra_args.get("awbench_meta")
        if not isinstance(raw, dict):
            return cls()
        steps_by_agent = (
            raw.get("agent_steps_to_execution")
            or raw.get("agent_next_call_distance")
            or {}
        )
        if not isinstance(steps_by_agent, dict):
            steps_by_agent = {}
        next_agent_ids = raw.get("next_agent_ids")
        if next_agent_ids is None:
            next_agent_ids = raw.get("next_agents")
        if next_agent_ids is None:
            next_agent_ids = [
                agent_id
                for agent_id, distance in steps_by_agent.items()
                if _is_number(distance) and float(distance) == 1
            ]
        if isinstance(next_agent_ids, str):
            next_agent_ids = [next_agent_ids]
        if not isinstance(next_agent_ids, list):
            next_agent_ids = []
        program_type = _as_str(raw.get("program_type"))
        if program_type is not None:
            program_type = program_type.lower()

        return cls(
            workflow_id=_as_str(raw.get("template_id"))
            or _as_str(raw.get("workflow_id")),
            program_id=_as_str(raw.get("program_id")),
            program_type=program_type if program_type in ("dag", "react") else None,
            agent_id=_as_str(raw.get("agent_id")),
            op_id=_as_str(raw.get("op_id")),
            fixed_prefix_len=_as_int(raw.get("fixed_prefix_len")),
            agent_steps_to_execution={
                str(k): float(v)
                for k, v in steps_by_agent.items()
                if _is_number(v)
            },
            next_agent_ids=[str(agent_id) for agent_id in next_agent_ids],
            cachettl_should_pin=_as_bool(raw.get("cachettl_should_pin")),
            cachettl_ttl_seconds=_as_float(raw.get("cachettl_ttl_seconds")),
            cachettl_is_last_step=_as_bool(raw.get("cachettl_is_last_step")),
            critical=bool(raw.get("critical", False)),
            session_id=_as_str(raw.get("session_id")),
            next_op_type=raw.get("next_op_type")
            if raw.get("next_op_type") in ("tool", "llm", "done")
            else None,
            next_tool_type=_as_str(
                raw.get("next_tool_type")
                or raw.get("tool_type")
                or raw.get("tool_name")
            ),
            predicted_tool_time=_as_float(
                raw.get("predicted_tool_time")
                if raw.get("predicted_tool_time") is not None
                else raw.get("predict_time")
            ),
            multi_turn_kv_reuse=bool(
                _as_bool(
                    raw.get("multi_turn_kv_reuse")
                    if raw.get("multi_turn_kv_reuse") is not None
                    else raw.get("kv_reuse_after_tool")
                )
            ),
            static_priority=_as_float(raw.get("static_priority")),
        )

    def step_for_agent(self, agent_id: str | None) -> float | None:
        if agent_id is None:
            return None
        return self.agent_steps_to_execution.get(agent_id)

    @property
    def workflow_key(self) -> str | None:
        return self.workflow_id


@dataclass
class RequestMetadataContext:
    request: "Request"
    metadata_for_block: Callable[[KVCacheBlock], KVBlockPolicyMetadata]
    get_agent_live_blocks: Callable[
        [str, str],
        dict[BlockHashWithGroupId, list[KVCacheBlock]],
    ]
    iter_block_metadata: Callable[[], Iterable[KVBlockPolicyMetadata]]



def get_prompt_part(
    block_index: int,
    block_size: int,
    fixed_prefix_len: int | None,
    num_prompt_tokens: int,
) -> PromptPart:
    aligned_fixed_prefix_len = align_fixed_prefix_len(
        fixed_prefix_len, block_size
    )
    if aligned_fixed_prefix_len is None:
        aligned_fixed_prefix_len = 0
    block_start = block_index * block_size
    if block_start < aligned_fixed_prefix_len:
        return "fixed"
    if block_start < num_prompt_tokens:
        return "dynamic"
    return "output"


def align_fixed_prefix_len(
    fixed_prefix_len: int | None,
    block_size: int,
) -> int | None:
    if fixed_prefix_len is None:
        return None
    if fixed_prefix_len <= 0:
        return 0
    return fixed_prefix_len // block_size * block_size


def ttl_deadline(now: float, request_metadata: KVRequestPolicyMetadata) -> float | None:
    ttl_seconds = request_metadata.cachettl_ttl_seconds
    if ttl_seconds is None or ttl_seconds <= 0:
        return None
    return now + ttl_seconds


def monotonic_time() -> float:
    return time.monotonic()


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_float(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
