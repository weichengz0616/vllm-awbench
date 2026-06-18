# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy metadata and helpers for agent-aware KV cache management."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    FreeBlockEvictionPolicy,
    KVCacheBlock,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

PromptPart = Literal["fixed", "dynamic", "output"]
KVBlockStatus = Literal["gpu", "cpu", "loading", "offloading", "reserved"]
KVPoolClass = Literal["shared", "reserved"]
AgentKVEvictionPolicy = Literal["lru", "cachettl", "kvflow", "tokencake"]
ProgramType = Literal["dag", "react"]


@dataclass
class KVBlockPolicyMetadata:
    workflow_id: str | None = None
    program_id: str | None = None
    agent_id: str | None = None
    prompt_part: PromptPart = "output"
    step_contributions: dict[tuple[str, str], float] = field(default_factory=dict)
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
        self.step_contributions.clear()
        self.steps_to_execution = None
        self.ttl_deadline = None
        self.critical = False
        self.pool_class = pool_class
        self.status = "gpu"


@dataclass
class KVRequestPolicyMetadata:
    # Common agent/workflow identifiers.
    # workflow_id identifies the workflow template; program_id identifies a
    # concrete workflow instance. External requests may pass template_id as an
    # alias for workflow_id.
    workflow_id: str | None = None
    program_id: str | None = None
    program_type: ProgramType | None = None
    agent_id: str | None = None
    fixed_prefix_len: int | None = None

    # KVFlow-specific metadata.
    steps_to_execution: float | None = None
    agent_steps_to_execution: dict[str, float] = field(default_factory=dict)
    next_agent_ids: list[str] = field(default_factory=list)

    # CacheTTL / Tokencake metadata.
    ttl_seconds: float | None = None
    is_program_last_step: bool = False
    critical: bool = False

    # Offload / tool-call metadata.
    session_id: str | None = None
    predicted_tool_time: float | None = None
    call_name: str | None = None
    call_duration: float | None = None
    function_event: Literal["call_start", "call_finish"] | None = None

    @classmethod
    def from_extra_args(
        cls, extra_args: dict[str, Any] | None
    ) -> "KVRequestPolicyMetadata":
        if not extra_args:
            return cls()
        raw = extra_args.get("awbench_meta")
        if not isinstance(raw, dict):
            return cls()
        steps_by_agent = raw.get("agent_steps_to_execution") or {}
        if not isinstance(steps_by_agent, dict):
            steps_by_agent = {}
        next_agent_ids = raw.get("next_agent_ids") or raw.get("next_agents") or []
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
            fixed_prefix_len=_as_int(raw.get("fixed_prefix_len")),
            steps_to_execution=_as_float(raw.get("steps_to_execution")),
            agent_steps_to_execution={
                str(k): float(v)
                for k, v in steps_by_agent.items()
                if _is_number(v)
            },
            next_agent_ids=[str(agent_id) for agent_id in next_agent_ids],
            ttl_seconds=_as_float(raw.get("ttl_seconds")),
            is_program_last_step=bool(raw.get("is_program_last_step", False)),
            critical=bool(raw.get("critical", False)),
            session_id=_as_str(raw.get("session_id")),
            predicted_tool_time=_as_float(
                raw.get("predicted_tool_time")
                if raw.get("predicted_tool_time") is not None
                else raw.get("predict_time")
            ),
            call_name=_as_str(raw.get("call_name") or raw.get("function_name")),
            call_duration=_as_float(
                raw.get("call_duration")
                if raw.get("call_duration") is not None
                else raw.get("predict_time")
            ),
            function_event=raw.get("function_event")
            if raw.get("function_event") in ("call_start", "call_finish")
            else None,
        )

    def step_for_agent(self, agent_id: str | None) -> float | None:
        if agent_id is not None and agent_id in self.agent_steps_to_execution:
            return self.agent_steps_to_execution[agent_id]
        return self.steps_to_execution

    @property
    def workflow_key(self) -> str | None:
        """Identifier for workflow-template scoped agent state."""
        return self.workflow_id


@dataclass
class EvictionContext:
    now: float
    metadata_for_block: Callable[[KVCacheBlock], KVBlockPolicyMetadata]
    free_block_pressure: float = 0.0
    request_metadata: KVRequestPolicyMetadata | None = None


@dataclass
class RequestMetadataContext:
    request: "Request"
    metadata_for_block: Callable[[KVCacheBlock], KVBlockPolicyMetadata]
    get_agent_live_blocks: Callable[
        [str, str],
        dict[BlockHashWithGroupId, list[KVCacheBlock]],
    ]
    iter_block_metadata: Callable[[], Iterable[KVBlockPolicyMetadata]]


class EvictionContextProvider(Protocol):
    def __call__(self) -> EvictionContext: ...


def make_free_block_eviction_policy(
    policy_name: AgentKVEvictionPolicy,
    context_provider: EvictionContextProvider | None,
) -> FreeBlockEvictionPolicy | None:
    if policy_name == "lru":
        return None
    if context_provider is None:
        return None
    if policy_name == "cachettl":
        return CacheTTLEvictionPolicy(context_provider)
    if policy_name == "kvflow":
        return KVFlowEvictionPolicy(context_provider)
    if policy_name == "tokencake":
        return TokencakeEvictionPolicy(context_provider)
    return None


class BaseEvictionPolicy(FreeBlockEvictionPolicy):
    policy_name: AgentKVEvictionPolicy = "lru"

    def __init__(self, context_provider: EvictionContextProvider):
        self.context_provider = context_provider

    def on_request_metadata(self, context: RequestMetadataContext) -> None:
        return

    def on_block_metadata_bound(
        self,
        context: RequestMetadataContext,
        block: KVCacheBlock,
    ) -> None:
        return

    def on_cached_block_metadata_hit(
        self,
        context: RequestMetadataContext,
        block: KVCacheBlock,
        prompt_part: PromptPart,
    ) -> None:
        return

    def insert_free_blocks(self, queue, blocks: list[KVCacheBlock]) -> None:
        queue._append_tail_n(blocks)

    def _select_by_key(self, queue, n: int, key_fn) -> list[KVCacheBlock]:
        candidates = queue.get_all_free_blocks()
        selected = sorted(
            enumerate(candidates), key=lambda item: key_fn(item[1], item[0])
        )[:n]
        blocks = [block for _, block in selected]
        for block in blocks:
            queue.remove(block)
        return blocks


class CacheTTLEvictionPolicy(BaseEvictionPolicy):
    """TTL-aware policy with LRU fallback."""

    policy_name: AgentKVEvictionPolicy = "cachettl"

    def select_victims(self, queue, n: int) -> list[KVCacheBlock]:
        ctx = self.context_provider()

        def key(block: KVCacheBlock, lru_index: int):
            meta = ctx.metadata_for_block(block)
            if meta.status in ("loading", "offloading", "reserved"):
                status_rank = 2
            elif meta.ttl_deadline is None:
                status_rank = 1
            elif meta.ttl_deadline <= ctx.now:
                status_rank = 0
            else:
                status_rank = 2
            deadline = (
                meta.ttl_deadline if meta.ttl_deadline is not None else float("inf")
            )
            return (status_rank, deadline, lru_index)

        return self._select_by_key(queue, n, key)


class KVFlowEvictionPolicy(BaseEvictionPolicy):
    """Workflow-aware policy for KVFlow."""

    policy_name: AgentKVEvictionPolicy = "kvflow"

    @staticmethod
    def _set_step_contribution(
        metadata: KVBlockPolicyMetadata,
        program_id: str,
        agent_id: str,
        step: float,
    ) -> None:
        metadata.step_contributions[(program_id, agent_id)] = step
        KVFlowEvictionPolicy._refresh_steps_to_execution(metadata)

    @staticmethod
    def _remove_program_contributions(
        metadata: KVBlockPolicyMetadata,
        program_id: str,
    ) -> None:
        for key in list(metadata.step_contributions):
            if key[0] == program_id:
                del metadata.step_contributions[key]
        KVFlowEvictionPolicy._refresh_steps_to_execution(metadata)

    @staticmethod
    def _refresh_steps_to_execution(metadata: KVBlockPolicyMetadata) -> None:
        metadata.steps_to_execution = (
            min(metadata.step_contributions.values())
            if metadata.step_contributions
            else None
        )

    def on_block_metadata_bound(
        self,
        context: RequestMetadataContext,
        block: KVCacheBlock,
    ) -> None:
        req_metadata = context.request.kv_cache_policy_metadata
        metadata = context.metadata_for_block(block)
        step = req_metadata.step_for_agent(req_metadata.agent_id)
        if (
            metadata.prompt_part == "fixed"
            and req_metadata.program_id is not None
            and req_metadata.agent_id is not None
            and step is not None
        ):
            self._set_step_contribution(
                metadata,
                req_metadata.program_id,
                req_metadata.agent_id,
                step,
            )

    def on_cached_block_metadata_hit(
        self,
        context: RequestMetadataContext,
        block: KVCacheBlock,
        prompt_part: PromptPart,
    ) -> None:
        req_metadata = context.request.kv_cache_policy_metadata
        step = req_metadata.step_for_agent(req_metadata.agent_id)
        if (
            prompt_part == "fixed"
            and req_metadata.program_id is not None
            and req_metadata.agent_id is not None
            and step is not None
        ):
            self._set_step_contribution(
                context.metadata_for_block(block),
                req_metadata.program_id,
                req_metadata.agent_id,
                step,
            )

    def on_request_metadata(self, context: RequestMetadataContext) -> None:
        req_metadata = context.request.kv_cache_policy_metadata
        workflow_id = req_metadata.workflow_key
        program_id = req_metadata.program_id
        if workflow_id is None or program_id is None:
            return

        if req_metadata.is_program_last_step:
            for metadata in context.iter_block_metadata():
                if metadata.workflow_id == workflow_id:
                    self._remove_program_contributions(metadata, program_id)
            return

        agent_steps = req_metadata.agent_steps_to_execution
        if (
            not agent_steps
            and req_metadata.agent_id is not None
            and req_metadata.steps_to_execution is not None
        ):
            agent_steps = {req_metadata.agent_id: req_metadata.steps_to_execution}

        for agent_id, step in agent_steps.items():
            blocks_by_hash = context.get_agent_live_blocks(workflow_id, agent_id)
            for blocks in blocks_by_hash.values():
                for block in blocks:
                    self._set_step_contribution(
                        context.metadata_for_block(block),
                        program_id,
                        agent_id,
                        step,
                    )

    def select_victims(self, queue, n: int) -> list[KVCacheBlock]:
        ctx = self.context_provider()

        def key(block: KVCacheBlock, lru_index: int):
            meta = ctx.metadata_for_block(block)
            if meta.status in ("loading", "offloading", "reserved"):
                return (4, 0, lru_index)
            if meta.prompt_part in ("dynamic", "output"):
                return (0, 0, lru_index)
            if meta.prompt_part == "fixed" and meta.steps_to_execution is not None:
                return (1, -meta.steps_to_execution, lru_index)
            return (2, 0, lru_index)

        return self._select_by_key(queue, n, key)


class TokencakeEvictionPolicy(BaseEvictionPolicy):
    """Reserved-pool aware policy for Tokencake."""

    policy_name: AgentKVEvictionPolicy = "tokencake"

    def select_victims(self, queue, n: int) -> list[KVCacheBlock]:
        ctx = self.context_provider()
        req_meta = ctx.request_metadata
        critical_request = bool(req_meta and req_meta.critical)

        def key(block: KVCacheBlock, lru_index: int):
            meta = ctx.metadata_for_block(block)
            if meta.status in ("loading", "offloading", "reserved"):
                return (4, lru_index)
            if critical_request:
                return (0 if meta.pool_class == "reserved" else 1, lru_index)
            return (0 if meta.pool_class == "shared" else 3, lru_index)

        return self._select_by_key(queue, n, key)


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
    if request_metadata.is_program_last_step:
        return None
    ttl_seconds = request_metadata.ttl_seconds
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


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
