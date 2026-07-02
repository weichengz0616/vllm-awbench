# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import PREFETCH_POOL_REQ_ID
from vllm.v1.core.kv_cache_utils import BlockHash

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVBlockRange:
    """A logical range of KV blocks for a request.

    `gpu_block_ids` are destination blocks for load plans and source blocks for
    offload plans.
    """

    req_id: str
    start_block_idx: int
    num_blocks: int
    block_hashes: list[BlockHash] = field(default_factory=list)
    gpu_block_ids: list[int] = field(default_factory=list)
    workflow_id: str | None = None
    agent_id: str | None = None


@dataclass
class KVLoadPlan:
    """Logical load decisions produced by an offload policy."""

    handled: bool = False
    num_external_tokens: int | None = 0
    load_async: bool = False
    block_ranges: list[KVBlockRange] = field(default_factory=list)


@dataclass
class KVOffloadPlan:
    """Logical offload decisions produced by an offload policy."""

    block_ranges: list[KVBlockRange] = field(default_factory=list)


@dataclass
class KVTransferPlan:
    """Per-step logical KV transfer plan.

    The plan intentionally describes request/block intent rather than concrete
    transfer specs. Connectors remain responsible for translating these logical
    decisions into medium-specific source/destination specs.
    """

    loads: list[KVLoadPlan] = field(default_factory=list)
    offloads: list[KVOffloadPlan] = field(default_factory=list)
    prepared_load_ranges: list[KVBlockRange] = field(default_factory=list)
    prepared_offload_ranges: list[KVBlockRange] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.loads and not self.offloads


@dataclass
class OffloadPolicyState:
    """Connector-owned state needed by scheduler-level offload policies."""

    gpu_block_size: int
    offloaded_block_size: int
    block_size_factor: int


@dataclass
class LoadRequestInfo:
    """Information for one request that may need a KV load."""

    request: Request
    num_local_computed_tokens: int
    num_external_computed_tokens: int | None
    load_kv_async: bool
    allocated_blocks: KVCacheBlocks | None


@dataclass
class LoadDecisionContext:
    """Context for per-step load decisions after scheduling."""

    request_infos: list[LoadRequestInfo]
    offload_state: OffloadPolicyState | None
    kv_cache_manager: KVCacheManager
    token_budget: int
    max_num_running_reqs: int
    num_running_reqs: int
    scheduled_requests: list[Request] = field(default_factory=list)
    next_agent_ids_by_request_id: dict[str, list[str]] = field(default_factory=dict)
    lookup_block_hashes: Callable[[list[BlockHash]], int | None] | None = None
    now: float = field(default_factory=time.monotonic)


@dataclass
class LoadCandidateContext:
    """Context for deciding whether a waiting request can load KV now."""

    request: Request
    num_local_computed_tokens: int
    num_external_computed_tokens: int | None
    load_kv_async: bool
    offload_state: OffloadPolicyState | None
    blocks_being_loaded: set[BlockHash] | None


@dataclass
class OffloadDecisionContext:
    """Context for per-step offload decisions after scheduling."""

    scheduler_output: SchedulerOutput
    requests: dict[str, Request]
    offload_state: OffloadPolicyState | None
    kv_cache_manager: KVCacheManager
    running: list[Request]
    waiting: list[Request]
    preempted_req_ids: set[str]
    scheduled_requests: list[Request] = field(default_factory=list)
    now: float = field(default_factory=time.monotonic)


class OffloadPolicy(ABC):
    """Base class for scheduler-level KV load/offload planning."""

    def should_schedule_load_candidate(
        self, context: LoadCandidateContext
    ) -> bool:
        """Return whether a waiting request's KV load can be scheduled now."""
        return True

    def record_scheduled_load(self, request_info: LoadRequestInfo) -> None:
        """Record a per-waiting-request load accepted during scheduling."""
        return

    @abstractmethod
    def get_load_plan(self, context: LoadDecisionContext) -> KVLoadPlan:
        """Return logical blocks that should be loaded for this step."""
        raise NotImplementedError

    @abstractmethod
    def get_offload_plan(self, context: OffloadDecisionContext) -> KVOffloadPlan:
        """Return logical blocks that should be offloaded for this step."""
        raise NotImplementedError

    def update_after_connector_meta(self, transfer_plan: KVTransferPlan | None):
        """Commit policy state after a connector prepares transfer metadata."""
        return

    def request_finished(self, request: Request):
        """Clear policy state associated with a finished request."""
        return


class DefaultOffloadPolicy(OffloadPolicy):
    """Default policy matching the legacy offloading connector behavior."""

    def __init__(self):
        self._next_stored_block_idx: dict[str, int] = {}
        self._pending_load_ranges: list[KVBlockRange] = []
        self._pending_load_block_hashes: set[BlockHash] = set()
        self._offload_state: OffloadPolicyState | None = None

    @staticmethod
    def _get_block_hashes(
        req: Request,
        state: OffloadPolicyState,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> list[BlockHash]:
        return list(
            req.block_hashes[
                state.block_size_factor * start_idx
                + state.block_size_factor
                - 1 : (
                    state.block_size_factor * end_idx if end_idx is not None else None
                ) : state.block_size_factor
            ]
        )

    @staticmethod
    def _iter_scheduled_req_data(
        scheduler_output: SchedulerOutput,
    ):
        for req_data in scheduler_output.scheduled_new_reqs:
            yield req_data.req_id, req_data.block_ids, False

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for req_id, new_block_id_groups in zip(
            cached_reqs.req_ids, cached_reqs.new_block_ids
        ):
            yield (
                req_id,
                new_block_id_groups,
                req_id in cached_reqs.resumed_req_ids,
            )

    def should_schedule_load_candidate(
        self, context: LoadCandidateContext
    ) -> bool:
        state = context.offload_state
        self._offload_state = state
        if state is None:
            return True
        if (
            context.num_external_computed_tokens is None
            or context.num_external_computed_tokens == 0
            or not context.load_kv_async
        ):
            return True

        block_hashes = self._get_load_block_hashes(
            request=context.request,
            state=state,
            num_local_computed_tokens=context.num_local_computed_tokens,
            num_external_computed_tokens=context.num_external_computed_tokens,
        )
        blocks_being_loaded = context.blocks_being_loaded
        if blocks_being_loaded and any(
            block_hash in blocks_being_loaded for block_hash in block_hashes
        ):
            return False
        return not any(
            block_hash in self._pending_load_block_hashes
            for block_hash in block_hashes
        )

    def record_scheduled_load(self, request_info: LoadRequestInfo) -> None:
        state = self._offload_state
        if state is None:
            return
        block_range = self._make_load_range(request_info, state)
        if block_range is None:
            return
        self._pending_load_ranges.append(block_range)
        self._pending_load_block_hashes.update(block_range.block_hashes)

    def _get_load_block_hashes(
        self,
        request: Request,
        state: OffloadPolicyState,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> list[BlockHash]:
        num_blocks = request.num_tokens // state.offloaded_block_size
        assert len(request.block_hashes) // state.block_size_factor == num_blocks
        start_block_idx = num_local_computed_tokens // state.offloaded_block_size
        full_block_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        assert full_block_tokens % state.offloaded_block_size == 0
        num_hit_blocks = full_block_tokens // state.offloaded_block_size
        hits = num_hit_blocks - start_block_idx
        return self._get_block_hashes(
            request,
            state,
            start_idx=start_block_idx,
            end_idx=start_block_idx + hits,
        )

    def _make_load_range(
        self,
        request_info: LoadRequestInfo,
        state: OffloadPolicyState,
    ) -> KVBlockRange | None:
        request = request_info.request
        num_computed_tokens = request_info.num_local_computed_tokens
        num_external_tokens = request_info.num_external_computed_tokens
        if (
            num_external_tokens is None
            or num_external_tokens == 0
            or not request_info.load_kv_async
            or request_info.allocated_blocks is None
        ):
            return None

        assert state.block_size_factor == 1
        start_block_idx = num_computed_tokens // state.offloaded_block_size
        block_hashes_to_load = self._get_load_block_hashes(
            request=request,
            state=state,
            num_local_computed_tokens=num_computed_tokens,
            num_external_computed_tokens=num_external_tokens,
        )
        hits = len(block_hashes_to_load)
        block_ids = request_info.allocated_blocks.get_block_ids()[0]
        num_computed_gpu_blocks = sum(
            block.block_hash is not None
            for block in request_info.allocated_blocks.blocks[0]
        )
        dst_block_ids = block_ids[
            num_computed_gpu_blocks : num_computed_gpu_blocks + hits
        ]
        return KVBlockRange(
            req_id=request.request_id,
            start_block_idx=start_block_idx,
            num_blocks=hits,
            block_hashes=block_hashes_to_load,
            gpu_block_ids=dst_block_ids,
        )

    def get_load_plan(self, context: LoadDecisionContext) -> KVLoadPlan:
        self._offload_state = context.offload_state
        state = context.offload_state
        if state is None:
            return KVLoadPlan()

        block_ranges = list(self._pending_load_ranges)
        seen = {
            (block_range.req_id, block_range.start_block_idx, block_range.num_blocks)
            for block_range in block_ranges
        }
        for request_info in context.request_infos:
            block_range = self._make_load_range(request_info, state)
            if block_range is None:
                continue
            key = (
                block_range.req_id,
                block_range.start_block_idx,
                block_range.num_blocks,
            )
            if key in seen:
                continue
            seen.add(key)
            block_ranges.append(block_range)

        return KVLoadPlan(
            handled=True,
            num_external_tokens=0,
            load_async=True,
            block_ranges=block_ranges,
        )

    def get_offload_plan(self, context: OffloadDecisionContext) -> KVOffloadPlan:
        state = context.offload_state
        if state is None:
            return KVOffloadPlan()

        block_ranges: list[KVBlockRange] = []
        for req_id, _new_block_id_groups, _preempted in self._iter_scheduled_req_data(
            context.scheduler_output
        ):
            req = context.requests[req_id]
            new_tokens = context.scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = req.num_computed_tokens + new_tokens
            num_blocks = total_tokens // state.offloaded_block_size
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            new_block_hashes = self._get_block_hashes(
                req, state, start_idx=start_block_idx, end_idx=num_blocks
            )
            if not new_block_hashes:
                continue
            assert state.block_size_factor == 1
            gpu_block_ids = context.kv_cache_manager.get_block_ids(req_id)[0][
                start_block_idx:num_blocks
            ]

            block_ranges.append(
                KVBlockRange(
                    req_id=req_id,
                    start_block_idx=start_block_idx,
                    num_blocks=num_new_blocks,
                    block_hashes=new_block_hashes,
                    gpu_block_ids=gpu_block_ids,
                )
            )

        return KVOffloadPlan(block_ranges=block_ranges)

    def update_after_connector_meta(self, transfer_plan: KVTransferPlan | None):
        self._pending_load_ranges = []
        self._pending_load_block_hashes.clear()
        if transfer_plan is None:
            return

        for block_range in (
            transfer_plan.prepared_load_ranges
            + transfer_plan.prepared_offload_ranges
        ):
            next_block_idx = block_range.start_block_idx + block_range.num_blocks
            self._next_stored_block_idx[block_range.req_id] = max(
                self._next_stored_block_idx.get(block_range.req_id, 0),
                next_block_idx,
            )

    def request_finished(self, request: Request):
        self._next_stored_block_idx.pop(request.request_id, None)


class BaseAgentOffloadPolicy(DefaultOffloadPolicy):
    """Common base for agent-aware offload policies."""

    @staticmethod
    def _metadata(req: Request):
        return req.kv_cache_policy_metadata

    @staticmethod
    def _num_full_blocks_for_tokens(
        num_tokens: int,
        state: OffloadPolicyState,
    ) -> int:
        return num_tokens // state.offloaded_block_size

    @staticmethod
    def _get_gpu_block_ids(
        context: OffloadDecisionContext,
        req: Request,
        start_block_idx: int,
        end_block_idx: int,
    ) -> list[int]:
        try:
            return context.kv_cache_manager.get_block_ids(req.request_id)[0][
                start_block_idx:end_block_idx
            ]
        except (KeyError, IndexError, AssertionError):
            return []

    def _make_offload_range(
        self,
        context: OffloadDecisionContext,
        req: Request,
        start_block_idx: int,
        end_block_idx: int,
    ) -> KVBlockRange | None:
        state = context.offload_state
        if state is None or end_block_idx <= start_block_idx:
            return None
        block_hashes = self._get_block_hashes(
            req, state, start_idx=start_block_idx, end_idx=end_block_idx
        )
        if not block_hashes:
            return None
        gpu_block_ids = self._get_gpu_block_ids(
            context, req, start_block_idx, end_block_idx
        )
        if len(gpu_block_ids) != len(block_hashes):
            return None
        return KVBlockRange(
            req_id=req.request_id,
            start_block_idx=start_block_idx,
            num_blocks=end_block_idx - start_block_idx,
            block_hashes=block_hashes,
            gpu_block_ids=gpu_block_ids,
        )

    @staticmethod
    def _dedupe_ranges(block_ranges: list[KVBlockRange]) -> list[KVBlockRange]:
        seen: set[tuple[str, int, int]] = set()
        deduped: list[KVBlockRange] = []
        for block_range in block_ranges:
            key = (
                block_range.req_id,
                block_range.start_block_idx,
                block_range.num_blocks,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(block_range)
        return deduped


class KVFlowOffloadPolicy(BaseAgentOffloadPolicy):
    """KVFlow-specific offload policy hook.

    Stores only reusable fixed-prefix blocks when fixed-prefix metadata is
    present. Loading is driven by the current scheduled agents' successors:
    if a successor request has already been allocated destination GPU blocks,
    load that successor agent's indexed fixed-prompt KV hashes.
    Without workflow metadata, it falls back to the default connector policy.
    """

    def get_load_plan(self, context: LoadDecisionContext) -> KVLoadPlan:
        state = context.offload_state
        default_plan = super().get_load_plan(context)
        if state is None:
            return default_plan

        successor_order: dict[tuple[str, str], int] = {}
        for order, request in enumerate(context.scheduled_requests):
            metadata = self._metadata(request)
            workflow_id = metadata.workflow_key
            if workflow_id is None:
                continue
            next_agent_ids = context.next_agent_ids_by_request_id.get(
                request.request_id,
                metadata.next_agent_ids,
            )
            for next_agent_id in next_agent_ids:
                successor_order.setdefault((workflow_id, next_agent_id), order)

        if not successor_order:
            return default_plan

        logger.info(
            "awbench ---- KVFlow load planning for successor agents: %s",
            sorted(successor_order),
        )
        block_ranges: list[KVBlockRange] = []
        for workflow_id, agent_id in successor_order:
            fixed_block_hashes = context.kv_cache_manager.get_agent_fixed_block_hashes(
                workflow_id,
                agent_id,
            )
            if not fixed_block_hashes:
                continue

            missing_hashes = context.kv_cache_manager.get_missing_gpu_prefetch_hashes(
                workflow_id,
                agent_id,
                fixed_block_hashes,
            )
            if not missing_hashes:
                continue

            lookup_block_hashes = context.lookup_block_hashes
            if lookup_block_hashes is None:
                continue
            ready_blocks = lookup_block_hashes(missing_hashes)
            if ready_blocks is None or ready_blocks <= 0:
                continue
            block_hashes_to_load = missing_hashes[:ready_blocks]

            allocated_blocks = context.kv_cache_manager.allocate_prefetch_blocks(
                workflow_id=workflow_id,
                agent_id=agent_id,
                block_hashes=block_hashes_to_load,
            )
            if allocated_blocks is None:
                continue
            gpu_block_ids = allocated_blocks.get_block_ids()[0]
            prefetch_req_id = allocated_blocks.request_id or PREFETCH_POOL_REQ_ID
            block_ranges.append(
                KVBlockRange(
                    req_id=prefetch_req_id,
                    start_block_idx=0,
                    num_blocks=len(block_hashes_to_load),
                    block_hashes=block_hashes_to_load,
                    gpu_block_ids=gpu_block_ids,
                    workflow_id=workflow_id,
                    agent_id=agent_id,
                )
            )
            logger.info(
                "awbench ---- KVFlow planned fixed-prefix KV prefetch: "
                "request_id=%s workflow_id=%s agent_id=%s num_blocks=%d "
                "gpu_block_ids=%s",
                prefetch_req_id,
                workflow_id,
                agent_id,
                len(block_hashes_to_load),
                gpu_block_ids,
            )
            break

        def load_priority(block_range: KVBlockRange) -> tuple[int, int]:
            if block_range.workflow_id is not None and block_range.agent_id is not None:
                return (
                    successor_order.get(
                        (block_range.workflow_id, block_range.agent_id),
                        len(successor_order),
                    ),
                    block_range.start_block_idx,
                )
            request = next(
                (
                    info.request
                    for info in context.request_infos
                    if info.request.request_id == block_range.req_id
                ),
                None,
            )
            if request is None:
                return (len(successor_order), block_range.start_block_idx)
            metadata = self._metadata(request)
            return (
                successor_order.get(
                    (metadata.workflow_key or "", metadata.agent_id or ""),
                    len(successor_order),
                ),
                block_range.start_block_idx,
            )

        block_ranges.sort(key=load_priority)

        default_block_ranges = list(default_plan.block_ranges)
        seen = {
            (block_range.req_id, block_range.start_block_idx, block_range.num_blocks)
            for block_range in default_block_ranges
        }
        deduped_prefetch_ranges = []
        for block_range in block_ranges:
            key = (
                block_range.req_id,
                block_range.start_block_idx,
                block_range.num_blocks,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped_prefetch_ranges.append(block_range)

        return KVLoadPlan(
            handled=True,
            num_external_tokens=0,
            load_async=True,
            block_ranges=default_block_ranges + deduped_prefetch_ranges,
        )

    def get_offload_plan(self, context: OffloadDecisionContext) -> KVOffloadPlan:
        state = context.offload_state
        if state is None:
            return KVOffloadPlan()

        block_ranges: list[KVBlockRange] = []
        has_kvflow_metadata = False
        for req_id, _new_block_id_groups, _preempted in self._iter_scheduled_req_data(
            context.scheduler_output
        ):
            req = context.requests[req_id]
            metadata = self._metadata(req)
            if metadata.fixed_prefix_len is None:
                continue
            has_kvflow_metadata = True
            total_tokens = req.num_computed_tokens + (
                context.scheduler_output.num_scheduled_tokens[req_id]
            )
            total_blocks = self._num_full_blocks_for_tokens(total_tokens, state)
            fixed_blocks = self._num_full_blocks_for_tokens(
                metadata.fixed_prefix_len, state
            )
            end_block_idx = min(total_blocks, fixed_blocks)
            start_block_idx = min(
                self._next_stored_block_idx.get(req_id, 0), end_block_idx
            )
            block_range = self._make_offload_range(
                context, req, start_block_idx, end_block_idx
            )
            if block_range is not None:
                block_ranges.append(block_range)
                logger.info(
                    "awbench ---- KVFlow planned fixed-prefix KV offload: request_id=%s "
                    "workflow_id=%s agent_id=%s start_block_idx=%d num_blocks=%d",
                    req.request_id,
                    metadata.workflow_key,
                    metadata.agent_id,
                    block_range.start_block_idx,
                    block_range.num_blocks,
                )

        if not has_kvflow_metadata:
            return super().get_offload_plan(context)
        return KVOffloadPlan(block_ranges=block_ranges)


class TokencakeOffloadPolicy(BaseAgentOffloadPolicy):
    """Tokencake-specific offload policy hook.

    This implements Tokencake's request-finish time scheduler using only
    request metadata. A request becomes a Tokencake candidate when its metadata
    says that the next op is a tool and that a later request in the same
    session can reuse its KV cache. Candidate requests are split into three
    finish actions:

    * NORMAL: non-candidates follow normal vLLM lifetime management.
    * PIN: short or unprofitable tool stalls retain GPU KV blocks.
    * OFFLOAD: profitable stalls store KV blocks to CPU, then release GPU KV.
    """

    DEFAULT_CALL_DURATION_SECS = 1.0
    DEFAULT_TOKENS_PER_SEC = 2048.0
    DEFAULT_TRANSFER_SECS_PER_BLOCK = 0.00001
    DEFAULT_TRANSFER_SECS = 0.002
    DEFAULT_PIN_TIMEOUT_SECS = 30.0

    FINISH_NORMAL = "normal"
    FINISH_PIN = "pin"
    FINISH_OFFLOAD = "offload"

    @dataclass
    class ActiveStall:
        request_id: str
        tool_name: str
        start_time: float
        predicted_duration: float
        num_blocks: int
        action: str
        deadline: float
        load_requested: bool = False

        @property
        def predicted_finish_time(self) -> float:
            return self.start_time + self.predicted_duration

    @dataclass
    class SessionState:
        hash_list: list[BlockHash] = field(default_factory=list)
        agent_id: str | None = None
        workflow_id: str | None = None
        active_stall: "TokencakeOffloadPolicy.ActiveStall | None" = None
        stored_blocks: int = 0
        pinned_request_id: str | None = None
        pending_offload_request_id: str | None = None

    @dataclass
    class ProgramState:
        program_type: str | None = None
        sessions: dict[str, "TokencakeOffloadPolicy.SessionState"] = field(
            default_factory=dict
        )

    def __init__(self):
        super().__init__()
        self._programs: dict[str, TokencakeOffloadPolicy.ProgramState] = {}
        self._request_session_keys: dict[str, tuple[str, str]] = {}
        self._pending_finished_offloads: set[str] = set()

    def _session_key(self, req: Request) -> tuple[str, str] | None:
        metadata = self._metadata(req)
        if metadata.program_id is None or metadata.session_id is None:
            return None
        return (metadata.program_id, metadata.session_id)

    def _get_program_state(
        self, req: Request
    ) -> "TokencakeOffloadPolicy.ProgramState | None":
        metadata = self._metadata(req)
        if metadata.program_id is None:
            return None
        program = self._programs.setdefault(
            metadata.program_id,
            TokencakeOffloadPolicy.ProgramState(
                program_type=metadata.program_type
            ),
        )
        if metadata.program_type is not None:
            program.program_type = metadata.program_type
        return program

    def _get_session_state(
        self, req: Request
    ) -> "TokencakeOffloadPolicy.SessionState | None":
        key = self._session_key(req)
        if key is None:
            return None
        program = self._get_program_state(req)
        if program is None:
            return None
        self._request_session_keys[req.request_id] = key
        session = program.sessions.setdefault(
            key[1], TokencakeOffloadPolicy.SessionState()
        )
        metadata = self._metadata(req)
        session.agent_id = metadata.agent_id or session.agent_id
        session.workflow_id = metadata.workflow_key or session.workflow_id
        return session

    def _is_stall_candidate(self, req: Request) -> bool:
        metadata = self._metadata(req)
        return (
            metadata.next_op_type == "tool"
            and metadata.multi_turn_kv_reuse
            and metadata.program_id is not None
            and metadata.session_id is not None
        )

    def _tool_name(self, req: Request) -> str:
        metadata = self._metadata(req)
        return metadata.next_tool_type or "tool"

    def _predicted_duration(self, req: Request) -> float:
        duration = self._metadata(req).predicted_tool_time
        if duration is None or duration <= 0:
            duration = self.DEFAULT_CALL_DURATION_SECS
        return duration

    def _observe_request_hashes(self, req: Request) -> None:
        session = self._get_session_state(req)
        if session is None:
            return
        if len(req.block_hashes) > len(session.hash_list):
            session.hash_list = list(req.block_hashes)

    def _offloaded_hashes(
        self,
        hash_list: list[BlockHash],
        state: OffloadPolicyState,
    ) -> list[BlockHash]:
        if state.block_size_factor <= 1:
            return list(hash_list)
        return list(hash_list[state.block_size_factor - 1 :: state.block_size_factor])

    def _estimate_one_way_transfer_time(self, num_blocks: int) -> float:
        return self.DEFAULT_TRANSFER_SECS + (
            num_blocks * self.DEFAULT_TRANSFER_SECS_PER_BLOCK
        )

    def _estimate_round_trip_transfer_time(self, num_blocks: int) -> float:
        return 2 * self._estimate_one_way_transfer_time(num_blocks)

    def _has_best_fit_waiting_request(
        self,
        waiting: list[Request],
        predicted_duration: float,
        num_blocks: int,
    ) -> bool:
        transfer_time = self._estimate_round_trip_transfer_time(num_blocks)
        if predicted_duration <= transfer_time:
            return False
        token_capacity = (
            predicted_duration - transfer_time
        ) * self.DEFAULT_TOKENS_PER_SEC
        return any(req.num_tokens <= token_capacity for req in waiting)

    def on_request_finished(
        self,
        request: Request,
        offload_state: OffloadPolicyState | None,
        waiting: list[Request],
        now: float,
    ) -> str:
        """Classify a finished request into normal, pinned, or offloaded."""
        self._observe_request_hashes(request)
        if not self._is_stall_candidate(request):
            return self.FINISH_NORMAL

        session = self._get_session_state(request)
        if session is None:
            return self.FINISH_NORMAL

        predicted_duration = self._predicted_duration(request)
        pin_deadline = now + max(
            self.DEFAULT_PIN_TIMEOUT_SECS,
            2 * predicted_duration,
        )
        num_blocks = 0
        if offload_state is not None:
            num_blocks = len(session.hash_list) // offload_state.block_size_factor

        action = self.FINISH_PIN
        if (
            offload_state is not None
            and num_blocks > 0
            and self._has_best_fit_waiting_request(
                waiting, predicted_duration, num_blocks
            )
        ):
            action = self.FINISH_OFFLOAD

        session.active_stall = TokencakeOffloadPolicy.ActiveStall(
            request_id=request.request_id,
            tool_name=self._tool_name(request),
            start_time=now,
            predicted_duration=predicted_duration,
            num_blocks=num_blocks,
            action=action,
            deadline=pin_deadline,
        )

        metadata = self._metadata(request)
        if action == self.FINISH_OFFLOAD:
            session.pending_offload_request_id = request.request_id
            session.pinned_request_id = None
            self._pending_finished_offloads.add(request.request_id)
            logger.info(
                "awbench ---- Tokencake finish action=offload: request_id=%s "
                "program_id=%s session_id=%s tool=%s predicted_duration=%.6f "
                "num_blocks=%d waiting=%d",
                request.request_id,
                metadata.program_id,
                metadata.session_id,
                session.active_stall.tool_name,
                predicted_duration,
                num_blocks,
                len(waiting),
            )
            return self.FINISH_OFFLOAD

        session.pinned_request_id = request.request_id
        session.pending_offload_request_id = None
        logger.info(
            "awbench ---- Tokencake finish action=pin: request_id=%s "
            "program_id=%s session_id=%s tool=%s predicted_duration=%.6f "
            "num_blocks=%d waiting=%d",
            request.request_id,
            metadata.program_id,
            metadata.session_id,
            session.active_stall.tool_name,
            predicted_duration,
            num_blocks,
            len(waiting),
        )
        return self.FINISH_PIN

    def get_load_plan(self, context: LoadDecisionContext) -> KVLoadPlan:
        state = context.offload_state
        default_plan = super().get_load_plan(context)
        if state is None:
            return default_plan

        for request_info in context.request_infos:
            self._observe_request_hashes(request_info.request)
        for request in context.scheduled_requests:
            self._observe_request_hashes(request)

        block_ranges: list[KVBlockRange] = []
        if context.lookup_block_hashes is not None:
            for program_id, program in self._programs.items():
                for session_id, session in program.sessions.items():
                    active_stall = session.active_stall
                    if (
                        active_stall is None
                        or active_stall.action != self.FINISH_OFFLOAD
                        or active_stall.load_requested
                        or session.stored_blocks <= 0
                    ):
                        continue
                    upload_time = self._estimate_one_way_transfer_time(
                        session.stored_blocks
                    )
                    if context.now + upload_time < active_stall.predicted_finish_time:
                        continue

                    block_hashes = self._offloaded_hashes(session.hash_list, state)[
                        : session.stored_blocks
                    ]
                    ready_blocks = context.lookup_block_hashes(block_hashes)
                    if ready_blocks is None or ready_blocks <= 0:
                        continue
                    block_hashes_to_load = block_hashes[:ready_blocks]
                    workflow_id = session.workflow_id or program_id
                    agent_id = session.agent_id or session_id
                    allocated_blocks = context.kv_cache_manager.allocate_prefetch_blocks(
                        workflow_id=workflow_id,
                        agent_id=agent_id,
                        block_hashes=block_hashes_to_load,
                    )
                    if allocated_blocks is None:
                        continue
                    gpu_block_ids = allocated_blocks.get_block_ids()[0]
                    prefetch_req_id = allocated_blocks.request_id or PREFETCH_POOL_REQ_ID
                    block_ranges.append(
                        KVBlockRange(
                            req_id=prefetch_req_id,
                            start_block_idx=0,
                            num_blocks=len(block_hashes_to_load),
                            block_hashes=block_hashes_to_load,
                            gpu_block_ids=gpu_block_ids,
                            workflow_id=workflow_id,
                            agent_id=agent_id,
                        )
                    )
                    active_stall.load_requested = True
                    logger.info(
                        "awbench ---- Tokencake planned predictive KV load: "
                        "request_id=%s program_id=%s session_id=%s num_blocks=%d",
                        prefetch_req_id,
                        program_id,
                        session_id,
                        len(block_hashes_to_load),
                    )
                    break

        seen = {
            (block_range.req_id, block_range.start_block_idx, block_range.num_blocks)
            for block_range in block_ranges
        }
        for block_range in default_plan.block_ranges:
            key = (
                block_range.req_id,
                block_range.start_block_idx,
                block_range.num_blocks,
            )
            if key not in seen:
                block_ranges.append(block_range)

        return KVLoadPlan(
            handled=True,
            num_external_tokens=0,
            load_async=True,
            block_ranges=block_ranges,
        )

    def get_offload_plan(self, context: OffloadDecisionContext) -> KVOffloadPlan:
        state = context.offload_state
        if state is None:
            return KVOffloadPlan()

        block_ranges: list[KVBlockRange] = []
        for req_id in sorted(self._pending_finished_offloads):
            req = context.requests.get(req_id)
            if req is None:
                self._pending_finished_offloads.discard(req_id)
                continue
            session_key = self._request_session_keys.get(req_id)
            if session_key is None:
                continue
            program = self._programs.get(session_key[0])
            session = program.sessions.get(session_key[1]) if program else None
            if session is None or session.active_stall is None:
                continue
            total_blocks = min(
                session.active_stall.num_blocks,
                len(session.hash_list) // state.block_size_factor,
            )
            start_block_idx = min(session.stored_blocks, total_blocks)
            block_range = self._make_offload_range(
                context, req, start_block_idx, total_blocks
            )
            if block_range is not None:
                block_ranges.append(block_range)
                logger.info(
                    "awbench ---- Tokencake planned finished KV offload: "
                    "request_id=%s program_id=%s session_id=%s "
                    "start_block_idx=%d num_blocks=%d",
                    req.request_id,
                    session_key[0],
                    session_key[1],
                    block_range.start_block_idx,
                    block_range.num_blocks,
                )

        return KVOffloadPlan(block_ranges=self._dedupe_ranges(block_ranges))

    def update_after_connector_meta(self, transfer_plan: KVTransferPlan | None):
        super().update_after_connector_meta(transfer_plan)
        if transfer_plan is None:
            return

        planned_offload_keys = {
            (block_range.req_id, block_range.start_block_idx, block_range.num_blocks)
            for offload_plan in transfer_plan.offloads
            for block_range in offload_plan.block_ranges
        }
        prepared_offload_keys = {
            (block_range.req_id, block_range.start_block_idx, block_range.num_blocks)
            for block_range in transfer_plan.prepared_offload_ranges
        }

        for block_range in transfer_plan.prepared_offload_ranges:
            session_key = self._request_session_keys.get(block_range.req_id)
            if session_key is None:
                continue
            program = self._programs.get(session_key[0])
            if program is None:
                continue
            session = program.sessions.get(session_key[1])
            if session is None:
                continue
            session.stored_blocks = max(
                session.stored_blocks,
                block_range.start_block_idx + block_range.num_blocks,
            )
            if (
                session.active_stall is not None
                and session.stored_blocks >= session.active_stall.num_blocks
            ):
                self._pending_finished_offloads.discard(block_range.req_id)
                session.pending_offload_request_id = None
            logger.info(
                "awbench ---- Tokencake committed prepared KV offload: request_id=%s "
                "program_id=%s session_id=%s stored_blocks=%d",
                block_range.req_id,
                session_key[0],
                session_key[1],
                session.stored_blocks,
            )

        failed_offload_req_ids = {
            req_id
            for req_id, start_idx, num_blocks in planned_offload_keys
            if (req_id, start_idx, num_blocks) not in prepared_offload_keys
        }
        for req_id in failed_offload_req_ids:
            session_key = self._request_session_keys.get(req_id)
            if session_key is None:
                continue
            program = self._programs.get(session_key[0])
            session = program.sessions.get(session_key[1]) if program else None
            if session is None or session.active_stall is None:
                continue
            self._pending_finished_offloads.discard(req_id)
            session.pending_offload_request_id = None
            session.pinned_request_id = req_id
            session.active_stall.action = self.FINISH_PIN
            logger.info(
                "awbench ---- Tokencake falling back to GPU pin after failed "
                "offload preparation: request_id=%s program_id=%s session_id=%s",
                req_id,
                session_key[0],
                session_key[1],
            )

        for block_range in transfer_plan.prepared_load_ranges:
            session_key = self._request_session_keys.get(block_range.req_id)
            if session_key is None:
                continue
            program = self._programs.get(session_key[0])
            if program is None:
                continue
            session = program.sessions.get(session_key[1])
            if session is not None and session.active_stall is not None:
                session.active_stall.load_requested = True
                logger.info(
                    "awbench ---- Tokencake committed prepared KV load: request_id=%s "
                    "program_id=%s session_id=%s",
                    block_range.req_id,
                    session_key[0],
                    session_key[1],
                )

    def take_pinned_releases_for_scheduled(
        self, scheduled_requests: list[Request]
    ) -> list[str]:
        release_req_ids: list[str] = []
        for request in scheduled_requests:
            session_key = self._session_key(request)
            if session_key is None:
                continue
            program = self._programs.get(session_key[0])
            session = program.sessions.get(session_key[1]) if program else None
            if session is None or session.active_stall is None:
                continue
            if (
                session.active_stall.action == self.FINISH_OFFLOAD
                and session.active_stall.request_id != request.request_id
            ):
                logger.info(
                    "awbench ---- Tokencake observed offloaded session resume: "
                    "producer_request_id=%s consumer_request_id=%s "
                    "program_id=%s session_id=%s",
                    session.active_stall.request_id,
                    request.request_id,
                    session_key[0],
                    session_key[1],
                )
                session.active_stall = None
                session.stored_blocks = 0
                continue
            if session.pinned_request_id is None:
                continue
            if session.pinned_request_id == request.request_id:
                continue
            release_req_ids.append(session.pinned_request_id)
            logger.info(
                "awbench ---- Tokencake releasing pinned producer after "
                "consumer schedule: producer_request_id=%s consumer_request_id=%s "
                "program_id=%s session_id=%s",
                session.pinned_request_id,
                request.request_id,
                session_key[0],
                session_key[1],
            )
            session.pinned_request_id = None
            session.active_stall = None
        return release_req_ids

    def take_expired_pinned_requests(self, now: float) -> list[str]:
        release_req_ids: list[str] = []
        for program_id, program in list(self._programs.items()):
            for session_id, session in list(program.sessions.items()):
                active_stall = session.active_stall
                if (
                    active_stall is None
                    or session.pinned_request_id is None
                    or active_stall.deadline > now
                ):
                    continue
                release_req_ids.append(session.pinned_request_id)
                logger.info(
                    "awbench ---- Tokencake releasing expired GPU pin: "
                    "request_id=%s program_id=%s session_id=%s",
                    session.pinned_request_id,
                    program_id,
                    session_id,
                )
                session.pinned_request_id = None
                session.active_stall = None
        return release_req_ids

    def num_pending_finished_offloads(self) -> int:
        return len(self._pending_finished_offloads)

    def request_finished(self, request: Request):
        super().request_finished(request)
        session_key = self._request_session_keys.pop(request.request_id, None)
        if session_key is None:
            return
        program = self._programs.get(session_key[0])
        if program is None:
            return
        session = program.sessions.get(session_key[1])
        if session is not None:
            if session.pinned_request_id == request.request_id:
                session.pinned_request_id = None
            if session.pending_offload_request_id == request.request_id:
                session.pending_offload_request_id = None
                self._pending_finished_offloads.discard(request.request_id)
        if session is not None and (
            session.pinned_request_id is not None
            or session.pending_offload_request_id is not None
            or session.active_stall is not None
        ):
            return
        program.sessions.pop(session_key[1], None)
        if not program.sessions:
            self._programs.pop(session_key[0], None)
