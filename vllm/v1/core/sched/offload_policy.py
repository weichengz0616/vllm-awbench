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

    Event-driven implementation of Tokencake's time scheduler.

    ReAct programs are modeled as a single looping agent with many sessions.
    A session can span multiple requests and its KV hashes grow monotonically.
    Function-call events are carried by request metadata:
    ``function_event=call_start`` opens a stall window and may offload that
    session's idle KV; ``function_event=call_finish`` or a predicted finish
    deadline causes the matching session hashes to be loaded before the agent
    resumes.
    """

    DEFAULT_CALL_DURATION_SECS = 1.0
    DEFAULT_TOKENS_PER_SEC = 2048.0
    DEFAULT_TRANSFER_SECS_PER_BLOCK = 0.00001
    DEFAULT_TRANSFER_SECS = 0.002

    @dataclass
    class CallInfo:
        name: str
        duration: float

    @dataclass
    class ActiveCall:
        request_id: str
        call_name: str
        start_time: float
        predicted_duration: float
        num_blocks: int
        offload_requested: bool = False
        load_requested: bool = False
        finished: bool = False

        @property
        def predicted_finish_time(self) -> float:
            return self.start_time + self.predicted_duration

    @dataclass
    class SessionState:
        hash_list: list[BlockHash] = field(default_factory=list)
        active_call: "TokencakeOffloadPolicy.ActiveCall | None" = None
        stored_blocks: int = 0

    @dataclass
    class ProgramState:
        program_type: str | None = None
        sessions: dict[str, "TokencakeOffloadPolicy.SessionState"] = field(
            default_factory=dict
        )
        call_infos: dict[str, "TokencakeOffloadPolicy.CallInfo"] = field(
            default_factory=dict
        )

    def __init__(self):
        super().__init__()
        self._programs: dict[str, TokencakeOffloadPolicy.ProgramState] = {}
        self._request_session_keys: dict[str, tuple[str, str]] = {}

    def _session_key(self, req: Request) -> tuple[str, str] | None:
        metadata = self._metadata(req)
        if metadata.program_id is None:
            return None
        program_type = (metadata.program_type or "").lower()
        if program_type != "react":
            return None
        session_id = metadata.session_id or req.request_id
        return (metadata.program_id, session_id)

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
        return program.sessions.setdefault(
            key[1], TokencakeOffloadPolicy.SessionState()
        )

    def _call_name(self, req: Request) -> str:
        metadata = self._metadata(req)
        return metadata.call_name or "default"

    def _call_duration(self, req: Request) -> float:
        metadata = self._metadata(req)
        duration = (
            metadata.call_duration
            if metadata.call_duration is not None
            else metadata.predicted_tool_time
        )
        if duration is None or duration <= 0:
            duration = self.DEFAULT_CALL_DURATION_SECS
        return duration

    def _observe_request(self, req: Request, now: float) -> None:
        session = self._get_session_state(req)
        if session is None:
            return

        if len(req.block_hashes) > len(session.hash_list):
            session.hash_list = list(req.block_hashes)

        metadata = self._metadata(req)
        call_name = self._call_name(req)
        program = self._get_program_state(req)
        if program is not None:
            program.call_infos[call_name] = TokencakeOffloadPolicy.CallInfo(
                name=call_name,
                duration=self._call_duration(req),
            )

        if metadata.function_event == "call_start":
            session.active_call = TokencakeOffloadPolicy.ActiveCall(
                request_id=req.request_id,
                call_name=call_name,
                start_time=now,
                predicted_duration=self._call_duration(req),
                num_blocks=len(session.hash_list),
            )
            logger.info(
                "awbench ---- Tokencake observed call_start: request_id=%s program_id=%s "
                "session_id=%s call_name=%s predicted_duration=%.6f "
                "num_session_blocks=%d",
                req.request_id,
                metadata.program_id,
                metadata.session_id or req.request_id,
                call_name,
                session.active_call.predicted_duration,
                len(session.hash_list),
            )
        elif metadata.function_event == "call_finish":
            if session.active_call is None:
                session.active_call = TokencakeOffloadPolicy.ActiveCall(
                    request_id=req.request_id,
                    call_name=call_name,
                    start_time=now,
                    predicted_duration=0,
                    num_blocks=len(session.hash_list),
                )
            session.active_call.finished = True
            logger.info(
                "awbench ---- Tokencake observed call_finish: request_id=%s program_id=%s "
                "session_id=%s call_name=%s num_session_blocks=%d",
                req.request_id,
                metadata.program_id,
                metadata.session_id or req.request_id,
                call_name,
                len(session.hash_list),
            )

    def _observe_requests(self, requests: list[Request], now: float) -> None:
        seen: set[str] = set()
        for req in requests:
            if req.request_id in seen:
                continue
            seen.add(req.request_id)
            self._observe_request(req, now)

    def _estimate_transfer_time(self, num_blocks: int) -> float:
        return self.DEFAULT_TRANSFER_SECS + (
            num_blocks * self.DEFAULT_TRANSFER_SECS_PER_BLOCK
        )

    def _has_best_fit_waiting_request(
        self,
        context: OffloadDecisionContext,
        active_call: "TokencakeOffloadPolicy.ActiveCall",
    ) -> bool:
        transfer_time = self._estimate_transfer_time(active_call.num_blocks)
        if active_call.predicted_duration <= transfer_time:
            return False
        token_capacity = (
            active_call.predicted_duration - transfer_time
        ) * self.DEFAULT_TOKENS_PER_SEC
        return any(req.num_tokens <= token_capacity for req in context.waiting)

    def get_load_plan(self, context: LoadDecisionContext) -> KVLoadPlan:
        state = context.offload_state
        default_plan = super().get_load_plan(context)
        if state is None:
            return default_plan

        self._observe_requests(
            [info.request for info in context.request_infos]
            + context.scheduled_requests,
            context.now,
        )

        block_ranges: list[KVBlockRange] = []
        for request_info in context.request_infos:
            request = request_info.request
            session = self._get_session_state(request)
            if session is None or session.active_call is None:
                continue
            active_call = session.active_call
            upload_time = self._estimate_transfer_time(len(session.hash_list))
            should_load = active_call.finished or (
                context.now + upload_time >= active_call.predicted_finish_time
            )
            if not should_load or active_call.load_requested:
                continue
            if (
                request_info.allocated_blocks is None
                or not request_info.load_kv_async
            ):
                continue

            hash_list = session.hash_list
            if state.block_size_factor > 1:
                hash_list = hash_list[
                    state.block_size_factor - 1 :: state.block_size_factor
                ]
            if not hash_list:
                continue

            start_block_idx = min(
                request_info.num_local_computed_tokens
                // state.offloaded_block_size,
                len(hash_list),
            )
            if start_block_idx == len(hash_list):
                active_call.load_requested = True
                continue

            block_ids = request_info.allocated_blocks.get_block_ids()[0]
            num_computed_gpu_blocks = sum(
                block.block_hash is not None
                for block in request_info.allocated_blocks.blocks[0]
            )
            num_blocks = min(
                len(hash_list) - start_block_idx,
                len(block_ids) - num_computed_gpu_blocks,
            )
            if num_blocks <= 0:
                continue

            block_ranges.append(
                KVBlockRange(
                    req_id=request.request_id,
                    start_block_idx=start_block_idx,
                    num_blocks=num_blocks,
                    block_hashes=hash_list[
                        start_block_idx : start_block_idx + num_blocks
                    ],
                    gpu_block_ids=block_ids[
                        num_computed_gpu_blocks : num_computed_gpu_blocks
                        + num_blocks
                    ],
                )
            )
            active_call.load_requested = True
            metadata = self._metadata(request)
            logger.info(
                "awbench ---- Tokencake planned KV load: request_id=%s program_id=%s "
                "session_id=%s call_name=%s start_block_idx=%d num_blocks=%d "
                "finished=%s",
                request.request_id,
                metadata.program_id,
                metadata.session_id or request.request_id,
                active_call.call_name,
                start_block_idx,
                num_blocks,
                active_call.finished,
            )

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

        scheduled_event_reqs = [
            context.requests[req_id]
            for req_id, _, _ in self._iter_scheduled_req_data(
                context.scheduler_output
            )
            if req_id in context.requests
        ]
        self._observe_requests(
            context.running + context.waiting + context.scheduled_requests
            + scheduled_event_reqs,
            context.now,
        )

        block_ranges: list[KVBlockRange] = []
        candidates = context.running + context.scheduled_requests + scheduled_event_reqs
        seen_reqs: set[str] = set()
        for req in candidates:
            if req.request_id in seen_reqs:
                continue
            seen_reqs.add(req.request_id)
            session = self._get_session_state(req)
            if session is None or session.active_call is None:
                continue
            active_call = session.active_call
            if active_call.request_id != req.request_id:
                continue
            if active_call.offload_requested:
                continue
            if not self._has_best_fit_waiting_request(context, active_call):
                continue

            total_tokens = req.num_computed_tokens + (
                context.scheduler_output.num_scheduled_tokens.get(req.request_id, 0)
            )
            total_blocks = min(
                self._num_full_blocks_for_tokens(total_tokens, state),
                len(req.block_hashes) // state.block_size_factor,
                len(session.hash_list) // state.block_size_factor,
            )
            start_block_idx = min(session.stored_blocks, total_blocks)
            block_range = self._make_offload_range(
                context, req, start_block_idx, total_blocks
            )
            if block_range is not None:
                block_ranges.append(block_range)
                active_call.offload_requested = True
                metadata = self._metadata(req)
                logger.info(
                    "awbench ---- Tokencake planned KV offload: request_id=%s program_id=%s "
                    "session_id=%s call_name=%s start_block_idx=%d num_blocks=%d",
                    req.request_id,
                    metadata.program_id,
                    metadata.session_id or req.request_id,
                    active_call.call_name,
                    block_range.start_block_idx,
                    block_range.num_blocks,
                )

        return KVOffloadPlan(block_ranges=self._dedupe_ranges(block_ranges))

    def update_after_connector_meta(self, transfer_plan: KVTransferPlan | None):
        super().update_after_connector_meta(transfer_plan)
        if transfer_plan is None:
            return

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
            logger.info(
                "awbench ---- Tokencake committed prepared KV offload: request_id=%s "
                "program_id=%s session_id=%s stored_blocks=%d",
                block_range.req_id,
                session_key[0],
                session_key[1],
                session.stored_blocks,
            )

        for block_range in transfer_plan.prepared_load_ranges:
            session_key = self._request_session_keys.get(block_range.req_id)
            if session_key is None:
                continue
            program = self._programs.get(session_key[0])
            if program is None:
                continue
            session = program.sessions.get(session_key[1])
            if session is not None and session.active_call is not None:
                session.active_call.load_requested = True
                logger.info(
                    "awbench ---- Tokencake committed prepared KV load: request_id=%s "
                    "program_id=%s session_id=%s",
                    block_range.req_id,
                    session_key[0],
                    session_key[1],
                )

    def request_finished(self, request: Request):
        super().request_finished(request)
        metadata = self._metadata(request)
        session_key = self._request_session_keys.pop(request.request_id, None)
        if session_key is None:
            return
        program = self._programs.get(session_key[0])
        if program is None:
            return
        program.sessions.pop(session_key[1], None)
        if not program.sessions:
            self._programs.pop(session_key[0], None)
