# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


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


class OffloadPolicy(ABC):
    """Base class for scheduler-level KV load/offload planning."""

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

    def get_load_plan(self, context: LoadDecisionContext) -> KVLoadPlan:
        state = context.offload_state
        if state is None:
            return KVLoadPlan()

        block_ranges: list[KVBlockRange] = []
        for request_info in context.request_infos:
            request = request_info.request
            num_computed_tokens = request_info.num_local_computed_tokens
            num_external_tokens = request_info.num_external_computed_tokens
            if (
                num_external_tokens is None
                or num_external_tokens == 0
                or not request_info.load_kv_async
                or request_info.allocated_blocks is None
            ):
                continue

            assert state.block_size_factor == 1

            num_blocks = request.num_tokens // state.offloaded_block_size

            assert len(request.block_hashes) // state.block_size_factor == num_blocks
            start_block_idx = num_computed_tokens // state.offloaded_block_size
            full_block_tokens = num_computed_tokens + num_external_tokens
            assert full_block_tokens % state.offloaded_block_size == 0
            num_hit_blocks = full_block_tokens // state.offloaded_block_size
            hits = num_hit_blocks - start_block_idx

            block_hashes_to_load = self._get_block_hashes(
                request,
                state,
                start_idx=start_block_idx,
                end_idx=start_block_idx + hits,
            )
            block_ids = request_info.allocated_blocks.get_block_ids()[0]
            num_computed_gpu_blocks = sum(
                block.block_hash is not None
                for block in request_info.allocated_blocks.blocks[0]
            )
            dst_block_ids = block_ids[
                num_computed_gpu_blocks : num_computed_gpu_blocks + hits
            ]

            block_ranges.append(
                KVBlockRange(
                    req_id=request.request_id,
                    start_block_idx=start_block_idx,
                    num_blocks=hits,
                    block_hashes=block_hashes_to_load,
                    gpu_block_ids=dst_block_ids,
                )
            )

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
