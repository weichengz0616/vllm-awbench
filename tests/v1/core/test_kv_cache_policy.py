# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_policy import KVRequestPolicyMetadata, ttl_deadline
from vllm.v1.core.kv_cache_manager import (
    KVCacheManager,
    MAX_INFLIGHT_PREFETCH_BATCHES,
    PREFETCH_POOL_REQ_ID,
    is_prefetch_request_id,
)
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.sched.offload_policy import (
    DefaultOffloadPolicy,
    LoadCandidateContext,
    KVFlowOffloadPolicy,
    LoadDecisionContext,
    LoadRequestInfo,
    OffloadDecisionContext,
    OffloadPolicyState,
    TokencakeOffloadPolicy,
)


def test_block_pool_requires_string_eviction_policy():
    with pytest.raises(TypeError, match="must be an AgentKVEvictionPolicy string"):
        BlockPool(
            num_gpu_blocks=4,
            enable_caching=False,
            hash_block_size=16,
            eviction_policy=object(),  # type: ignore[arg-type]
        )


def test_block_pool_rejects_unknown_eviction_policy():
    with pytest.raises(ValueError, match="Unknown eviction_policy"):
        BlockPool(
            num_gpu_blocks=4,
            enable_caching=False,
            hash_block_size=16,
            eviction_policy="unknown",  # type: ignore[arg-type]
        )


class DummyRequest:
    def __init__(
        self,
        metadata: KVRequestPolicyMetadata,
        request_id: str = "r0",
        num_computed_tokens: int = 0,
        num_tokens: int | None = None,
        num_prompt_tokens: int | None = None,
        block_hashes: list[bytes] | None = None,
    ):
        self.kv_cache_policy_metadata = metadata
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.block_hashes = block_hashes or []
        self.num_tokens = num_tokens if num_tokens is not None else (
            len(self.block_hashes) * 2
        )
        self.num_prompt_tokens = (
            num_prompt_tokens
            if num_prompt_tokens is not None
            else self.num_tokens
        )


class DummyKVCacheManager:
    def __init__(
        self,
        block_ids: dict[str, list[int]],
        fixed_hashes: dict[tuple[str, str], list[bytes]] | None = None,
        live_hashes: set[bytes] | None = None,
    ):
        self.block_ids = block_ids
        self.fixed_hashes = fixed_hashes or {}
        self.live_hashes = live_hashes or set()
        self.prefetch_allocations: list[tuple[str, str, list[bytes]]] = []
        self.next_prefetch_id = 0

    def get_block_ids(self, request_id: str):
        return (self.block_ids[request_id],)

    def get_agent_fixed_block_hashes(self, workflow_id: str, agent_id: str):
        return self.fixed_hashes.get((workflow_id, agent_id), [])

    def get_missing_gpu_prefetch_hashes(
        self, workflow_id: str, agent_id: str, block_hashes: list[bytes]
    ):
        return [h for h in block_hashes if h not in self.live_hashes]

    def allocate_prefetch_blocks(
        self, workflow_id: str, agent_id: str, block_hashes: list[bytes]
    ):
        self.prefetch_allocations.append((workflow_id, agent_id, block_hashes))
        request_id = f"{PREFETCH_POOL_REQ_ID}:{self.next_prefetch_id}"
        self.next_prefetch_id += 1
        return DummyBlocks(
            list(range(100, 100 + len(block_hashes))),
            request_id=request_id,
        )


class DummyBlocks:
    def __init__(
        self,
        block_ids: list[int],
        computed_blocks: int = 0,
        request_id: str | None = None,
    ):
        self._block_ids = block_ids
        self.request_id = request_id
        self.blocks = (
            [
                SimpleNamespace(
                    block_hash=(b"cached" if idx < computed_blocks else None)
                )
                for idx, _ in enumerate(block_ids)
            ],
        )

    def get_block_ids(self):
        return (self._block_ids,)


def test_cachettl_eviction_policy_uses_default_free_queue():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=False,
        hash_block_size=16,
        eviction_policy="cachettl",
    )

    victim = pool.free_block_queue.popleft()

    assert victim.block_id == 1


def test_kvflow_eviction_prefers_dynamic_then_larger_step_distance():
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=False,
        hash_block_size=16,
        eviction_policy="kvflow",
    )
    for block_id in (1, 2, 3, 4):
        pool.blocks[block_id].block_hash = make_block_hash_with_group_id(
            f"block-{block_id}".encode(), 0
        )
    pool.block_metadata[1].prompt_part = "fixed"
    pool.block_metadata[1].steps_to_execution = 1
    pool.block_metadata[2].prompt_part = "fixed"
    pool.block_metadata[2].steps_to_execution = 5
    pool.block_metadata[3].prompt_part = "dynamic"
    pool.block_metadata[4].prompt_part = "fixed"
    pool.block_metadata[4].steps_to_execution = 0
    for block_id in (1, 2, 3, 4):
        pool.free_block_queue.reposition_if_free(pool.blocks[block_id])

    first = pool.free_block_queue.popleft()
    second = pool.free_block_queue.popleft()

    assert first.block_id == 3
    assert second.block_id == 2


def test_request_policy_metadata_binds_to_blocks():
    request = DummyRequest(
        KVRequestPolicyMetadata.from_extra_args(
            {
                "awbench_meta": {
                    "workflow_id": "wf0",
                    "program_id": "p0",
                    "agent_id": "a0",
                    "fixed_prefix_len": 2,
                    "agent_steps_to_execution": {"a0": 3},
                    "critical": True,
                }
            }
        )
    )
    pool = BlockPool(num_gpu_blocks=3, enable_caching=False, hash_block_size=2)
    block = pool.free_block_queue.popleft()

    pool.bind_block_metadata(block, request, block_index=0, block_size=2)
    metadata = pool.get_block_metadata(block)

    assert metadata.program_id == "p0"
    assert metadata.workflow_id == "wf0"
    assert metadata.agent_id == "a0"
    assert metadata.prompt_part == "fixed"
    assert metadata.steps_to_execution == 3
    assert metadata.critical is True


def test_request_policy_metadata_parses_workflow_id():
    metadata = KVRequestPolicyMetadata.from_extra_args(
        {
            "awbench_meta": {
                "workflow_id": "workflow-template",
                "program_id": "program-instance",
                "agent_id": "agent-a",
                "fixed_prefix_len": 4,
            }
        }
    )

    assert metadata.workflow_id == "workflow-template"
    assert metadata.program_id == "program-instance"
    assert metadata.agent_id == "agent-a"
    assert metadata.workflow_key == "workflow-template"


def test_request_policy_metadata_parses_cachettl_fields():
    metadata = KVRequestPolicyMetadata.from_extra_args(
        {
            "awbench_meta": {
                "program_id": "program-instance",
                "cachettl_should_pin": True,
                "cachettl_ttl_seconds": 1.5,
                "cachettl_is_last_step": False,
            }
        }
    )

    assert metadata.program_id == "program-instance"
    assert metadata.cachettl_should_pin is True
    assert metadata.cachettl_ttl_seconds == 1.5
    assert metadata.cachettl_is_last_step is False
    assert ttl_deadline(10.0, metadata) == 11.5


def test_request_policy_metadata_ignores_legacy_ttl_seconds():
    metadata = KVRequestPolicyMetadata.from_extra_args(
        {
            "awbench_meta": {
                "ttl_seconds": 99.0,
            }
        }
    )

    assert metadata.cachettl_ttl_seconds is None
    assert ttl_deadline(10.0, metadata) is None


def test_request_policy_metadata_accepts_external_template_id():
    metadata = KVRequestPolicyMetadata.from_extra_args(
        {
            "awbench_meta": {
                "template_id": "legacy-template",
                "program_id": "program-instance",
            }
        }
    )

    assert metadata.workflow_id == "legacy-template"
    assert metadata.workflow_key == "legacy-template"


def test_request_policy_metadata_accepts_agent_next_call_distance():
    metadata = KVRequestPolicyMetadata.from_extra_args(
        {
            "awbench_meta": {
                "workflow_id": "wf0",
                "agent_next_call_distance": {
                    "agent-a": 0,
                    "agent-b": 1,
                    "agent-c": 2,
                },
            }
        }
    )

    assert metadata.agent_steps_to_execution == {
        "agent-a": 0.0,
        "agent-b": 1.0,
        "agent-c": 2.0,
    }
    assert metadata.next_agent_ids == ["agent-b"]


def test_request_policy_metadata_explicit_next_agent_ids_win():
    metadata = KVRequestPolicyMetadata.from_extra_args(
        {
            "awbench_meta": {
                "workflow_id": "wf0",
                "agent_next_call_distance": {"agent-b": 1},
                "next_agent_ids": ["agent-c"],
            }
        }
    )

    assert metadata.agent_steps_to_execution == {"agent-b": 1.0}
    assert metadata.next_agent_ids == ["agent-c"]


def test_agent_fixed_hashes_are_keyed_by_workflow_agent():
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.hash_block_size = 2
    manager._agent_fixed_block_hashes = {}

    request = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="a0",
            fixed_prefix_len=4,
        ),
        block_hashes=[b"h0", b"h1", b"dynamic"],
    )

    manager._record_agent_fixed_block_hashes(request)

    assert manager.get_agent_fixed_block_hashes("wf0", "a0") == [b"h0", b"h1"]
    assert manager.get_agent_fixed_block_hashes("p0", "a0") == []


def test_agent_fixed_hashes_use_block_aligned_prefix_len():
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.hash_block_size = 2
    manager._agent_fixed_block_hashes = {}

    request = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="a0",
            fixed_prefix_len=3,
        ),
        block_hashes=[b"h0", b"partial-fixed"],
    )

    manager._record_agent_fixed_block_hashes(request)

    assert manager.get_agent_fixed_block_hashes("wf0", "a0") == [b"h0"]


def test_agent_live_blocks_survive_free_and_are_removed_on_eviction():
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=2)
    request = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="a0",
            fixed_prefix_len=2,
        ),
        block_hashes=[b"h0"],
    )
    block = pool.get_new_blocks(1)[0]
    pool.bind_block_metadata(block, request, block_index=0, block_size=2)

    pool.cache_full_blocks(
        request=request,
        blocks=[block],
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=2,
        kv_cache_group_id=0,
    )
    block_hash = make_block_hash_with_group_id(b"h0", 0)

    assert pool.get_agent_live_blocks("wf0", "a0")[block_hash] == [block]

    pool.free_blocks([block])
    assert pool.get_agent_live_blocks("wf0", "a0")[block_hash] == [block]

    fresh_block = pool.get_new_blocks(1)[0]
    assert fresh_block is not block

    evicted_block = pool.get_new_blocks(1)[0]

    assert evicted_block is block
    assert pool.get_agent_live_blocks("wf0", "a0") == {}


def test_cached_fixed_hit_uses_shared_agent_min_step():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=2,
        eviction_policy="kvflow",
    )
    request_a = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="agent_a",
            fixed_prefix_len=2,
            agent_steps_to_execution={"agent_a": 5},
        ),
        block_hashes=[b"shared"],
    )
    block = pool.get_new_blocks(1)[0]
    pool.bind_block_metadata(block, request_a, block_index=0, block_size=2)
    pool.cache_full_blocks(
        request=request_a,
        blocks=[block],
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=2,
        kv_cache_group_id=0,
    )

    request_b = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p1",
            agent_id="agent_b",
            fixed_prefix_len=2,
            agent_steps_to_execution={"agent_a": 5, "agent_b": 1},
        ),
        request_id="r1",
        block_hashes=[b"shared"],
    )
    pool.record_cached_block_hit_metadata(
        block=block,
        request=request_b,
        block_index=0,
        block_size=2,
    )

    block_hash = make_block_hash_with_group_id(b"shared", 0)
    metadata = pool.get_block_metadata(block)

    assert pool.get_agent_live_blocks("wf0", "agent_a")[block_hash] == [block]
    assert pool.get_agent_live_blocks("wf0", "agent_b")[block_hash] == [block]
    assert metadata.steps_to_execution == 1

    assert pool._maybe_evict_cached_block(block)
    assert pool.get_agent_live_blocks("wf0", "agent_a") == {}
    assert pool.get_agent_live_blocks("wf0", "agent_b") == {}


def test_kvflow_request_graph_overwrites_live_fixed_block_steps():
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=2,
        eviction_policy="kvflow",
    )
    request = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="a0",
            fixed_prefix_len=2,
        ),
        block_hashes=[b"h0"],
    )
    block = pool.get_new_blocks(1)[0]
    pool.bind_block_metadata(block, request, block_index=0, block_size=2)
    pool.cache_full_blocks(
        request=request,
        blocks=[block],
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=2,
        kv_cache_group_id=0,
    )

    pool.on_request_metadata(
        DummyRequest(
            KVRequestPolicyMetadata(
                workflow_id="wf0",
                program_id="p0",
                agent_steps_to_execution={"a0": 4},
            )
        )
    )
    pool.on_request_metadata(
        DummyRequest(
            KVRequestPolicyMetadata(
                workflow_id="wf0",
                program_id="p1",
                agent_steps_to_execution={"a0": 1},
            )
        )
    )

    metadata = pool.get_block_metadata(block)
    assert metadata.steps_to_execution == 1

    pool.on_request_metadata(
        DummyRequest(
            KVRequestPolicyMetadata(
                workflow_id="wf0",
                program_id="p2",
                agent_steps_to_execution={"a0": 4},
            )
        )
    )

    assert metadata.steps_to_execution == 4



def test_kvflow_offload_policy_stores_fixed_prefix_only():
    policy = KVFlowOffloadPolicy()
    req = DummyRequest(
        KVRequestPolicyMetadata(
            program_id="p0",
            agent_id="a0",
            fixed_prefix_len=4,
            agent_steps_to_execution={"a0": 1},
        ),
        request_id="r0",
        num_computed_tokens=0,
        block_hashes=[b"h0", b"h1", b"h2"],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r0", block_ids=([1, 2, 3],))],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={"r0": 6},
    )
    context = OffloadDecisionContext(
        scheduler_output=scheduler_output,
        requests={"r0": req},
        offload_state=OffloadPolicyState(
            gpu_block_size=2,
            offloaded_block_size=2,
            block_size_factor=1,
        ),
        kv_cache_manager=DummyKVCacheManager({"r0": [1, 2, 3]}),
        running=[],
        waiting=[],
        preempted_req_ids=set(),
    )

    plan = policy.get_offload_plan(context)

    assert len(plan.block_ranges) == 1
    assert plan.block_ranges[0].num_blocks == 2
    assert plan.block_ranges[0].block_hashes == [b"h0", b"h1"]
    assert plan.block_ranges[0].gpu_block_ids == [1, 2]


def test_default_offload_policy_rejects_loads_already_being_loaded():
    policy = DefaultOffloadPolicy()
    state = OffloadPolicyState(
        gpu_block_size=2,
        offloaded_block_size=2,
        block_size_factor=1,
    )
    req = DummyRequest(
        KVRequestPolicyMetadata(),
        request_id="r0",
        block_hashes=[b"h0", b"h1"],
    )

    allowed = policy.should_schedule_load_candidate(
        LoadCandidateContext(
            request=req,
            num_local_computed_tokens=0,
            num_external_computed_tokens=2,
            load_kv_async=True,
            offload_state=state,
            blocks_being_loaded={b"h0"},
        )
    )

    assert allowed is False


def test_default_offload_policy_tracks_step_local_planned_loads():
    policy = DefaultOffloadPolicy()
    state = OffloadPolicyState(
        gpu_block_size=2,
        offloaded_block_size=2,
        block_size_factor=1,
    )
    first_req = DummyRequest(
        KVRequestPolicyMetadata(),
        request_id="r0",
        block_hashes=[b"h0", b"h1"],
    )
    second_req = DummyRequest(
        KVRequestPolicyMetadata(),
        request_id="r1",
        block_hashes=[b"h0", b"h2"],
    )
    assert policy.should_schedule_load_candidate(
        LoadCandidateContext(
            request=first_req,
            num_local_computed_tokens=0,
            num_external_computed_tokens=2,
            load_kv_async=True,
            offload_state=state,
            blocks_being_loaded=set(),
        )
    )
    policy.record_scheduled_load(
        LoadRequestInfo(
            request=first_req,
            num_local_computed_tokens=0,
            num_external_computed_tokens=2,
            load_kv_async=True,
            allocated_blocks=DummyBlocks([11, 12]),
        )
    )

    allowed = policy.should_schedule_load_candidate(
        LoadCandidateContext(
            request=second_req,
            num_local_computed_tokens=0,
            num_external_computed_tokens=2,
            load_kv_async=True,
            offload_state=state,
            blocks_being_loaded=set(),
        )
    )
    plan = policy.get_load_plan(
        LoadDecisionContext(
            request_infos=[],
            offload_state=state,
            kv_cache_manager=DummyKVCacheManager({}),
            token_budget=0,
            max_num_running_reqs=1,
            num_running_reqs=1,
        )
    )

    assert allowed is False
    assert len(plan.block_ranges) == 1
    assert plan.block_ranges[0].req_id == "r0"
    assert plan.block_ranges[0].block_hashes == [b"h0"]
    assert plan.block_ranges[0].gpu_block_ids == [11]


def test_kvflow_load_policy_prefetches_next_agent_fixed_prompt_hashes():
    policy = KVFlowOffloadPolicy()
    current_req = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="agent_a",
            next_agent_ids=["agent_b"],
        ),
        request_id="current",
    )
    kv_cache_manager = DummyKVCacheManager(
        {},
        fixed_hashes={("wf0", "agent_b"): [b"fixed0", b"fixed1"]},
    )
    context = LoadDecisionContext(
        request_infos=[],
        offload_state=OffloadPolicyState(
            gpu_block_size=2,
            offloaded_block_size=2,
            block_size_factor=1,
        ),
        kv_cache_manager=kv_cache_manager,
        token_budget=0,
        max_num_running_reqs=1,
        num_running_reqs=1,
        scheduled_requests=[current_req],
        lookup_block_hashes=lambda block_hashes: len(block_hashes),
    )

    plan = policy.get_load_plan(context)

    assert len(plan.block_ranges) == 1
    assert is_prefetch_request_id(plan.block_ranges[0].req_id)
    assert plan.block_ranges[0].req_id != PREFETCH_POOL_REQ_ID
    assert plan.block_ranges[0].block_hashes == [b"fixed0", b"fixed1"]
    assert plan.block_ranges[0].gpu_block_ids == [100, 101]
    assert kv_cache_manager.prefetch_allocations == [
        ("wf0", "agent_b", [b"fixed0", b"fixed1"])
    ]


def test_kvflow_load_policy_prioritizes_external_hit_load_before_prefetch():
    policy = KVFlowOffloadPolicy()
    state = OffloadPolicyState(
        gpu_block_size=2,
        offloaded_block_size=2,
        block_size_factor=1,
    )
    external_hit_req = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="agent_x",
        ),
        request_id="external",
        block_hashes=[b"external0", b"external1"],
    )
    current_req = DummyRequest(
        KVRequestPolicyMetadata(
            workflow_id="wf0",
            program_id="p0",
            agent_id="agent_a",
            next_agent_ids=["agent_b"],
        ),
        request_id="current",
    )
    request_info = LoadRequestInfo(
        request=external_hit_req,
        num_local_computed_tokens=0,
        num_external_computed_tokens=2,
        load_kv_async=True,
        allocated_blocks=DummyBlocks([11, 12]),
    )
    policy.should_schedule_load_candidate(
        LoadCandidateContext(
            request=external_hit_req,
            num_local_computed_tokens=0,
            num_external_computed_tokens=2,
            load_kv_async=True,
            offload_state=state,
            blocks_being_loaded=set(),
        )
    )
    policy.record_scheduled_load(request_info)
    kv_cache_manager = DummyKVCacheManager(
        {},
        fixed_hashes={("wf0", "agent_b"): [b"fixed0"]},
    )

    plan = policy.get_load_plan(
        LoadDecisionContext(
            request_infos=[request_info],
            offload_state=state,
            kv_cache_manager=kv_cache_manager,
            token_budget=0,
            max_num_running_reqs=1,
            num_running_reqs=1,
            scheduled_requests=[current_req],
            lookup_block_hashes=lambda block_hashes: len(block_hashes),
        )
    )

    assert len(plan.block_ranges) == 2
    assert plan.block_ranges[0].req_id == "external"
    assert plan.block_ranges[0].block_hashes == [b"external0"]
    assert is_prefetch_request_id(plan.block_ranges[1].req_id)
    assert plan.block_ranges[1].block_hashes == [b"fixed0"]



def test_prefetch_manager_uses_unique_batches_and_limits_inflight():
    class DummyBlockPool:
        def __init__(self):
            self.next_block_id = 1
            self.bound = []

        def get_num_free_blocks(self):
            return 100

        def get_new_blocks(self, num_blocks):
            blocks = [
                SimpleNamespace(block_id=self.next_block_id + i, is_null=False)
                for i in range(num_blocks)
            ]
            self.next_block_id += num_blocks
            return blocks

        def bind_prefetch_block_metadata(
            self, block, workflow_id, agent_id, status="loading"
        ):
            self.bound.append((block.block_id, workflow_id, agent_id, status))

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.num_kv_cache_groups = 1
    manager.empty_kv_cache_blocks = DummyBlocks([])
    manager.block_pool = DummyBlockPool()
    manager.coordinator = SimpleNamespace(
        single_type_managers=[SimpleNamespace(req_to_blocks={})]
    )
    manager._prefetch_batches = {}
    manager._prefetch_loading = {}
    manager._next_prefetch_batch_id = 0

    request_ids = []
    for i in range(MAX_INFLIGHT_PREFETCH_BATCHES):
        blocks = manager.allocate_prefetch_blocks(
            workflow_id="wf0",
            agent_id=f"agent_{i}",
            block_hashes=[f"h{i}".encode()],
        )
        assert blocks is not None
        assert blocks.request_id is not None
        request_ids.append(blocks.request_id)

    assert len(set(request_ids)) == MAX_INFLIGHT_PREFETCH_BATCHES
    assert all(is_prefetch_request_id(req_id) for req_id in request_ids)
    assert set(manager._prefetch_batches) == set(request_ids)
    assert len(manager._prefetch_loading) == MAX_INFLIGHT_PREFETCH_BATCHES
    assert manager.allocate_prefetch_blocks(
        workflow_id="wf0",
        agent_id="overflow",
        block_hashes=[b"overflow"],
    ) is None



def test_tokencake_offload_policy_offloads_profitable_tool_stall():
    policy = TokencakeOffloadPolicy()
    req = DummyRequest(
        KVRequestPolicyMetadata(
            program_id="p0",
            program_type="react",
            agent_id="a0",
            session_id="s0",
            next_op_type="tool",
            next_tool_type="search",
            predicted_tool_time=10.0,
            multi_turn_kv_reuse=True,
        ),
        request_id="r0",
        num_computed_tokens=4,
        block_hashes=[b"h0", b"h1"],
    )
    state = OffloadPolicyState(
        gpu_block_size=2,
        offloaded_block_size=2,
        block_size_factor=1,
    )
    action = policy.on_request_finished(
        request=req,
        offload_state=state,
        waiting=[DummyRequest(KVRequestPolicyMetadata(), request_id="waiting")],
        now=0.0,
    )
    assert action == TokencakeOffloadPolicy.FINISH_OFFLOAD

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={},
    )
    context = OffloadDecisionContext(
        scheduler_output=scheduler_output,
        requests={"r0": req},
        offload_state=state,
        kv_cache_manager=DummyKVCacheManager({"r0": [4, 5]}),
        running=[],
        waiting=[DummyRequest(KVRequestPolicyMetadata(), request_id="waiting")],
        preempted_req_ids=set(),
    )

    plan = policy.get_offload_plan(context)

    assert len(plan.block_ranges) == 1
    assert plan.block_ranges[0].num_blocks == 2
    assert plan.block_ranges[0].block_hashes == [b"h0", b"h1"]
    assert plan.block_ranges[0].gpu_block_ids == [4, 5]


def test_tokencake_offload_policy_pins_short_tool_stall():
    policy = TokencakeOffloadPolicy()
    req = DummyRequest(
        KVRequestPolicyMetadata(
            program_id="p0",
            program_type="react",
            agent_id="a0",
            session_id="s0",
            next_op_type="tool",
            next_tool_type="search",
            predicted_tool_time=0.001,
            multi_turn_kv_reuse=True,
        ),
        request_id="r0",
        num_computed_tokens=4,
        block_hashes=[b"h0", b"h1"],
    )

    action = policy.on_request_finished(
        request=req,
        offload_state=OffloadPolicyState(
            gpu_block_size=2,
            offloaded_block_size=2,
            block_size_factor=1,
        ),
        waiting=[DummyRequest(KVRequestPolicyMetadata(), request_id="waiting")],
        now=0.0,
    )

    assert action == TokencakeOffloadPolicy.FINISH_PIN
    assert policy.num_pending_finished_offloads() == 0
    consumer = DummyRequest(
        KVRequestPolicyMetadata(
            program_id="p0",
            program_type="react",
            agent_id="a0",
            session_id="s0",
        ),
        request_id="r1",
    )
    assert policy.take_pinned_releases_for_scheduled([consumer]) == ["r0"]


def test_tokencake_load_policy_predictively_prefetches_offloaded_session():
    policy = TokencakeOffloadPolicy()
    start_req = DummyRequest(
        KVRequestPolicyMetadata(
            program_id="p0",
            workflow_id="wf0",
            program_type="react",
            agent_id="a0",
            session_id="s0",
            next_op_type="tool",
            next_tool_type="search",
            predicted_tool_time=1.0,
            multi_turn_kv_reuse=True,
        ),
        request_id="r0",
        num_computed_tokens=4,
        block_hashes=[b"h0", b"h1"],
    )
    state = OffloadPolicyState(
        gpu_block_size=2,
        offloaded_block_size=2,
        block_size_factor=1,
    )
    policy.on_request_finished(
        request=start_req,
        offload_state=state,
        waiting=[DummyRequest(KVRequestPolicyMetadata(), request_id="waiting")],
        now=0.0,
    )
    offload_plan = policy.get_offload_plan(
        OffloadDecisionContext(
            scheduler_output=SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=[], new_block_ids=[], resumed_req_ids=set()
                ),
                num_scheduled_tokens={},
            ),
            requests={"r0": start_req},
            offload_state=state,
            kv_cache_manager=DummyKVCacheManager({"r0": [4, 5]}),
            running=[],
            waiting=[DummyRequest(KVRequestPolicyMetadata(), request_id="waiting")],
            preempted_req_ids=set(),
            now=0.0,
        )
    )
    policy.update_after_connector_meta(
        SimpleNamespace(
            loads=[],
            offloads=[offload_plan],
            prepared_load_ranges=[],
            prepared_offload_ranges=list(offload_plan.block_ranges),
        )
    )
    manager = DummyKVCacheManager({})

    plan = policy.get_load_plan(
        LoadDecisionContext(
            request_infos=[],
            offload_state=state,
            kv_cache_manager=manager,
            token_budget=0,
            max_num_running_reqs=1,
            num_running_reqs=0,
            now=1.0,
            lookup_block_hashes=lambda hashes: len(hashes),
        )
    )

    assert len(plan.block_ranges) == 1
    assert plan.block_ranges[0].req_id.startswith(PREFETCH_POOL_REQ_ID)
    assert plan.block_ranges[0].block_hashes == [b"h0", b"h1"]
    assert plan.block_ranges[0].gpu_block_ids == [100, 101]
    assert manager.prefetch_allocations == [("wf0", "a0", [b"h0", b"h1"])]
