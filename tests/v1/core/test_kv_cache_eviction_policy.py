# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.core.kv_cache_policy import (
    KVFlowPolicy,
    KVRequestPolicyMetadata,
    LRUPolicy,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


def _request(
    request_id: str, agent_id: str, scores: dict[str, int]
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        kv_cache_policy_metadata=KVRequestPolicyMetadata(
            workflow_id="workflow",
            program_id="program",
            agent_id=agent_id,
            agent_steps_to_execution=scores,
        ),
    )


def _allocated_blocks(
    num_blocks: int,
) -> tuple[list[KVCacheBlock], FreeKVCacheBlockQueue]:
    blocks = [KVCacheBlock(block_id) for block_id in range(num_blocks)]
    queue = FreeKVCacheBlockQueue(blocks)
    queue.popleft_n(num_blocks - 1)
    for block in blocks[:-1]:
        block.ref_cnt = 1
        block.block_hash = BlockHashWithGroupId(f"hash-{block.block_id}".encode())
    return blocks, queue


def test_lru_keeps_workflow_map_empty():
    blocks, queue = _allocated_blocks(2)
    workflow_map = {}
    policy = LRUPolicy(queue, workflow_map)

    blocks[0].ref_cnt = 0
    policy.on_blocks_freed([blocks[0]])

    assert workflow_map == {}
    assert queue.num_free_blocks == 2


def test_kvflow_releases_only_private_suffix_and_preserves_hash():
    blocks, queue = _allocated_blocks(5)
    workflow_map = {}
    policy = KVFlowPolicy(queue, workflow_map)

    req_a = _request("a", "a", {"workflow+program+a": 9})
    policy.on_request_finished(req_a, blocks[:3])
    for block in blocks[:3]:
        block.ref_cnt = 0
    policy.on_blocks_freed(blocks[:3])

    for block in blocks[:2]:
        block.ref_cnt = 1
    req_b = _request(
        "b", "b", {"workflow+program+a": 9, "workflow+program+b": 1}
    )
    policy.on_request_arrived(req_b)
    policy.on_request_finished(req_b, [*blocks[:2], blocks[3]])
    for block in [*blocks[:2], blocks[3]]:
        block.ref_cnt -= 1
    policy.on_blocks_freed([*blocks[:2], blocks[3]])

    policy.reclaim_blocks(queue.num_free_blocks + 1)

    entry_blocks = {
        entry.key.agent_id: [block.block_id for block in entry.blocks]
        for entry in workflow_map.values()
    }
    assert entry_blocks == {"a": [0, 1], "b": [0, 1, 3]}
    assert blocks[2].block_hash is not None
    assert blocks[2].block_id not in policy.block_id_to_workflow_keys
    assert queue.num_free_blocks == 2


def test_kvflow_snapshot_absence_marks_entry_evictable():
    blocks, queue = _allocated_blocks(3)
    workflow_map = {}
    policy = KVFlowPolicy(queue, workflow_map)
    req_a = _request("a", "a", {"workflow+program+a": 0})

    policy.on_request_finished(req_a, [blocks[0]])
    blocks[0].ref_cnt = 0
    policy.on_blocks_freed([blocks[0]])
    policy.on_request_arrived(_request("next", "b", {"workflow+program+b": 0}))

    entry = next(iter(workflow_map.values()))
    assert entry.score == policy.INF_SCORE


def test_kvflow_releases_hybrid_entry_as_a_unit():
    blocks, queue = _allocated_blocks(7)
    workflow_map = {}
    policy = KVFlowPolicy(queue, workflow_map)

    req_a = _request("a", "a", {"workflow+program+a": 9})
    policy.on_hybrid_request_finished(req_a, [[blocks[0], blocks[1]], [blocks[2]]])
    for block in blocks[:3]:
        block.ref_cnt = 0
    policy.on_blocks_freed(blocks[:3])

    blocks[0].ref_cnt = 1
    req_b = _request(
        "b", "b", {"workflow+program+a": 9, "workflow+program+b": 1}
    )
    policy.on_request_arrived(req_b)
    policy.on_hybrid_request_finished(req_b, [[blocks[0], blocks[3]], [blocks[4]]])
    for block in (blocks[0], blocks[3], blocks[4]):
        block.ref_cnt = 0
    policy.on_blocks_freed([blocks[0], blocks[3], blocks[4]])

    policy.reclaim_blocks(queue.num_free_blocks + 1)

    assert {entry.key.agent_id for entry in workflow_map.values()} == {"b"}
    assert policy.block_id_to_workflow_keys[blocks[0].block_id] == {
        next(iter(workflow_map))
    }
    assert blocks[1].block_id not in policy.block_id_to_workflow_keys
    assert blocks[2].block_id not in policy.block_id_to_workflow_keys
    assert queue.num_free_blocks == 3
