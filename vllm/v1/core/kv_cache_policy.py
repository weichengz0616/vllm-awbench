# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


# 创建请求时附带的 awbench_meta 数据
@dataclass
class KVRequestPolicyMetadata:
    workflow_id: str | None = None
    program_id: str | None = None
    agent_id: str | None = None
    op_id: str | None = None
    agent_steps_to_execution: dict[str, int] | None = None

    @classmethod
    def from_extra_args(
        cls, extra_args: dict[str, Any] | None
    ) -> KVRequestPolicyMetadata:
        raw = extra_args.get("awbench_meta", {}) if extra_args else {}
        return cls(
            workflow_id=raw.get("template_id") or raw.get("workflow_id"),
            program_id=raw.get("program_id"),
            agent_id=raw.get("agent_id"),
            op_id=raw.get("op_id"),
            agent_steps_to_execution=raw.get("agent_steps_to_execution"),
        )

    @property
    def workflow_key(self) -> str | None:
        return self.workflow_id



# workflow_kv_cache_map 的 key & value
@dataclass(frozen=True, slots=True)
class WorkflowKVCacheKey:
    workflow_id: str
    program_id: str
    agent_id: str

    def as_agent_steps_key(self) -> str:
        return "+".join((self.workflow_id, self.program_id, self.agent_id))

    @classmethod
    def from_request(cls, request: Request | None) -> WorkflowKVCacheKey | None:
        if request is None:
            return None
        metadata = request.kv_cache_policy_metadata
        if (
            metadata.workflow_id is None
            or metadata.program_id is None
            or metadata.agent_id is None
        ):
            assert False, "Request is missing workflow KV cache metadata"
        return cls(
            workflow_id=metadata.workflow_id,
            program_id=metadata.program_id,
            agent_id=metadata.agent_id,
        )


@dataclass
class WorkflowKVCacheEntry:
    key: WorkflowKVCacheKey
    blocks: list[KVCacheBlock] = field(default_factory=list)
    score: int = 0



class KVCacheEvictionPolicy:
    def __init__(
        self,
        free_block_queue: FreeKVCacheBlockQueue,
        workflow_kv_cache_map: dict[WorkflowKVCacheKey, WorkflowKVCacheEntry],
    ) -> None:
        self.free_block_queue = free_block_queue
        self.workflow_kv_cache_map = workflow_kv_cache_map

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def pop_free_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self.free_block_queue.popleft_n(num_blocks)

    def on_request_arrived(self, request: Request | None) -> None:
        pass

    def on_request_finished(
        self, request: Request | None, blocks: Sequence[KVCacheBlock]
    ) -> None:
        pass

    def on_block_touched(self, block: KVCacheBlock) -> None:
        self.free_block_queue.remove(block)

    def on_blocks_freed(self, blocks: Sequence[KVCacheBlock]) -> None:
        self.free_block_queue.append_n(list(blocks))

    # 目标是从 map 中移除 ref cnt = 0 的 block 到 free block list 中，直到 free block 数量达到 num_blocks
    # 这里就是 eviction
    def reclaim_blocks(self, num_blocks: int) -> None:
        pass

    def on_prefix_cache_reset(self) -> None:
        pass

class LRUPolicy(KVCacheEvictionPolicy):
    """The default free-list LRU behavior; its workflow map stays empty."""

class KVFlowPolicy(KVCacheEvictionPolicy):
    INF_SCORE = sys.maxsize

    def __init__(
        self,
        free_block_queue: FreeKVCacheBlockQueue,
        workflow_kv_cache_map: dict[WorkflowKVCacheKey, WorkflowKVCacheEntry],
    ) -> None:
        super().__init__(free_block_queue, workflow_kv_cache_map)

        # 反向索引
        self.block_id_to_workflow_keys: dict[int, set[WorkflowKVCacheKey]] = {}

    def on_request_arrived(self, request: Request | None) -> None:
        if request is None:
            return

        metadata = request.kv_cache_policy_metadata
        if metadata.agent_steps_to_execution is None:
            return
        for key, entry in self.workflow_kv_cache_map.items():
            entry.score = metadata.agent_steps_to_execution.get(
                key.as_agent_steps_key(), self.INF_SCORE
            )

    def on_request_finished(
        self, request: Request | None, blocks: Sequence[KVCacheBlock]
    ) -> None:
        if request is None:
            return
        key = WorkflowKVCacheKey.from_request(request)
        if key is None:
            return
        cached_blocks = [
            block
            for block in blocks
            if block.block_hash is not None and not block.is_null
        ]
        self._replace_entry_blocks(key, request, cached_blocks)

    def on_block_touched(self, block: KVCacheBlock) -> None:
        if self._is_in_free_queue(block):
            self.free_block_queue.remove(block)

    def on_blocks_freed(self, blocks: Sequence[KVCacheBlock]) -> None:
        self.free_block_queue.append_n(
            [
                block
                for block in blocks
                if block.block_id not in self.block_id_to_workflow_keys
                and not self._is_in_free_queue(block)
            ]
        )

    def reclaim_blocks(self, num_blocks: int) -> None:
        """Release retained suffix blocks until the free queue is sufficient."""
        if self.free_block_queue.num_free_blocks >= num_blocks:
            return

        # 给 entry 排序，而不是给 block 排序
        # 按照 score 降序排列
        entries = sorted(
            list(self.workflow_kv_cache_map.values()),
            key=lambda entry: (
                -entry.score,
                entry.key.workflow_id,
                entry.key.program_id,
                entry.key.agent_id,
            ),
        )
        for entry in entries:
            while entry.blocks:
                block = entry.blocks[-1]
                if block.ref_cnt > 0:
                    self._detach_tail_block(entry)
                    continue
                if self._has_other_key(block, entry.key):
                    break
                self._detach_tail_block(entry)
                if self.free_block_queue.num_free_blocks >= num_blocks:
                    return
            self._remove_empty_entry(entry)

    def on_prefix_cache_reset(self) -> None:
        retained_blocks = {
            block.block_id: block
            for entry in self.workflow_kv_cache_map.values()
            for block in entry.blocks
        }
        self.workflow_kv_cache_map.clear()
        self.block_id_to_workflow_keys.clear()
        self.free_block_queue.append_n(
            [
                block
                for block in retained_blocks.values()
                if block.ref_cnt == 0
                and not block.is_null
                and not self._is_in_free_queue(block)
            ]
        )

    def _replace_entry_blocks(
        self,
        key: WorkflowKVCacheKey,
        request: Request,
        new_blocks: list[KVCacheBlock],
    ) -> None:
        entry = self.workflow_kv_cache_map.get(key)
        if entry is None:
            entry = WorkflowKVCacheEntry(
                key=key, score=self._score_for_request(request)
            )
            self.workflow_kv_cache_map[key] = entry

        common_prefix_len = self._common_prefix_len(entry.blocks, new_blocks)
        while len(entry.blocks) > common_prefix_len:
            self._detach_tail_block(entry)
        for block in new_blocks[common_prefix_len:]:
            if key in self.block_id_to_workflow_keys.get(block.block_id, ()):
                raise AssertionError("KVFlow entry contains a duplicate block")
            if self._is_in_free_queue(block):
                raise AssertionError("A map block must not also be free")
            entry.blocks.append(block)
            self.block_id_to_workflow_keys.setdefault(block.block_id, set()).add(key)
        self._remove_empty_entry(entry)

    def _detach_tail_block(self, entry: WorkflowKVCacheEntry) -> KVCacheBlock:
        block = entry.blocks.pop()
        keys = self.block_id_to_workflow_keys.get(block.block_id)
        if keys is None or entry.key not in keys:
            raise AssertionError("KVFlow block ownership index is inconsistent")
        keys.remove(entry.key)
        if keys:
            return block
        self.block_id_to_workflow_keys.pop(block.block_id)
        if block.ref_cnt == 0 and not self._is_in_free_queue(block):
            self.free_block_queue.append(block)
        return block

    def _remove_empty_entry(self, entry: WorkflowKVCacheEntry) -> None:
        if not entry.blocks:
            self.workflow_kv_cache_map.pop(entry.key, None)

    def _has_other_key(self, block: KVCacheBlock, key: WorkflowKVCacheKey) -> bool:
        return any(
            other_key != key
            for other_key in self.block_id_to_workflow_keys.get(block.block_id, ())
        )

    @staticmethod
    def _common_prefix_len(
        old_blocks: Sequence[KVCacheBlock], new_blocks: Sequence[KVCacheBlock]
    ) -> int:
        prefix_len = 0
        for old_block, new_block in zip(old_blocks, new_blocks):
            if old_block.block_id != new_block.block_id:
                break
            prefix_len += 1
        return prefix_len

    @staticmethod
    def _is_in_free_queue(block: KVCacheBlock) -> bool:
        return block.prev_free_block is not None and block.next_free_block is not None

    def _score_for_request(self, request: Request) -> int:
        metadata = request.kv_cache_policy_metadata
        key = WorkflowKVCacheKey.from_request(request)
        if metadata.agent_steps_to_execution is None or key is None:
            return self.INF_SCORE
        return metadata.agent_steps_to_execution.get(
            key.as_agent_steps_key(), self.INF_SCORE
        )


def create_kv_cache_eviction_policy(
    policy_name: str,
    free_block_queue: FreeKVCacheBlockQueue,
    workflow_kv_cache_map: dict[WorkflowKVCacheKey, WorkflowKVCacheEntry],
) -> KVCacheEvictionPolicy:
    if policy_name == "lru":
        return LRUPolicy(free_block_queue, workflow_kv_cache_map)
    if policy_name == "kvflow":
        return KVFlowPolicy(free_block_queue, workflow_kv_cache_map)
    raise NotImplementedError(
        f"Agent KV eviction policy {policy_name!r} is not implemented yet."
    )
