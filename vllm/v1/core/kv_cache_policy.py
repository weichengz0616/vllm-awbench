# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


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
            raise AssertionError("Request is missing workflow KV cache metadata")
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
    blocks_by_group: list[list[KVCacheBlock]] | None = None


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

    def on_hybrid_request_finished(
        self,
        request: Request | None,
        blocks_by_group: Sequence[Sequence[KVCacheBlock]],
    ) -> None:
        pass

    def on_block_touched(self, block: KVCacheBlock) -> None:
        self.free_block_queue.remove(block)

    def on_blocks_freed(self, blocks: Sequence[KVCacheBlock]) -> None:
        self.free_block_queue.append_n(list(blocks))

    def reclaim_blocks(self, num_blocks: int) -> None:
        pass

    def on_prefix_cache_reset(self) -> None:
        pass


class LRUPolicy(KVCacheEvictionPolicy):
    """The default free-list LRU behavior."""


class KVFlowPolicy(KVCacheEvictionPolicy):
    INF_SCORE = sys.maxsize

    def __init__(
        self,
        free_block_queue: FreeKVCacheBlockQueue,
        workflow_kv_cache_map: dict[WorkflowKVCacheKey, WorkflowKVCacheEntry],
    ) -> None:
        super().__init__(free_block_queue, workflow_kv_cache_map)
        self.block_id_to_workflow_keys: dict[int, set[WorkflowKVCacheKey]] = {}

    def on_request_arrived(self, request: Request | None) -> None:
        if request is None:
            return
        steps = request.kv_cache_policy_metadata.agent_steps_to_execution
        if steps is None:
            return
        for key, entry in self.workflow_kv_cache_map.items():
            entry.score = steps.get(key.as_agent_steps_key(), self.INF_SCORE)

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

    def on_hybrid_request_finished(
        self,
        request: Request | None,
        blocks_by_group: Sequence[Sequence[KVCacheBlock]],
    ) -> None:
        if request is None:
            return
        key = WorkflowKVCacheKey.from_request(request)
        if key is None:
            return
        cached_blocks_by_group = [
            [
                block
                for block in blocks
                if block.block_hash is not None and not block.is_null
            ]
            for blocks in blocks_by_group
        ]
        self._replace_entry_block_groups(key, request, cached_blocks_by_group)

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
        if self.free_block_queue.num_free_blocks >= num_blocks:
            return

        entries = sorted(
            self.workflow_kv_cache_map.values(),
            key=lambda entry: (
                -entry.score,
                entry.key.workflow_id,
                entry.key.program_id,
                entry.key.agent_id,
            ),
        )
        for entry in entries:
            if entry.blocks_by_group is not None:
                self._release_hybrid_entry(entry)
                if self.free_block_queue.num_free_blocks >= num_blocks:
                    return
                continue
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
            for block in self._entry_blocks(entry)
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

    def _replace_entry_block_groups(
        self,
        key: WorkflowKVCacheKey,
        request: Request,
        new_blocks_by_group: list[list[KVCacheBlock]],
    ) -> None:
        entry = self.workflow_kv_cache_map.get(key)
        if entry is None:
            entry = WorkflowKVCacheEntry(
                key=key,
                score=self._score_for_request(request),
                blocks_by_group=[],
            )
            self.workflow_kv_cache_map[key] = entry

        old_blocks = {block.block_id: block for block in self._entry_blocks(entry)}
        new_blocks = {
            block.block_id: block
            for blocks in new_blocks_by_group
            for block in blocks
        }

        for block_id in old_blocks.keys() - new_blocks.keys():
            self._detach_block(key, old_blocks[block_id])
        for block_id in new_blocks.keys() - old_blocks.keys():
            block = new_blocks[block_id]
            if self._is_in_free_queue(block):
                raise AssertionError("A map block must not also be free")
            self.block_id_to_workflow_keys.setdefault(block_id, set()).add(key)

        entry.blocks.clear()
        entry.blocks_by_group = new_blocks_by_group
        self._remove_empty_entry(entry)

    def _detach_tail_block(self, entry: WorkflowKVCacheEntry) -> KVCacheBlock:
        block = entry.blocks.pop()
        self._detach_block(entry.key, block)
        return block

    def _detach_block(self, key: WorkflowKVCacheKey, block: KVCacheBlock) -> None:
        keys = self.block_id_to_workflow_keys.get(block.block_id)
        if keys is None or key not in keys:
            raise AssertionError("KVFlow block ownership index is inconsistent")
        keys.remove(key)
        if keys:
            return
        self.block_id_to_workflow_keys.pop(block.block_id)
        if block.ref_cnt == 0 and not self._is_in_free_queue(block):
            self.free_block_queue.append(block)

    def _release_hybrid_entry(self, entry: WorkflowKVCacheEntry) -> None:
        seen_block_ids: set[int] = set()
        assert entry.blocks_by_group is not None
        for blocks in entry.blocks_by_group:
            for block in reversed(blocks):
                if block.block_id in seen_block_ids:
                    continue
                seen_block_ids.add(block.block_id)
                self._detach_block(entry.key, block)
        self.workflow_kv_cache_map.pop(entry.key, None)

    def _remove_empty_entry(self, entry: WorkflowKVCacheEntry) -> None:
        if not entry.blocks and not any(entry.blocks_by_group or ()):
            self.workflow_kv_cache_map.pop(entry.key, None)

    @staticmethod
    def _entry_blocks(entry: WorkflowKVCacheEntry) -> list[KVCacheBlock]:
        if entry.blocks_by_group is None:
            return entry.blocks
        return [block for blocks in entry.blocks_by_group for block in blocks]

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
