# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    PLAS = "plas"    # sequential programs — additive service accumulation
    ATLAS = "atlas"  # DAG programs — critical-path (max) service tracking


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


class MLFQRequestQueue(RequestQueue):
    """Multi-Level Feedback Queue for PLAS scheduling (Autellix Algorithm 1).

    Organises requests into K FCFS sub-queues indexed 0 (highest priority)
    to K-1 (lowest priority).  pop_request() always returns from the
    highest-priority non-empty sub-queue.

    The scheduler drives level changes: new arrivals start at level 0;
    assign_level() moves a request when the scheduler demotes or promotes it.
    """

    def __init__(self, num_levels: int = 4) -> None:
        self.num_levels = num_levels
        self._queues: list[deque[Request]] = [deque() for _ in range(num_levels)]
        self._req_level: dict[str, int] = {}

    def add_request(self, request: Request) -> None:
        self._queues[0].append(request)
        self._req_level[request.request_id] = 0

    def pop_request(self) -> Request:
        for q in self._queues:
            if q:
                req = q.popleft()
                self._req_level.pop(req.request_id, None)
                return req
        raise IndexError("pop from empty MLFQRequestQueue")

    def peek_request(self) -> Request:
        for q in self._queues:
            if q:
                return q[0]
        raise IndexError("peek from empty MLFQRequestQueue")

    def prepend_request(self, request: Request) -> None:
        level = self._req_level.get(request.request_id, 0)
        self._queues[level].appendleft(request)
        self._req_level[request.request_id] = level

    def prepend_requests(self, requests: "RequestQueue") -> None:
        for req in reversed(list(requests)):
            self.prepend_request(req)

    def remove_request(self, request: Request) -> None:
        level = self._req_level.pop(request.request_id, None)
        if level is not None:
            self._queues[level].remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = {r.request_id for r in requests}
        for req_id in to_remove:
            level = self._req_level.pop(req_id, None)
            if level is not None:
                self._queues[level] = deque(
                    r for r in self._queues[level] if r.request_id != req_id
                )

    def assign_level(self, request: Request, level: int) -> None:
        """Move a waiting request to a different MLFQ level."""
        cur = self._req_level.get(request.request_id)
        if cur is None:
            return
        self._queues[cur].remove(request)
        target = min(max(level, 0), self.num_levels - 1)
        self._queues[target].append(request)
        self._req_level[request.request_id] = target

    def __bool__(self) -> bool:
        return any(q for q in self._queues)

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues)

    def __iter__(self) -> Iterator[Request]:
        for q in self._queues:
            yield from q

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return reversed(list(self))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy in (SchedulingPolicy.PLAS, SchedulingPolicy.ATLAS):
        return MLFQRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
