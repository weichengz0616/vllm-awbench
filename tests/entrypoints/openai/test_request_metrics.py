# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    UsageInfo,
    build_vllm_request_metrics,
)
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.metrics.stats import IterationStats, PrefillStats, RequestStateStats


def make_request_stats() -> RequestStateStats:
    return RequestStateStats(
        arrival_time=100.0,
        queued_ts=10.0,
        scheduled_ts=10.25,
        first_token_ts=10.75,
        last_token_ts=12.75,
        num_generation_tokens=5,
        first_token_latency=0.75,
        num_prompt_tokens=20,
        num_cached_tokens=8,
        num_external_computed_tokens=3,
        num_preemptions=2,
        is_corrupted=True,
        finished_time=103.0,
    )


def test_build_vllm_request_metrics() -> None:
    metrics = build_vllm_request_metrics(
        "chatcmpl-request-1",
        [make_request_stats()],
        requested_max_tokens=16,
        finish_reason="stop",
        stop_reason="done",
    )

    assert metrics is not None
    assert metrics.request_id == "chatcmpl-request-1"
    assert metrics.num_engine_requests == 1
    assert metrics.queue_time_seconds == pytest.approx(0.25)
    assert metrics.time_to_first_token_seconds == pytest.approx(0.75)
    assert metrics.prefill_time_seconds == pytest.approx(0.5)
    assert metrics.decode_time_seconds == pytest.approx(2.0)
    assert metrics.inference_time_seconds == pytest.approx(2.5)
    assert metrics.e2e_latency_seconds == pytest.approx(3.0)
    assert metrics.mean_time_per_output_token_seconds == pytest.approx(0.5)
    assert metrics.prompt_tokens == 20
    assert metrics.completion_tokens == 5
    assert metrics.total_tokens == 25
    assert metrics.cached_prompt_tokens == 8
    assert metrics.recomputed_prompt_tokens == 0
    assert metrics.local_computed_prompt_tokens == 12
    assert metrics.local_cache_hit_prompt_tokens == 5
    assert metrics.external_kv_transfer_prompt_tokens == 3
    assert metrics.num_preemptions == 2
    assert metrics.is_corrupted is True
    assert metrics.requested_max_tokens == 16
    assert metrics.finish_reason == "stop"
    assert metrics.stop_reason == "done"


def test_request_metrics_are_serialized_at_response_top_level() -> None:
    metrics = build_vllm_request_metrics("chatcmpl-request-1", [make_request_stats()])
    response = ChatCompletionResponse(
        id="chatcmpl-request-1",
        model="test-model",
        choices=[],
        usage=UsageInfo(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        vllm_request_metrics=metrics,
    )

    data = response.model_dump()
    assert data["vllm_request_metrics"]["request_id"] == "chatcmpl-request-1"
    assert data["vllm_request_metrics"]["queue_time_seconds"] == pytest.approx(0.25)


def test_full_prefix_hit_tracks_recomputed_token() -> None:
    prefill_stats = PrefillStats(
        num_prompt_tokens=9,
        num_computed_tokens=1,
        num_cached_tokens=8,
        num_local_cached_tokens=8,
    )
    request_stats = RequestStateStats(arrival_time=100.0)

    IterationStats().update_from_output(
        EngineCoreOutput(
            request_id="request-1",
            new_token_ids=[1],
            prefill_stats=prefill_stats,
        ),
        engine_core_timestamp=10.0,
        is_prefilling=True,
        req_stats=request_stats,
        lora_states=None,  # type: ignore[arg-type]
        lora_name=None,
    )

    assert request_stats.num_recomputed_tokens == 1


def test_chat_request_accepts_nested_awbench_meta() -> None:
    awbench_meta = {
        "workflow_id": "workflow",
        "agent_steps_to_execution": {"workflow+program+agent": 0},
        "timestep_agents": {"0": ["workflow+program+agent"]},
    }

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        vllm_xargs={"awbench_meta": awbench_meta},
    )

    assert request.vllm_xargs == {"awbench_meta": awbench_meta}


def test_build_request_metrics_without_stats() -> None:
    assert build_vllm_request_metrics("request-without-stats", [None]) is None
