# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.engine.protocol import (
    UsageInfo,
    build_vllm_request_metrics,
)
from vllm.v1.metrics.stats import RequestStateStats


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
        num_recomputed_tokens=0,
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


def test_build_request_metrics_without_stats() -> None:
    assert build_vllm_request_metrics("request-without-stats", [None]) is None
