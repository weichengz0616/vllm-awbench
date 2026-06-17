# SPDX-License-Identifier: Apache-2.0

from vllm import SamplingParams
from vllm.v1.request import Request

from .test_offloading_connector import (
    EOS_TOKEN_ID,
    RequestRunner,
    generate_store_output,
)


def test_awbench_default_offload_store_and_load_factor_one():
    runner = RequestRunner(
        offloaded_block_size=16,
        gpu_block_size=16,
        num_gpu_blocks=100,
    )

    runner.new_request(token_ids=[0] * 16)
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(block_hashes)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored_gpu_block_indexes=(0,),
    )

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * 16)
    runner.manager.lookup.return_value = 1
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded_gpu_block_indexes=(0,),
    )


def test_awbench_request_meta_binds_during_real_schedule_factor_one():
    runner = RequestRunner(
        offloaded_block_size=16,
        gpu_block_size=16,
        num_gpu_blocks=100,
    )
    request = Request(
        request_id="meta-req",
        prompt_token_ids=[0] * 32,
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "awbench_meta": {
                    "workflow_id": "wf0",
                    "program_id": "p0",
                    "agent_id": "a0",
                    "fixed_prefix_len": 16,
                    "steps_to_execution": 7,
                    "critical": True,
                }
            },
        ),
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        block_hasher=runner._block_hasher,
    )

    runner.scheduler.add_request(request)
    scheduler_output = runner.scheduler.schedule()

    assert scheduler_output.kv_connector_metadata is not None
    (block_ids,) = runner.scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )
    assert block_ids

    block_pool = runner.scheduler.kv_cache_manager.block_pool
    first_metadata = block_pool.block_metadata[block_ids[0]]
    second_metadata = block_pool.block_metadata[block_ids[1]]

    assert first_metadata.workflow_id == "wf0"
    assert first_metadata.program_id == "p0"
    assert first_metadata.agent_id == "a0"
    assert first_metadata.prompt_part == "fixed"
    assert first_metadata.steps_to_execution == 7
    assert first_metadata.critical is True

    assert second_metadata.workflow_id == "wf0"
    assert second_metadata.program_id == "p0"
    assert second_metadata.agent_id == "a0"
    assert second_metadata.prompt_part == "dynamic"
    assert second_metadata.steps_to_execution == 7
    assert second_metadata.critical is True
