# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Any

from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(
    raw_request: Request,
    reset_running_requests: bool = Query(default=False),
    reset_external: bool = Query(default=False),
):
    """
    Reset the local prefix cache.

    Optionally, if the query parameter `reset_external=true`
    also resets the external (connector-managed) prefix cache.

    Note that we currently do not check if the prefix cache
    is successfully reset in the API server.

    Example:
       POST /reset_prefix_cache?reset_external=true
    """
    logger.info("Resetting prefix cache...")

    await engine_client(raw_request).reset_prefix_cache(
        reset_running_requests, reset_external
    )
    return Response(status_code=200)


@router.post("/finish_program")
async def finish_program(
    raw_request: Request,
    payload: dict[str, Any] = Body(...),
):
    """Notify cache policies that a program has finished."""
    workflow_id = payload.get("workflow_id") or payload.get("template_id")
    program_id = payload.get("program_id")
    if (
        not isinstance(workflow_id, str)
        or not workflow_id
        or not isinstance(program_id, str)
        or not program_id
    ):
        raise HTTPException(
            status_code=400,
            detail="workflow_id/template_id and program_id are required",
        )

    logger.info(
        "Finishing program: workflow_id=%s program_id=%s",
        workflow_id,
        program_id,
    )
    cleared_contributions = await engine_client(raw_request).finish_program(
        workflow_id, program_id
    )
    return {"cleared_contributions": cleared_contributions}


@router.post("/reset_mm_cache")
async def reset_mm_cache(raw_request: Request):
    """
    Reset the multi-modal cache. Note that we currently do not check if the
    multi-modal cache is successfully reset in the API server.
    """
    logger.info("Resetting multi-modal cache...")
    await engine_client(raw_request).reset_mm_cache()
    return Response(status_code=200)


@router.post("/reset_encoder_cache")
async def reset_encoder_cache(raw_request: Request):
    """
    Reset the encoder cache. Note that we currently do not check if the
    encoder cache is successfully reset in the API server.
    """
    logger.info("Resetting encoder cache...")
    await engine_client(raw_request).reset_encoder_cache()
    return Response(status_code=200)


def attach_router(app: FastAPI):
    # if not envs.VLLM_SERVER_DEV_MODE:
    #     return
    app.include_router(router)
