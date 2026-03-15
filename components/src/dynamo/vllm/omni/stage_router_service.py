# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run OmniStageRouter as a standalone Dynamo service.

Registers with etcd as a backend (ModelType.Images), so the standard frontend
discovers it like any other omni backend. Internally discovers stage workers
and orchestrates the 2-stage pipeline.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime

from .stage_router import OmniStageRouter

logger = logging.getLogger(__name__)


async def init_omni_stage_router(
    runtime: DistributedRuntime,
    config,
    shutdown_event: asyncio.Event,
) -> None:
    """Initialize OmniStageRouter as a Dynamo service.

    The router registers with etcd as a regular backend so the frontend
    can discover and route to it. Internally, it discovers stage workers
    and orchestrates the 2-stage DAG.
    """
    stage_configs_path = config.stage_configs_path
    if stage_configs_path is None:
        raise ValueError("--stage-configs-path is required for --omni-router")

    router = OmniStageRouter(stage_configs_path)

    # Discover stage worker endpoints via etcd
    await _discover_stage_endpoints(runtime, config, router)

    # Create handler that adapts OmniStageRouter.generate() to Dynamo endpoint
    handler = OmniStageRouterHandler(router)

    # Register as a regular backend endpoint (same as monolithic OmniHandler)
    endpoint_name = f"{config.namespace}.{config.component}.{config.endpoint}"
    generate_endpoint = runtime.endpoint(endpoint_name)

    # Determine model type from stage configs (last stage's final_output_type)
    final_stage = router.stage_configs[-1]
    final_output_type = getattr(final_stage, "final_output_type", "image")
    model_type = _resolve_model_type(final_output_type)

    await register_model(
        ModelInput.Text,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
    )

    logger.info(
        "OmniStageRouter registered as '%s' with model_type=%s",
        endpoint_name,
        model_type,
    )

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
        )
    except Exception as e:
        logger.error("OmniStageRouter endpoint failed: %s", e)
        raise


async def _discover_stage_endpoints(
    runtime: DistributedRuntime,
    config,
    router: OmniStageRouter,
) -> None:
    """Discover stage worker endpoints via etcd and register them with the router."""
    for stage_cfg in router.stage_configs:
        model_stage = getattr(stage_cfg.engine_args, "model_stage", None)
        if model_stage is None:
            model_stage = f"stage{stage_cfg.stage_id}"

        endpoint_name = f"{config.namespace}.{model_stage}.generate"
        endpoint = runtime.endpoint(endpoint_name)
        client = await endpoint.client()
        router.set_stage_endpoint(model_stage, client)
        logger.info("Discovered stage endpoint: %s", endpoint_name)


class OmniStageRouterHandler:
    """Adapts OmniStageRouter.generate() to Dynamo endpoint handler interface."""

    def __init__(self, router: OmniStageRouter):
        self.router = router

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        request_id = context.id()
        logger.debug("OmniStageRouterHandler request=%s", request_id)

        async for chunk in self.router.generate(request):
            yield chunk


def _resolve_model_type(final_output_type: str) -> ModelType:
    if final_output_type == "image":
        return ModelType.Images
    elif final_output_type == "text":
        return ModelType.Chat
    elif final_output_type == "video":
        return ModelType.Videos
    else:
        return ModelType.Images
