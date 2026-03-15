# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize a single vLLM-Omni stage as an independent Dynamo service."""

import logging
import os

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


async def init_omni_stage(
    runtime: DistributedRuntime,
    config,
    shutdown_event,
) -> None:
    """Initialize a single omni stage worker.

    Follows the init_prefill() pattern: load config, create engine, register
    with etcd, serve endpoint.
    """
    from vllm_omni.distributed.omni_connectors import (
        initialize_orchestrator_connectors,
    )
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        build_stage_connectors,
        get_stage_connector_config,
    )
    from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml

    from dynamo.vllm.omni.stage_handler import OmniStageWorkerHandler

    stage_id = config.stage_id
    stage_configs_path = config.stage_configs_path

    if stage_configs_path is None:
        raise ValueError("--stage-configs-path is required for disaggregated omni mode")

    # Load all stage configs from YAML
    stage_configs = load_stage_configs_from_yaml(stage_configs_path)
    if stage_id >= len(stage_configs):
        raise ValueError(
            f"--stage-id {stage_id} out of range (YAML has {len(stage_configs)} stages)"
        )
    my_config = stage_configs[stage_id]

    # Set GPU visibility from stage config
    devices = getattr(my_config.runtime, "devices", None)
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        logger.info("Stage %d: CUDA_VISIBLE_DEVICES=%s", stage_id, devices)

    stage_type = getattr(my_config, "stage_type", "llm")
    model_stage = getattr(my_config.engine_args, "model_stage", f"stage{stage_id}")

    # Create engine
    engine = _create_engine(config.model, my_config, stage_type)
    logger.info("Stage %d (%s): engine created, type=%s", stage_id, model_stage, stage_type)

    # Build connectors for this stage (receiving side)
    transfer_config, orchestrator_connectors = initialize_orchestrator_connectors(
        stage_configs_path
    )
    connector_config = get_stage_connector_config(transfer_config, stage_id)
    connectors = build_stage_connectors(stage_id, connector_config) or {}

    # Also keep orchestrator connectors for edges where this stage receives
    # try_recv_via_connector expects connectors keyed by (from_stage, to_stage)
    for key, conn in orchestrator_connectors.items():
        if key not in connectors:
            connectors[key] = conn

    logger.info("Stage %d: connectors initialized, keys=%s", stage_id, list(connectors.keys()))

    # Create handler
    handler = OmniStageWorkerHandler(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        stage_id=stage_id,
    )

    # Register endpoint with etcd
    endpoint_name = f"{config.namespace}.{model_stage}.generate"
    generate_endpoint = runtime.endpoint(endpoint_name)

    final_output_type = getattr(my_config, "final_output_type", "text")
    model_type = _resolve_model_type(final_output_type)

    await register_model(
        ModelInput.Tokens,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
    )

    logger.info(
        "Stage %d (%s): registered endpoint '%s' as %s",
        stage_id,
        model_stage,
        endpoint_name,
        model_type,
    )

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
        )
    except Exception as e:
        logger.error("Stage %d: endpoint failed: %s", stage_id, e)
        raise
    finally:
        handler.cleanup()


def _create_engine(model: str, stage_config, stage_type: str):
    """Create OmniLLM or OmniDiffusion engine from stage config."""
    engine_args = stage_config.engine_args

    if stage_type == "llm":
        from vllm_omni.entrypoints.omni_llm import OmniLLM

        kwargs = _engine_kwargs_from_config(engine_args)
        kwargs["model"] = model
        return OmniLLM(**kwargs)

    elif stage_type == "diffusion":
        from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion

        kwargs = _diffusion_kwargs_from_config(engine_args, model)
        return OmniDiffusion(**kwargs)

    else:
        raise ValueError(f"Unknown stage_type: {stage_type}")


def _engine_kwargs_from_config(engine_args) -> dict:
    """Extract OmniLLM kwargs from stage engine_args."""
    kwargs = {}
    _copy_if_set(kwargs, engine_args, "trust_remote_code")
    _copy_if_set(kwargs, engine_args, "gpu_memory_utilization")
    _copy_if_set(kwargs, engine_args, "enforce_eager")
    _copy_if_set(kwargs, engine_args, "tensor_parallel_size")
    _copy_if_set(kwargs, engine_args, "pipeline_parallel_size")
    _copy_if_set(kwargs, engine_args, "max_num_batched_tokens")
    _copy_if_set(kwargs, engine_args, "max_num_seqs")
    _copy_if_set(kwargs, engine_args, "max_model_len")
    _copy_if_set(kwargs, engine_args, "enable_prefix_caching")
    _copy_if_set(kwargs, engine_args, "distributed_executor_backend")
    _copy_if_set(kwargs, engine_args, "worker_cls")
    _copy_if_set(kwargs, engine_args, "scheduler_cls")
    return kwargs


def _diffusion_kwargs_from_config(engine_args, model: str) -> dict:
    """Extract OmniDiffusion kwargs from stage engine_args."""
    kwargs = {"model": model}
    _copy_if_set(kwargs, engine_args, "trust_remote_code")
    _copy_if_set(kwargs, engine_args, "enforce_eager")
    _copy_if_set(kwargs, engine_args, "num_gpus")
    return kwargs


def _copy_if_set(target: dict, source, key: str):
    val = getattr(source, key, None)
    if val is not None:
        target[key] = val


def _resolve_model_type(final_output_type: str) -> ModelType:
    if final_output_type == "image":
        return ModelType.Images
    elif final_output_type == "text":
        return ModelType.Chat
    elif final_output_type == "video":
        return ModelType.Videos
    else:
        return ModelType.Images
