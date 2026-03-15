# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router for 2-stage omni disaggregation.

Orchestrates a 2-stage DAG: discovers stage workers via etcd, calls stage
endpoints, runs stage_input_processors between stages, transfers bulk data
via OmniConnector.
"""

import importlib
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict

logger = logging.getLogger(__name__)


class StageOutputProxy:
    """Minimal proxy satisfying what stage_input_processors expect.

    vLLM-Omni processor functions (ar2diffusion, thinker2talker, etc.) access
    stage_list[source_stage_id].engine_outputs. This proxy mimics that
    interface with just the .engine_outputs attribute.
    """

    def __init__(self):
        self.engine_outputs = None


class OmniStageRouter:
    """Orchestrates a 2-stage omni disaggregation pipeline.

    Runs in the frontend process. Discovers stage workers via etcd,
    routes requests through the stage DAG, and handles inter-stage
    data transfer via OmniConnector.
    """

    def __init__(self, stage_configs_path: str):
        from vllm_omni.distributed.omni_connectors import (
            initialize_orchestrator_connectors,
        )
        from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml

        self.stage_configs_path = stage_configs_path

        # Load stage DAG from YAML
        self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)
        if len(self.stage_configs) != 2:
            raise ValueError(
                f"OmniStageRouter Phase 1 supports exactly 2 stages, "
                f"got {len(self.stage_configs)}"
            )

        # Load runtime edges from YAML
        self.edges = self._load_edges(stage_configs_path)

        # One proxy per stage for processor compatibility
        self.stage_proxies = [StageOutputProxy(), StageOutputProxy()]

        # Dynamically import stage input processor functions from YAML
        self.processors: Dict[int, Any] = {}
        for cfg in self.stage_configs:
            func_path = getattr(cfg, "custom_process_input_func", None)
            if func_path:
                module_path, func_name = func_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                self.processors[cfg.stage_id] = getattr(module, func_name)
                logger.info(
                    "Loaded processor for stage %d: %s", cfg.stage_id, func_path
                )

        # Create orchestrator-level connectors for inter-stage transfer
        self.omni_transfer_config, self.connectors = (
            initialize_orchestrator_connectors(stage_configs_path)
        )
        logger.info(
            "OmniStageRouter initialized: %d stages, %d connectors, %d processors",
            len(self.stage_configs),
            len(self.connectors),
            len(self.processors),
        )

        # Stage endpoints discovered via etcd (set by caller after construction)
        self.stage_endpoints: Dict[str, Any] = {}

    def _load_edges(self, config_path: str) -> list:
        """Load runtime edges from stage config YAML."""
        from omegaconf import OmegaConf

        config_data = OmegaConf.load(config_path)
        runtime = config_data.get("runtime", {})
        edges = runtime.get("edges", [])
        return list(edges) if edges else []

    def set_stage_endpoint(self, model_stage: str, endpoint):
        """Register a discovered stage endpoint (called by frontend)."""
        self.stage_endpoints[model_stage] = endpoint
        logger.info("Registered endpoint for stage '%s'", model_stage)

    async def generate(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute the 2-stage pipeline for a single request."""
        request_id = str(uuid.uuid4())
        stage0_cfg = self.stage_configs[0]
        stage1_cfg = self.stage_configs[1]

        stage0_name = getattr(
            stage0_cfg.engine_args, "model_stage", "stage0"
        )
        stage1_name = getattr(
            stage1_cfg.engine_args, "model_stage", "stage1"
        )

        logger.debug(
            "Request %s: starting 2-stage pipeline (%s -> %s)",
            request_id,
            stage0_name,
            stage1_name,
        )

        # --- Stage 0 ---
        t0 = time.time()
        stage0_request = {
            "request_id": request_id,
            "engine_inputs": request,
            "sampling_params": self._get_sampling_params(stage0_cfg),
        }
        stage0_result = await self._call_stage(stage0_name, stage0_request)
        logger.debug(
            "Request %s: stage 0 (%s) completed in %.2fs",
            request_id,
            stage0_name,
            time.time() - t0,
        )

        # Stream final output from stage 0 if applicable (e.g. BAGEL text)
        if stage0_cfg.final_output:
            final_data = stage0_result.get("final_data")
            if final_data:
                yield final_data

        # --- Transform: stage_input_processor ---
        stage_output = stage0_result.get("stage_output")
        if stage_output is None:
            yield {
                "error": f"Stage 0 ({stage0_name}) returned no stage_output",
                "finished": True,
            }
            return

        # Set proxy so processor can read upstream output via
        # stage_list[0].engine_outputs
        self.stage_proxies[0].engine_outputs = stage_output

        next_inputs = self._run_processor(stage1_cfg, request)

        # --- Transfer via OmniConnector ---
        connector_key = ("0", "1")
        connector = self.connectors.get(connector_key)

        if connector is not None:
            t_transfer = time.time()
            payload = {
                "engine_inputs": next_inputs,
                "sampling_params": self._get_sampling_params(stage1_cfg),
                "metadata": {
                    "original_prompt": request,
                    "stage_transition": "0->1",
                    "timestamp": time.time(),
                },
            }
            success, serialized_size, metadata = connector.put(
                "0", "1", str(request_id), payload
            )
            logger.debug(
                "Request %s: connector.put() %s, %d bytes, %.1fms",
                request_id,
                "ok" if success else "FAILED",
                serialized_size,
                (time.time() - t_transfer) * 1000,
            )

            if not success:
                yield {
                    "error": "Inter-stage connector.put() failed",
                    "finished": True,
                }
                return

            # Lightweight notification to stage 1
            stage1_request = {
                "request_id": request_id,
                "from_connector": True,
                "from_stage": "0",
                "to_stage": "1",
                "connector_metadata": metadata,
                "sampling_params": self._get_sampling_params(stage1_cfg),
            }
        else:
            # No connector configured — send inputs inline
            logger.debug(
                "Request %s: no connector for edge 0->1, sending inline",
                request_id,
            )
            stage1_request = {
                "request_id": request_id,
                "engine_inputs": next_inputs,
                "sampling_params": self._get_sampling_params(stage1_cfg),
            }

        # --- Stage 1 ---
        t1 = time.time()
        stage1_result = await self._call_stage(stage1_name, stage1_request)
        logger.debug(
            "Request %s: stage 1 (%s) completed in %.2fs",
            request_id,
            stage1_name,
            time.time() - t1,
        )

        # Stream final output from stage 1 (images, video, etc.)
        if stage1_cfg.final_output:
            final_data = stage1_result.get("final_data")
            if final_data:
                yield final_data

        # Clean up connector resources for this request
        if connector is not None:
            try:
                connector.cleanup(str(request_id))
            except Exception:
                pass

    def _run_processor(self, next_stage_cfg, original_request: Dict) -> Any:
        """Run stage_input_processor to transform upstream output."""
        stage_id = next_stage_cfg.stage_id

        if stage_id in self.processors:
            processor_fn = self.processors[stage_id]
            engine_input_source = getattr(next_stage_cfg, "engine_input_source", [0])
            requires_mm = getattr(next_stage_cfg, "requires_multimodal_data", False)

            return processor_fn(
                self.stage_proxies,
                engine_input_source,
                [original_request],
                requires_mm,
            )
        else:
            # Default: pass stage_output directly
            return self.stage_proxies[0].engine_outputs

    async def _call_stage(
        self, model_stage: str, request_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a stage worker via its Dynamo endpoint."""
        endpoint = self.stage_endpoints.get(model_stage)
        if endpoint is None:
            raise RuntimeError(
                f"No endpoint discovered for stage '{model_stage}'. "
                f"Available: {list(self.stage_endpoints.keys())}"
            )

        result: Dict[str, Any] = {}
        async for chunk in endpoint.send(request_payload):
            result.update(chunk)
        return result

    def _get_sampling_params(self, stage_cfg) -> dict:
        """Get default sampling params from stage config."""
        sp = getattr(stage_cfg, "default_sampling_params", None)
        if sp is None:
            return {}
        # OmegaConf -> dict
        from omegaconf import OmegaConf

        if OmegaConf.is_config(sp):
            return OmegaConf.to_container(sp, resolve=True)
        return dict(sp)
