"""Benchmark-only patches for local SGLang experiments.

This is loaded by Python only when BENCH_PYTHONPATH points at bench_patches.
It keeps the server path untouched while allowing sglang.bench_serving to build
InternVL image datasets when HuggingFace AutoProcessor fails on the local
InternVL3.5 tokenizer metadata.
"""

import json
import os


def _is_internvl_path(model_path):
    if not model_path:
        return False
    if "InternVL" in model_path or "internvl" in model_path:
        return True
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("model_type") == "internvl_chat"
    except Exception:
        return False


class _InternVLBenchProcessor:
    def __init__(self, model_path):
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if getattr(self.tokenizer, "chat_template", None) is None:
            template_path = os.path.join(model_path, "chat_template.jinja")
            if os.path.isfile(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    self.tokenizer.chat_template = f.read()

        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self._image_size = 448
        self._max_dynamic_patch = 12
        self._use_thumbnail = True
        self._tokens_per_tile = 256

        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._max_dynamic_patch = int(cfg.get("max_dynamic_patch", 12))
            self._use_thumbnail = bool(cfg.get("use_thumbnail", True))
            vision = cfg.get("vision_config", {})
            image_size = int(cfg.get("force_image_size") or vision.get("image_size", 448))
            patch_size = int(vision.get("patch_size", 14))
            downsample_ratio = float(cfg.get("downsample_ratio", 0.5))
            self._image_size = image_size
            self._tokens_per_tile = int((image_size // patch_size) ** 2 * downsample_ratio**2)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
            )
        except Exception:
            parts = []
            for message in messages:
                parts.append(f"<|im_start|>{message.get('role', 'user')}\n")
                content = message.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                else:
                    for item in content:
                        if item.get("type") in ("image", "image_url"):
                            parts.append("<image>\n")
                        elif item.get("type") == "text":
                            parts.append(item.get("text", ""))
                parts.append("<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            rendered = "".join(parts)
            if tokenize:
                return self.tokenizer.encode(rendered)
            return rendered

    def _tile_count(self, image):
        width, height = image.size
        aspect_ratio = width / max(height, 1)
        best_ratio = (1, 1)
        best_diff = float("inf")
        for n in range(1, self._max_dynamic_patch + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    blocks = i * j
                    if blocks > self._max_dynamic_patch:
                        continue
                    diff = abs(aspect_ratio - (i / j))
                    best_blocks = best_ratio[0] * best_ratio[1]
                    if diff < best_diff or (diff == best_diff and blocks > best_blocks):
                        best_diff = diff
                        best_ratio = (i, j)
        tiles = best_ratio[0] * best_ratio[1]
        if self._use_thumbnail and tiles > 1:
            tiles += 1
        return tiles

    def __call__(self, text=None, images=None, padding=False, return_tensors=None, **kwargs):
        import torch

        if isinstance(text, list):
            text_value = text[0] if text else ""
        else:
            text_value = text or ""
        text_tokens = len(self.tokenizer.encode(text_value))
        image_tokens = 0
        for image in images or []:
            image_tokens += self._tile_count(image) * self._tokens_per_tile
        token_count = max(1, text_tokens + image_tokens)
        return {"input_ids": torch.zeros((1, token_count), dtype=torch.long)}


def _install():
    import sglang.benchmark.utils as bench_utils

    original_get_processor = bench_utils.get_processor

    def get_processor_with_internvl_fallback(model_path):
        try:
            return original_get_processor(model_path)
        except AttributeError as exc:
            if "start_image_token" not in str(exc) or not _is_internvl_path(model_path):
                raise
            print(
                "bench_patches: using tokenizer-based InternVL processor fallback "
                f"for {model_path}"
            )
            return _InternVLBenchProcessor(model_path)

    bench_utils.get_processor = get_processor_with_internvl_fallback


_install()
