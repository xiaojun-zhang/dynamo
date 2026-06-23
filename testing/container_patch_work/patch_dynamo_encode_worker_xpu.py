from pathlib import Path


HANDLER = Path(
    "/usr/local/lib/python3.12/dist-packages/dynamo/sglang/"
    "request_handlers/multimodal/encode_worker_handler.py"
)

text = HANDLER.read_text()

old = "            grid_sizes.append(int(image_grid_thw[1] * image_grid_thw[2]))"
new = "            grid_sizes.append(int(image_grid_thw[0] * image_grid_thw[1] * image_grid_thw[2]))"

if old in text:
    text = text.replace(old, new, 1)
elif new not in text and "return int(grid_item[0] * grid_item[1] * grid_item[2])" not in text:
    raise RuntimeError("Could not find Dynamo encode worker grid split logic to patch")

old = """        image_token_str = (
            chat_templates[getattr(config.server_args, "chat_template")]
            .copy()
            .image_token
        )
"""
new = """        image_token_str = (
            chat_templates[getattr(config.server_args, "chat_template")]
            .copy()
            .image_token
        )
        self.image_token_str = image_token_str
"""

if old in text:
    text = text.replace(old, new, 1)
elif "        self.image_token_str = image_token_str\n" not in text:
    raise RuntimeError("Could not patch image token string retention")

old = """        self.min_workers = 1
"""
new = """        self.image_start_token_id = self._optional_token_id("<img>")
        self.image_end_token_id = self._optional_token_id("</img>")

        self.min_workers = 1
"""

if old in text:
    text = text.replace(old, new, 1)
elif "        self.image_start_token_id = self._optional_token_id(\"<img>\")\n" not in text:
    raise RuntimeError("Could not patch InternVL image boundary token ids")

old = """    def cleanup(self) -> None:
        pass
"""
new = """    def _optional_token_id(self, token: str) -> int | None:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return None

    def cleanup(self) -> None:
        pass
"""

if old in text:
    text = text.replace(old, new, 1)
elif "    def _optional_token_id(self, token: str) -> int | None:\n" not in text:
    raise RuntimeError("Could not insert optional token id helper")

old = """    @_nvtx.range_decorator("mm:enc:generate", color="blue")
    async def generate(
"""
new = """    def _insert_missing_image_placeholders(
        self, token_ids: list[int], needed: int
    ) -> list[int]:
        missing = needed - token_ids.count(self.image_token_id)
        if missing <= 0:
            return token_ids

        # InternVL Dynamo/SGLang preprocessing can carry image URLs in
        # multi_modal_data without preserving one textual image placeholder per
        # image in token_ids. Reinsert minimal placeholders so the existing
        # expansion path can align precomputed embeddings with input_ids.
        placeholder_ids = [self.image_token_id] * missing
        try:
            prompt = self.tokenizer.decode(token_ids)
            insert = "\\n" + "\\n".join([self.image_token_str] * missing) + "\\n"
            for marker in (
                "<|im_start|>assistant",
                "<|start_header_id|>assistant<|end_header_id|>",
                "Assistant:",
                "ASSISTANT:",
            ):
                idx = prompt.rfind(marker)
                if idx >= 0:
                    patched = prompt[:idx] + insert + prompt[idx:]
                    encoded = self.tokenizer.encode(
                        patched, add_special_tokens=False
                    )
                    if encoded.count(self.image_token_id) >= needed:
                        return encoded
                    break
        except Exception as exc:
            logger.warning(
                "Failed to retokenize prompt with image placeholders: %s", exc
            )

        return placeholder_ids + token_ids

    @_nvtx.range_decorator("mm:enc:generate", color="blue")
    async def generate(
"""

if old in text:
    text = text.replace(old, new, 1)
elif "    def _insert_missing_image_placeholders(\n" not in text:
    raise RuntimeError("Could not insert missing image placeholder helper")

old = """            image_placeholder_count = request.request.token_ids.count(
                self.image_token_id
            )
            if image_placeholder_count < len(multimodal_groups):
                raise ValueError(
                    "Not enough image placeholders in token_ids for provided images"
                )
"""
new = """            request.request.token_ids = self._insert_missing_image_placeholders(
                request.request.token_ids, len(multimodal_groups)
            )
            image_placeholder_count = request.request.token_ids.count(
                self.image_token_id
            )
            if image_placeholder_count < len(multimodal_groups):
                raise ValueError(
                    "Not enough image placeholders in token_ids for provided images"
                )
"""

if old in text:
    text = text.replace(old, new, 1)
elif "            request.request.token_ids = self._insert_missing_image_placeholders(\n" not in text:
    raise RuntimeError("Could not patch missing placeholder call site")

old = """                request.request.token_ids = (
                    request.request.token_ids[:image_token_id_index]
                    + [self.image_token_id] * num_image_tokens
                    + request.request.token_ids[image_token_id_index + 1 :]
                )
                search_start = image_token_id_index + num_image_tokens
"""
new = """                replacement = [self.image_token_id] * num_image_tokens
                has_image_boundaries = (
                    self.image_start_token_id is not None
                    and self.image_end_token_id is not None
                )
                already_wrapped = (
                    has_image_boundaries
                    and image_token_id_index > 0
                    and image_token_id_index + 1 < len(request.request.token_ids)
                    and request.request.token_ids[image_token_id_index - 1]
                    == self.image_start_token_id
                    and request.request.token_ids[image_token_id_index + 1]
                    == self.image_end_token_id
                )
                if has_image_boundaries and not already_wrapped:
                    replacement = (
                        [self.image_start_token_id]
                        + replacement
                        + [self.image_end_token_id]
                    )

                request.request.token_ids = (
                    request.request.token_ids[:image_token_id_index]
                    + replacement
                    + request.request.token_ids[image_token_id_index + 1 :]
                )
                search_start = image_token_id_index + len(replacement)
"""

if old in text:
    text = text.replace(old, new, 1)
elif "                already_wrapped = (\n" not in text:
    raise RuntimeError("Could not patch InternVL image boundary expansion")

HANDLER.write_text(text)
