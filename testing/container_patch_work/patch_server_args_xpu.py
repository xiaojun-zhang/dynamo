from pathlib import Path


SERVER_ARGS = Path("/opt/sglang/python/sglang/srt/server_args.py")


def insert_once(text: str, marker: str, addition: str) -> str:
    if addition.strip() in text:
        return text
    if marker not in text:
        raise RuntimeError(f"Could not find server_args.py marker: {marker!r}")
    return text.replace(marker, marker.replace("\n", f"\n{addition}", 1), 1)


text = SERVER_ARGS.read_text()

text = insert_once(
    text,
    '            "KimiK25ForConditionalGeneration",\n        ]:',
    '            "InternVLChatModel",\n',
)

text = insert_once(
    text,
    '                "flashinfer_cudnn",\n            ],',
    '                "xpu_attn",\n',
)

SERVER_ARGS.write_text(text)
