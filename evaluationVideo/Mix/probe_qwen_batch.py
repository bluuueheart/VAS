"""Opt-in Qwen multimodal batch-generation probe.

This probe never writes Mix progress or sample results.  Run it in a detached
tmux session with the same fallback environment as the formal batch.
"""

from __future__ import annotations

import os
import time

from PIL import Image

from preprocessing.video.caption import VisualProcessor


def main() -> None:
    model_base_path = os.environ.get("MODEL_BASE_PATH", "/gemini/pretrain/model")
    model_path = os.path.join(model_base_path, "Qwen", "Qwen3.5-9B")
    processor = VisualProcessor(
        model_name=model_path,
        model_path=model_path,
        device="cuda",
        require_local=True,
    )
    started = time.perf_counter()
    processor._load_model()
    print(
        "BATCH_PROBE_LOADED seconds={:.3f} model={} processor={}".format(
            time.perf_counter() - started,
            type(processor._model).__name__,
            type(processor._processor).__name__,
        ),
        flush=True,
    )
    images = [
        Image.new("RGB", (224, 224), color=(96, 128, 160)),
        Image.new("RGB", (224, 224), color=(160, 96, 128)),
    ]
    generation_started = time.perf_counter()
    answers = processor._generate_caption_batch(
        [[images[0]], [images[1]]],
        max_new_tokens_override=64,
        use_cache_override=True,
    )
    print(
        "BATCH_PROBE_GENERATED seconds={:.3f} batch_size={} answers={!r}".format(
            time.perf_counter() - generation_started,
            len(answers),
            answers,
        ),
        flush=True,
    )
    processor.unload_model(clear_runtime_cache=True)


if __name__ == "__main__":
    main()
