"""Single-frame Qwen3.5 runtime probe.

Run this helper from a detached tmux session.  It intentionally exercises one
visual generation instead of the full video pipeline so fused-kernel modes can
be compared without producing a batch result.
"""

from __future__ import annotations

import os
import time

from PIL import Image

from preprocessing.video.caption import VisualProcessor


def main() -> None:
    model_path = os.environ.get("MODEL_BASE_PATH", "/gemini/pretrain/model")
    resolved_model_path = os.path.join(model_path, "Qwen", "Qwen3.5-9B")
    processor = VisualProcessor(
        model_name=resolved_model_path,
        model_path=resolved_model_path,
        device="cuda",
        require_local=True,
    )
    started = time.perf_counter()
    processor._load_model()
    loaded_seconds = time.perf_counter() - started
    print(
        "PROBE_LOADED mode={} seconds={:.3f} model={} processor={}".format(
            os.environ.get("TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS", ""),
            loaded_seconds,
            type(processor._model).__name__,
            type(processor._processor).__name__,
        ),
        flush=True,
    )
    image = Image.new("RGB", (224, 224), color=(96, 128, 160))
    started = time.perf_counter()
    answer = processor._generate_caption([image], "")
    print(
        "PROBE_GENERATED seconds={:.3f} answer={!r}".format(
            time.perf_counter() - started,
            answer,
        ),
        flush=True,
    )
    processor.unload_model(clear_runtime_cache=True)


if __name__ == "__main__":
    main()
