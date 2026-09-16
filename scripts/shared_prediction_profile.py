"""Optional, process-local instrumentation for the fixed-checkpoint benchmark."""
from __future__ import annotations

import json
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import torch

from intrep.problems.shared_prediction.questions import QuestionSource


class TrainingProfile:
    """Record measured updates without changing the training implementation."""

    def __init__(self, trainer, callbacks, output, *, device):
        self.trainer, self.callbacks, self.output = trainer, callbacks, output
        self.original_callbacks = callbacks.copy()
        self.source = "unknown"
        self.stack = ExitStack()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # Shape/stack/memory tracing adds overhead and can retain tensor references.
        self.profiler = torch.profiler.profile(activities=activities)

    @staticmethod
    def annotated(function, name):
        @wraps(function)
        def run(*args, **kwargs):
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)
        return run

    def __enter__(self):
        self.stack.enter_context(self.profiler)
        try:
            for owner, attribute, label in (
                (self.trainer, "synchronize_parameters", "synchronize_parameters"),
                (self.trainer.optimizer, "zero_grad", "zero_grad"),
                (self.trainer.optimizer, "step", "optimizer"),
                (torch.nn.utils, "clip_grad_norm_", "clip_grad_norm"),
                (self.trainer.model.core, "forward", "body_forward"),
                (QuestionSource, "_records", "read_records"),
                (QuestionSource, "_question", "prepare_question"),
            ):
                self.stack.enter_context(patch.object(
                    owner, attribute, self.annotated(getattr(owner, attribute), "intrep/" + label)))
            backward = torch.autograd.backward

            def annotated_backward(*args, **kwargs):
                with torch.profiler.record_function("intrep/backward/" + self.source):
                    return backward(*args, **kwargs)

            self.stack.enter_context(patch.object(torch.autograd, "backward", annotated_backward))
            for name, callback in self.original_callbacks.items():
                def annotated_source(name=name, callback=callback):
                    self.source = name
                    with torch.profiler.record_function("intrep/source/" + name):
                        return callback()
                self.callbacks[name] = annotated_source
        except BaseException:
            self.callbacks.update(self.original_callbacks)
            self.stack.close()
            raise
        return self

    def __exit__(self, *exception):
        self.callbacks.update(self.original_callbacks)
        self.stack.__exit__(*exception)
        if exception[0] is None:
            self.profiler.export_chrome_trace(str(self.output / "profile-trace.json.gz"))
            rows = [{"name": row.key, "count": row.count,
                     "cpu_total_ms": row.cpu_time_total / 1000,
                     "cpu_self_ms": row.self_cpu_time_total / 1000,
                     "device_total_ms": row.device_time_total / 1000,
                     "device_self_ms": row.self_device_time_total / 1000}
                    for row in self.profiler.key_averages()]
            summary = {
                "scope": "Instrumented measured updates only. Nested CPU/device totals overlap; do not sum them. Use unprofiled runs for speed comparisons.",
                "record_shapes": False, "with_stack": False, "profile_memory": False,
                "operators": sorted(rows, key=lambda row: row["device_self_ms"], reverse=True),
            }
            (self.output / "profile-operators.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
