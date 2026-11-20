"""Deterministic supercombo generation detection from ONNX input/output names.

The selector supports three generations; everything else is rejected at install
time with a clear error instead of crash-looping modeld on the device.

  Gen A  queues outside the ONNX (Lebowski/RDF/TSF era, Aug 2026):
         img, big_img, desire_pulse, traffic_convention, action_t, features_buffer
  Gen B  queues inside the ONNX, one image queue per camera (the first two exports of
         comma PR #38916, 2026-09-15 — "Onnx Model History" v1/v2 in the catalog):
         new_img | warped, desire, traffic_convention, action_t,
         state_img_q, state_big_img_q, state_desire_q, state_prev_feat, state_feat_q
  Gen C  queues inside the ONNX, one merged image queue (comma master since Oct 2026):
         new_img, desire, traffic_convention, action_t,
         state_img_q, state_desire_q, state_feat_q

Gen B and Gen C are one family ("state-queue"): the image input is the same
(2 cameras, 6 planes, 128, 256) warp output with the road camera first, the host feeds
only desire / traffic_convention / action_t, and every `state_*` input is fed back from
its `next_state_*` output. They differ only in which state tensors exist, so the engine
in genc/ runs both. The state sets are listed explicitly rather than matched by prefix:
a model with a state tensor nobody has looked at is rejected.
"""
from __future__ import annotations

from collections.abc import Iterable

GEN_A_INPUTS = frozenset({"img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"})

# Inputs the host fills every frame; identical for Gen B and Gen C.
HOST_INPUTS = ("desire", "traffic_convention", "action_t")

GEN_C_STATE_INPUTS = ("state_img_q", "state_desire_q", "state_feat_q")
GEN_C_IMAGE_INPUT = "new_img"
GEN_C_INPUTS = frozenset({GEN_C_IMAGE_INPUT, *HOST_INPUTS, *GEN_C_STATE_INPUTS})

GEN_B_STATE_INPUTS = ("state_img_q", "state_big_img_q", "state_desire_q", "state_prev_feat", "state_feat_q")
# comma renamed `warped` to `new_img` one commit after the first export; it is the same tensor.
GEN_B_IMAGE_INPUTS = (GEN_C_IMAGE_INPUT, "warped")

# Generations run by the state-queue engine (genc/), most recent first.
STATE_QUEUE_GENERATIONS = ("C", "B")
_STATE_INPUTS = {"C": GEN_C_STATE_INPUTS, "B": GEN_B_STATE_INPUTS}
_IMAGE_INPUTS = {"C": (GEN_C_IMAGE_INPUT,), "B": GEN_B_IMAGE_INPUTS}


def next_state_name(state_input: str) -> str:
    return f"next_{state_input}"


def state_input_names(generation: str | None) -> tuple[str, ...]:
    """State tensors of a state-queue generation, in a stable order; () for anything else."""
    return _STATE_INPUTS.get(generation or "", ())


def image_input_name(input_names: Iterable[str]) -> str | None:
    """Name of the warped-image input of a state-queue model, or None if absent/ambiguous."""
    found = [name for name in GEN_B_IMAGE_INPUTS if name in frozenset(input_names)]
    return found[0] if len(found) == 1 else None


def detect_generation(input_names: Iterable[str], output_names: Iterable[str] = ()) -> str | None:
    """Return "A", "B", "C", or None (unsupported) for the given ONNX graph I/O names."""
    inputs = frozenset(input_names)
    outputs = frozenset(output_names)
    if inputs == GEN_A_INPUTS:
        return "A"
    image = image_input_name(inputs)
    if image is None:
        return None
    for generation in STATE_QUEUE_GENERATIONS:
        states = _STATE_INPUTS[generation]
        if (image in _IMAGE_INPUTS[generation] and inputs == frozenset({image, *HOST_INPUTS, *states})
                and all(next_state_name(s) in outputs for s in states)):
            return generation
    return None


def describe_inputs(input_names: Iterable[str]) -> str:
    return ", ".join(sorted(input_names))
