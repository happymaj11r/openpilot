"""Deterministic supercombo generation detection from ONNX input/output names.

The selector supports two generations; everything else is rejected at install
time with a clear error instead of crash-looping modeld on the device.

  Gen A  queues outside the ONNX (Lebowski/RDF/TSF era, Aug 2026):
         img, big_img, desire_pulse, traffic_convention, action_t, features_buffer
  Gen C  queues inside the ONNX (comma master since Oct 2026):
         new_img, desire, traffic_convention, action_t,
         state_img_q, state_desire_q, state_feat_q  ->  outputs + next_state_*
  Gen B  (Sep 2026 intermediate: warped/new_img + state_big_img_q + state_prev_feat)
         is not supported.
"""
from __future__ import annotations

from collections.abc import Iterable

GEN_A_INPUTS = frozenset({"img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"})
GEN_C_STATE_INPUTS = ("state_img_q", "state_desire_q", "state_feat_q")
GEN_C_INPUTS = frozenset({"new_img", "desire", "traffic_convention", "action_t", *GEN_C_STATE_INPUTS})
GEN_C_IMAGE_INPUT = "new_img"


def next_state_name(state_input: str) -> str:
    return f"next_{state_input}"


def detect_generation(input_names: Iterable[str], output_names: Iterable[str] = ()) -> str | None:
    """Return "A", "C", or None (unsupported) for the given ONNX graph I/O names."""
    inputs = frozenset(input_names)
    outputs = frozenset(output_names)
    if inputs == GEN_A_INPUTS:
        return "A"
    if inputs == GEN_C_INPUTS and all(next_state_name(s) in outputs for s in GEN_C_STATE_INPUTS):
        return "C"
    return None


def describe_inputs(input_names: Iterable[str]) -> str:
    return ", ".join(sorted(input_names))
