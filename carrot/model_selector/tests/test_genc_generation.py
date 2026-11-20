"""Generation detection must be exact: only the listed Gen A / B / C I/O sets are accepted."""
from __future__ import annotations

from carrot.model_selector.genc.generation import (GEN_A_INPUTS, GEN_B_STATE_INPUTS, GEN_C_INPUTS, GEN_C_STATE_INPUTS,
                                                   detect_generation, image_input_name, state_input_names)

# Real names seen in the openpilot-models catalog (2026-11-20).
TSFDO_INPUTS = ["img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"]
SSHV4_INPUTS = ["new_img", "desire", "traffic_convention", "action_t", "state_img_q", "state_desire_q", "state_feat_q"]
SSHV4_OUTPUTS = ["outputs", "next_state_img_q", "next_state_desire_q", "next_state_feat_q"]
# "Onnx Model History" v1 / v2: the first two exports of comma PR #38916 (one image queue per camera).
OMH_V1_INPUTS = ["warped", "state_img_q", "state_big_img_q", "desire", "state_desire_q", "state_prev_feat",
                 "state_feat_q", "traffic_convention", "action_t"]
OMH_V2_INPUTS = ["new_img", "state_img_q", "state_big_img_q", "desire", "state_desire_q", "state_prev_feat",
                 "state_feat_q", "traffic_convention", "action_t"]
OMH_B_OUTPUTS = ["outputs", "next_state_img_q", "next_state_big_img_q", "next_state_desire_q", "next_state_feat_q",
                 "next_state_prev_feat"]


def test_gen_a_from_tsfdo_inputs():
  assert detect_generation(TSFDO_INPUTS, ["outputs"]) == "A"
  assert set(TSFDO_INPUTS) == GEN_A_INPUTS


def test_gen_c_from_sshv4_io():
  assert detect_generation(SSHV4_INPUTS, SSHV4_OUTPUTS) == "C"
  assert set(SSHV4_INPUTS) == GEN_C_INPUTS


def test_gen_c_requires_every_next_state_output():
  missing_feat = [o for o in SSHV4_OUTPUTS if o != "next_state_feat_q"]
  assert detect_generation(SSHV4_INPUTS, missing_feat) is None
  assert detect_generation(SSHV4_INPUTS, []) is None


def test_gen_b_from_both_omh_exports():
  assert detect_generation(OMH_V1_INPUTS, OMH_B_OUTPUTS) == "B"
  assert detect_generation(OMH_V2_INPUTS, OMH_B_OUTPUTS) == "B"


def test_gen_b_requires_every_next_state_output():
  for dropped in ("next_state_prev_feat", "next_state_big_img_q"):
    assert detect_generation(OMH_V2_INPUTS, [o for o in OMH_B_OUTPUTS if o != dropped]) is None
  assert detect_generation(OMH_V1_INPUTS, SSHV4_OUTPUTS) is None


def test_image_input_name():
  assert image_input_name(SSHV4_INPUTS) == "new_img"
  assert image_input_name(OMH_V1_INPUTS) == "warped"
  assert image_input_name(TSFDO_INPUTS) is None
  assert image_input_name(SSHV4_INPUTS + ["warped"]) is None  # ambiguous


def test_state_input_names():
  assert state_input_names("C") == GEN_C_STATE_INPUTS
  assert state_input_names("B") == GEN_B_STATE_INPUTS
  assert state_input_names("A") == ()
  assert set(GEN_C_STATE_INPUTS) < set(GEN_B_STATE_INPUTS)


def test_unlisted_state_queue_layouts_are_rejected():
  # an unknown state tensor, even with a matching next_state output
  assert detect_generation(SSHV4_INPUTS + ["state_action_q"], SSHV4_OUTPUTS + ["next_state_action_q"]) is None
  # Gen B minus one of its state tensors is neither B nor C
  partial = [n for n in OMH_V2_INPUTS if n != "state_prev_feat"]
  assert detect_generation(partial, OMH_B_OUTPUTS) is None
  # the pre-rename image name only ever shipped with the Gen B state set
  assert detect_generation(["warped" if n == "new_img" else n for n in SSHV4_INPUTS], SSHV4_OUTPUTS) is None
  # two image inputs
  assert detect_generation(OMH_V2_INPUTS + ["warped"], OMH_B_OUTPUTS) is None


def test_extra_or_missing_inputs_are_rejected():
  assert detect_generation(TSFDO_INPUTS + ["lateral_control_params"], ["outputs"]) is None
  assert detect_generation([n for n in SSHV4_INPUTS if n != "desire"], SSHV4_OUTPUTS) is None
  assert detect_generation([n for n in OMH_V1_INPUTS if n != "action_t"], OMH_B_OUTPUTS) is None
  assert detect_generation(OMH_V2_INPUTS + ["prev_action"], OMH_B_OUTPUTS) is None
  assert detect_generation([], []) is None
