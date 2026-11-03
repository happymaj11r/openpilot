"""Generation detection must be exact: only Gen A and Gen C I/O sets are accepted."""
from __future__ import annotations

from carrot.model_selector.genc.generation import GEN_A_INPUTS, GEN_C_INPUTS, detect_generation

# Real shapes seen in the openpilot-models catalog (2026-11-03).
TSFDO_INPUTS = ["img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"]
SSHV4_INPUTS = ["new_img", "desire", "traffic_convention", "action_t", "state_img_q", "state_desire_q", "state_feat_q"]
SSHV4_OUTPUTS = ["outputs", "next_state_img_q", "next_state_desire_q", "next_state_feat_q"]
OMH_V1_INPUTS = ["warped", "state_img_q", "state_big_img_q", "desire", "state_desire_q", "state_prev_feat",
                 "state_feat_q", "traffic_convention", "action_t"]
OMH_V2_INPUTS = ["new_img", "state_img_q", "state_big_img_q", "desire", "state_desire_q", "state_prev_feat",
                 "state_feat_q", "traffic_convention", "action_t"]


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


def test_gen_b_variants_are_rejected():
  outputs = ["outputs", "next_state_img_q", "next_state_big_img_q", "next_state_desire_q", "next_state_feat_q",
             "next_state_prev_feat"]
  assert detect_generation(OMH_V1_INPUTS, outputs) is None
  assert detect_generation(OMH_V2_INPUTS, outputs) is None


def test_extra_or_missing_inputs_are_rejected():
  assert detect_generation(TSFDO_INPUTS + ["lateral_control_params"], ["outputs"]) is None
  assert detect_generation([n for n in SSHV4_INPUTS if n != "desire"], SSHV4_OUTPUTS) is None
  assert detect_generation([], []) is None
