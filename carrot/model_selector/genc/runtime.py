"""Gen C runtime core: comma master `modeld.py::ModelState` (2026-11-02) ported for the
carrot model selector.

Differences from comma's class:
  * artifacts come from the installed custom model dir (model JIT pkl + one warp JIT pkl
    per camera + genc_metadata.json) instead of the built-in models dir
  * no chestnut/eGPU branch — the Gen C engine runs on the integrated GPU only
  * `run()` takes the carrot main-loop signature; `prepare_only` is accepted but ignored
    because Gen C history lives inside the ONNX (every frame is a full inference, which
    is also what comma master does)
  * `reset()` in addition to `warmup()`

Only numpy, the vendored tinygrad and pure-python openpilot helpers are imported so the
class can be exercised on a PC (see pc_check.py). `activate_tinygrad()` runs at import,
so this module must be imported before anything that imports the system tinygrad.
"""
from __future__ import annotations

import base64
import json
import math
import os
import pickle
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from . import activate_tinygrad

activate_tinygrad()

from tinygrad import Tensor
from tinygrad.device import Buffer
from tinygrad.dtype import DType, dtypes
from tinygrad.engine.jit import CapturedJit
from tinygrad.engine.realize import compile_linear, lower_and_compile
from tinygrad.helpers import DEV, round_up
from tinygrad.uop.ops import UOp
# NOTE: `examples.openpilot.helpers` (load_pickle) is imported lazily inside `_load`,
# after DEV is pinned: its module body evaluates Device.DEFAULT, which would otherwise
# auto-select a device (and export DEV into the environment) before we choose one.

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

from ..config import GENC_METADATA_NAME, GENC_PKL_NAME, GENC_WARP_PKL_FMT
from .generation import GEN_C_IMAGE_INPUT

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
VISION_INPUT_NAMES = ('img', 'big_img')


def default_device() -> str:
  """tinygrad device spec for this host when the artifact metadata records none."""
  return 'QCOM' if os.path.isfile('/AGNOS') else 'CPU:LLVM'


def read_metadata(models_dir: Path) -> dict:
  return json.loads((Path(models_dir) / GENC_METADATA_NAME).read_text())


def input_view(buffer: Buffer, shape: tuple[int, ...], dtype: DType, offset: int) -> Tensor:
  view = buffer.view(math.prod(shape) * dtype.itemsize, offset).ensure_allocated()
  return Tensor(UOp.from_buffer(view, dtype)).reshape(shape)


def compile_copy(tensor: Tensor) -> Callable[[], Tensor]:
  jit = CapturedJit(tensor, compile_linear(tensor.schedule_linear()), [], [])
  return lambda: jit([], {})


class ModelState:
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, models_dir: Path, cam_w: int, cam_h: int, device: str | None = None):
    models_dir = Path(models_dir)
    self.metadata = read_metadata(models_dir)
    # The compile subprocess records its DEV; replaying the pickled JITs on another
    # backend would fail at load, so always follow the artifact.
    self.device_spec = device or self.metadata.get('compile_dev') or default_device()
    self.vision_input_names = VISION_INPUT_NAMES
    self.usbgpu = False
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    stride, y_height, uv_height, _ = get_nv12_info(cam_w, cam_h)
    self.frame_copy_size = stride * (y_height + uv_height)
    # Pin the backend for the whole process, not just the load: tinygrad resolves the
    # renderer of a device (e.g. CPU → clang vs LLVM) from DEV at every kernel compile,
    # and kernels are still compiled lazily after load (state-queue resets, copies).
    DEV.value = self.device_spec
    self._load(models_dir, cam_w, cam_h)

  def _load(self, models_dir: Path, cam_w: int, cam_h: int) -> None:
    from examples.openpilot.helpers import load_pickle  # vendored tinygrad examples; see note above
    jits = load_pickle(models_dir / GENC_PKL_NAME, out_of_band=True)
    self.model_device = jits['input_specs'][GEN_C_IMAGE_INPUT][2]
    self.input_shapes = {name: (tuple(shape), np.dtype(dtype)) for name, (shape, dtype, _) in jits['input_specs'].items()}
    self.state_pairs = {name: f'next_{name}' for name in self.input_shapes if f'next_{name}' in jits['metadata']['output_shapes']}
    # comma stores {name: slice} as a base64 pickle in the ONNX metadata; the ONNX was
    # sha256-verified against the signed manifest before it was compiled.
    self.output_slices = pickle.loads(base64.b64decode(jits['metadata']['metadata']['output_slices']))

    self.pack_inputs()
    with open(models_dir / GENC_WARP_PKL_FMT.format(w=cam_w, h=cam_h), 'rb') as f:
      self.run_warp = pickle.load(f)['run']
    self.run_warp.captured._linear = lower_and_compile(self.run_warp.captured._linear)
    self.run_model = jits['run']
    self.run_model.captured._linear = lower_and_compile(self.run_model.captured._linear)
    self.outputs = {name: Tensor(np.zeros(shape, dtype=dtype), device=device).realize()
                    for name, (shape, dtype, device) in jits['output_specs'].items()}
    for name, next_name in self.state_pairs.items():
      state = self.input_queues[name]
      self.outputs[next_name] = input_view(state._buffer(), state.shape, state.dtype, 0)
    self.copy_outputs = compile_copy(self.outputs['outputs'].to('CPU'))
    self.parser = Parser()

  def pack_inputs(self) -> None:
    # Pack host inputs into one upload (comma: reduces transfer overhead; one copy per frame here).
    self.input_queues = {name: Tensor(np.zeros(shape, dtype=dtype), device=self.model_device).realize()
                         for name, (shape, dtype) in self.input_shapes.items() if name in self.state_pairs}
    shapes = {'tfm': (2, 3, 3)} | {name: shape for name, (shape, _) in self.input_shapes.items()
                                   if name not in self.state_pairs and name != GEN_C_IMAGE_INPUT}
    npy_size = sum(round_up(math.prod(shape) * 4, 128) for shape in shapes.values())
    self.packed_input = np.zeros(npy_size + 2 * self.frame_copy_size, dtype=np.uint8)
    self.input_host = Tensor(self.packed_input, device='NPY')
    self.input_device = self.input_host.to(self.model_device)
    self.copy_inputs = compile_copy(self.input_device)
    self.copy_inputs()
    self.npy: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in shapes.items():
      self.npy[name] = np.ndarray(shape, dtype=np.float32, buffer=self.packed_input, offset=offset)
      self.input_queues[name] = input_view(self.input_device._buffer(), shape, dtypes.float32, offset)
      offset += round_up(self.npy[name].nbytes, 128)
    self.frames = self.packed_input[npy_size:].reshape(2, self.frame_copy_size)
    self.warp_inputs = {'input_frame': input_view(self.input_device._buffer(), self.frames.shape, dtypes.uint8, npy_size),
                        'M_inv': self.input_queues.pop('tfm')}

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {k: model_outputs[np.newaxis, v] for k, v in output_slices.items()}

  def run(self, bufs: Mapping[str, object], transforms: Mapping[str, np.ndarray],
          inputs: dict[str, np.ndarray], prepare_only: bool = False) -> dict[str, np.ndarray]:
    for i, key in enumerate(self.vision_input_names):
      np.copyto(self.frames[i], np.frombuffer(bufs[key].data, dtype=np.uint8, count=self.frame_copy_size))
      self.npy['tfm'][i] = transforms[key]

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']

    self.copy_inputs()
    self.input_queues[GEN_C_IMAGE_INPUT] = self.run_warp(**self.warp_inputs)
    self.run_model(output_buffers=self.outputs, **self.input_queues)
    model_output = self.copy_outputs().numpy()[0]
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))

    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    return outputs_dict

  def state_snapshot(self) -> dict[str, np.ndarray]:
    """Copies of the in-ONNX history queues (diagnostics / PC checks)."""
    return {name: self.input_queues[name].numpy().copy() for name in self.state_pairs}

  def reset(self) -> None:
    self.packed_input[:] = 0
    for key in self.state_pairs:
      self.input_queues[key].assign(0).realize()
    self.prev_desire[:] = 0

  def warmup(self) -> None:
    dummy_frames = {k: np.zeros(self.frame_copy_size, dtype=np.uint8) for k in self.vision_input_names}
    eye = np.eye(3, dtype=np.float32)
    dims = {'desire_pulse': ModelConstants.DESIRE_LEN, 'traffic_convention': 2, 'action_t': 2}
    self.run(dummy_frames, dict.fromkeys(self.vision_input_names, eye), {k: np.zeros(v, dtype=np.float32) for k, v in dims.items()})
    self.reset()
