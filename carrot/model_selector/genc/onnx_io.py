#!/usr/bin/env python3
"""Read ONNX graph input/output names+shapes and metadata without loading weights.

Runs with the vendored tinygrad (its protobuf parser) so the manager process
never imports tinygrad itself. Used by the installer to detect the model
generation before choosing a compile path.

    python3 onnx_io.py /data/models_tmp/driving_supercombo.onnx [--json /path/out.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TINYGRAD_DIR = _HERE / "tinygrad_repo"
if str(_TINYGRAD_DIR) not in sys.path:
    sys.path.insert(0, str(_TINYGRAD_DIR))

from tinygrad.nn.onnx import OnnxPBParser

_FIELD_METADATA_PROPS = 14  # ModelProto.metadata_props
_FIELD_GRAPH = 7            # ModelProto.graph
_FIELD_GRAPH_INPUT = 11     # GraphProto.input
_FIELD_GRAPH_OUTPUT = 12    # GraphProto.output


def read_onnx_io(path: Path) -> dict:
    """Return {"inputs": {name: [dims]}, "outputs": {...}, "metadata": {...}, "output_slices": {name: [start, stop]}}."""
    parser = OnnxPBParser(path, load_external_data=False)
    metadata: dict[str, str] = {}
    inputs: dict[str, list[int]] = {}
    outputs: dict[str, list[int]] = {}
    for field, wire_type in parser._parse_message(parser.reader.len):
        if field == _FIELD_METADATA_PROPS:
            entry = parser._parse_StringStringEntryProto()
            metadata[entry["key"]] = entry["value"]
        elif field == _FIELD_GRAPH:
            for gfield, gwire in parser._parse_message(parser._decode_end_pos()):
                if gfield in (_FIELD_GRAPH_INPUT, _FIELD_GRAPH_OUTPUT):
                    value = parser._parse_ValueInfoProto()
                    parsed = value.get("parsed_type")
                    shape = [int(d) for d in parsed.shape] if parsed is not None else []
                    (inputs if gfield == _FIELD_GRAPH_INPUT else outputs)[value["name"]] = shape
                else:
                    parser.reader.skip_field(gwire)
        else:
            parser.reader.skip_field(wire_type)

    output_slices: dict[str, list[int | None]] = {}
    if "output_slices" in metadata:
        # comma stores a base64 pickle of {name: slice}; the ONNX itself is sha256-verified
        # against the signed manifest before it ever reaches this point.
        raw = pickle.loads(base64.b64decode(metadata["output_slices"]))
        output_slices = {k: [v.start, v.stop] for k, v in raw.items()}
    return {
        "inputs": inputs,
        "outputs": outputs,
        "metadata": {k: v for k, v in metadata.items() if k != "output_slices"},
        "output_slices": output_slices,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--json", type=Path, help="write the result here instead of stdout")
    args = ap.parse_args()
    result = read_onnx_io(args.onnx)
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
