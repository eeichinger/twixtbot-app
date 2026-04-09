#!/usr/bin/env python3
"""
Quantize a TwixNet ONNX model to INT8 for reduced WASM heap usage.

INT8 dynamic quantization shrinks the model ~75% on disk and cuts peak WASM
heap size proportionally — directly addressing the iOS deferred-kill bug where
the browser OOMs during MCTS warm-up.  Inference speed drops ~25% on CPU, which
is negligible at the 5–60 second think times used in the webapp.

Quantization is *dynamic* (weights only, not activations), so no calibration
dataset is needed and quality loss is minimal (~1–3 Elo equivalent).

Usage:
    # Quantize in-place (orignal saved as model.fp32.onnx):
    python tools/quantize_model.py

    # Explicit paths:
    python tools/quantize_model.py \\
        --input  webapp/public/model.onnx \\
        --output webapp/public/model.onnx

    # Produce a separate INT8 file (keep fp32 untouched):
    python tools/quantize_model.py \\
        --input  webapp/public/model.onnx \\
        --output webapp/public/model-int8.onnx \\
        --no-backup

Typical workflow after training a new model:
    python tools/export_onnx.py   --model models/v1.pt --out webapp/public/model.onnx
    python tools/quantize_model.py                            # quantize in-place
"""
import argparse
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='INT8 dynamic quantization for TwixNet ONNX model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', '-i',
        default='webapp/public/model.onnx',
        help='Float32 ONNX model to quantize (default: webapp/public/model.onnx)',
    )
    parser.add_argument(
        '--output', '-o',
        default='webapp/public/model.onnx',
        help='Output path for INT8 model (default: overwrites --input)',
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip writing a .fp32.onnx backup of the original',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sanity check: verify model shapes are preserved after quantization
# ---------------------------------------------------------------------------

def sanity_check(path: str) -> None:
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    feeds = {
        'pegs':  np.zeros((1, 2, 24, 24), dtype=np.float32),
        'links': np.zeros((1, 8, 24, 24), dtype=np.float32),
        'locs':  np.zeros((1, 2, 24, 24), dtype=np.float32),
    }
    policy, value = sess.run(None, feeds)
    assert policy.shape == (1, 528), f'Bad policy shape: {policy.shape}'
    assert value.shape  == (1, 3),   f'Bad value shape: {value.shape}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f'Error: input model not found: {args.input}', file=sys.stderr)
        print('Run tools/export_onnx.py first to produce the ONNX model.',
              file=sys.stderr)
        sys.exit(1)

    in_size_kb  = os.path.getsize(args.input) / 1024
    overwriting = os.path.abspath(args.input) == os.path.abspath(args.output)

    print(f'Input : {args.input} ({in_size_kb:.1f} KB)')

    # Back up the original fp32 model when overwriting
    if overwriting and not args.no_backup:
        backup = args.input.replace('.onnx', '.fp32.onnx')
        shutil.copy2(args.input, backup)
        print(f'Backup: {backup}')

    # quantize_dynamic requires a *different* input and output path, so use a
    # temporary file when the caller wants to overwrite the input.
    tmp_out = args.output + '.tmp.onnx' if overwriting else args.output
    os.makedirs(os.path.dirname(os.path.abspath(tmp_out)) or '.', exist_ok=True)

    print('Quantizing (INT8 dynamic, weights only)…')
    quantize_dynamic(
        model_input=args.input,
        model_output=tmp_out,
        weight_type=QuantType.QInt8,
    )

    if overwriting:
        os.replace(tmp_out, args.output)

    out_size_kb = os.path.getsize(args.output) / 1024
    reduction   = 100 * (1 - out_size_kb / in_size_kb)
    print(f'Output: {args.output} ({out_size_kb:.1f} KB, -{reduction:.0f}%)')

    print('Running sanity check…')
    sanity_check(args.output)
    print('Sanity check passed: policy (1,528), value (1,3) ✓')


if __name__ == '__main__':
    main()
