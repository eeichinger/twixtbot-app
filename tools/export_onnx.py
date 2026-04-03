#!/usr/bin/env python3
"""
Export TwixNet PyTorch model to ONNX format for use with onnxruntime-web.

Usage:
    python tools/export_onnx.py --model models/v1.pt --out webapp/public/model.onnx

The exported model takes three NCHW float32 inputs:
    pegs   [B, 2, 24, 24]
    links  [B, 8, 24, 24]
    locs   [B, 2, 24, 24]

And produces two outputs:
    policy [B, 528]   raw move logits
    value  [B,   3]   raw (Loss, Draw, Win) logits
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

def main():
    parser = argparse.ArgumentParser(description='Export TwixNet to ONNX')
    parser.add_argument('--model', '-m', required=True, help='Path to .pt model file')
    parser.add_argument('--out', '-o', default='webapp/public/model.onnx',
                        help='Output .onnx path (default: webapp/public/model.onnx)')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = torch.load(args.model, weights_only=False, map_location='cpu')
    model.eval()
    print(f"Model class: {model.__class__.__name__}")

    # Dummy inputs matching what nneval.py / naf.py produce: NCHW float32
    dummy = (
        torch.zeros(1, 2, 24, 24, dtype=torch.float32),   # pegs
        torch.zeros(1, 8, 24, 24, dtype=torch.float32),   # links
        torch.zeros(1, 2, 24, 24, dtype=torch.float32),   # locs
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Trace + constant-fold: fuses BatchNorm layers into preceding convolutions,
    # giving ~10-20% faster WASM inference at no accuracy cost.
    traced = torch.jit.trace(model, dummy)

    print(f"Exporting to {args.out} (opset {args.opset}) ...")
    # Use the legacy TorchScript-based exporter (dynamo=False) for compatibility
    # with traced JIT modules and to support dynamic_axes.
    torch.onnx.export(
        traced,
        dummy,
        args.out,
        input_names=['pegs', 'links', 'locs'],
        output_names=['policy', 'value'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            'pegs':   {0: 'batch'},
            'links':  {0: 'batch'},
            'locs':   {0: 'batch'},
            'policy': {0: 'batch'},
            'value':  {0: 'batch'},
        },
        dynamo=False,
    )

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Done. File size: {size_kb:.1f} KB")

    # Quick sanity check with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        feeds = {
            'pegs':  np.zeros((1,2,24,24), dtype=np.float32),
            'links': np.zeros((1,8,24,24), dtype=np.float32),
            'locs':  np.zeros((1,2,24,24), dtype=np.float32),
        }
        policy, value = sess.run(None, feeds)
        assert policy.shape == (1, 528), f"Bad policy shape: {policy.shape}"
        assert value.shape  == (1, 3),   f"Bad value shape: {value.shape}"
        print(f"Sanity check passed: policy {policy.shape}, value {value.shape}")
    except ImportError:
        print("(onnxruntime not installed; skipping sanity check)")

if __name__ == '__main__':
    main()
