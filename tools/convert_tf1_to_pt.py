#!/usr/bin/env python3
"""
NOTE: REQUIRES PYTHON <= 3.12 b/c of TF1

Convert a TwixBot TensorFlow 1 checkpoint to a PyTorch TwixNet .pt file.

The "six-917000" checkpoint was trained with:
  --num_conv_filters 48  --num_tower_blocks 20
  --pwin_reductions 4    --pwin_conv_padding SAME
  --pwin_triple          --num_pwin_hidden 80
  --Optimizer ADAM       (abs activation)

This script reads the checkpoint variables as numpy arrays and maps them
directly into a TwixNet model, handling the NHWC→NCHW weight transposition
required by PyTorch.

Usage:
    python tools/convert_tf1_to_pt.py \\
        --checkpoint models/six-917000 \\
        --out models/six-917000.pt \\
        [--num_filters 48] [--num_blocks 20] [--value_reductions 4]
"""
import argparse
import os
import sys

import numpy as np
import torch
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import model as mdl


# ---------------------------------------------------------------------------
# TF1 batch-norm variable → PyTorch BatchNorm parameter names
# ---------------------------------------------------------------------------
TF_TO_PT_BN = {
    'beta':             'bias',
    'gamma':            'weight',
    'moving_mean':      'running_mean',
    'moving_variance':  'running_var',
}


def load_tf_checkpoint(path):
    """Return a dict {var_name: numpy_array} from a TF1 checkpoint."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf   # only needed here; not required at inference time
    reader = tf.train.load_checkpoint(path)
    var_map = reader.get_variable_to_shape_map()
    shapes = reader.get_variable_to_shape_map()

    # num_filters: output channels of any primary conv
    num_filters = shapes['primary_pegs/Variable'][3]

    # num_blocks: count residual blocks (named block0, block1, ...)
    block_re = re.compile(r'^block(\d+)/Variable$')
    block_indices = [int(m.group(1)) for k in shapes if (m := block_re.match(k))]
    num_blocks = max(block_indices) + 1   # or just len(block_indices)

    # value head reductions: pwin/Variable_N — count the 4D (conv) ones only
    pwin_re = re.compile(r'^pwin/Variable(?:_(\d+))?$')
    pwin_convs = sum(1 for k, s in shapes.items()
                     if pwin_re.match(k) and len(s) == 4)
    value_reductions = pwin_convs

    # num_pwin_hidden: shape of the first FC weight in the value head (2D)
    fc_key = f'pwin/Variable_{value_reductions}'   # first 2D one after the convs
    num_pwin_hidden = shapes[fc_key][1]

    print(f'num_filters       = {num_filters}')
    print(f'num_blocks        = {num_blocks}')
    print(f'value_reductions  = {value_reductions}')
    print(f'num_pwin_hidden   = {num_pwin_hidden}')
    return {name: reader.get_tensor(name) for name in var_map}


def tf_conv_to_pt(w):
    """Transpose TF conv weight HWIO → PyTorch OIHW."""
    return torch.from_numpy(w.transpose(3, 2, 0, 1).copy())


def set_bn(pt_bn, tf_vars, prefix):
    """Copy TF batch-norm variables into a PyTorch BatchNorm layer."""
    for tf_suffix, pt_attr in TF_TO_PT_BN.items():
        key = f'{prefix}/{tf_suffix}'
        if key not in tf_vars:
            raise KeyError(f'Missing TF variable: {key}')
        val = torch.from_numpy(tf_vars[key].copy())
        getattr(pt_bn, pt_attr).data.copy_(val)


def main():
    parser = argparse.ArgumentParser(description='Convert TF1 checkpoint to PyTorch .pt')
    parser.add_argument('--checkpoint', '-c', default='models/six-917000',
                        help='Path to TF1 checkpoint (without extension)')
    parser.add_argument('--out', '-o', default='models/six-917000.pt',
                        help='Output .pt path')
    parser.add_argument('--num_filters',       type=int, default=48)
    parser.add_argument('--num_blocks',        type=int, default=20)
    parser.add_argument('--num_value_hidden',  type=int, default=80)
    parser.add_argument('--value_reductions',  type=int, default=4)
    parser.add_argument('--value_padding',     default='same',
                        choices=['valid', 'same'])
    args = parser.parse_args()

    print(f'Loading TF1 checkpoint: {args.checkpoint} …')
    tf_vars = load_tf_checkpoint(args.checkpoint)
    print(f'  {len(tf_vars)} variables loaded.')

    print('Building PyTorch TwixNet …')
    net = mdl.TwixNet(
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        num_value_hidden=args.num_value_hidden,
        value_reductions=args.value_reductions,
        value_padding=args.value_padding,
        activation='abs',   # TF1 model used abs()
    )
    net.eval()

    with torch.no_grad():
        # ------------------------------------------------------------------
        # Primary layer convolutions  (HWIO → OIHW)
        # ------------------------------------------------------------------
        net.primary_loc.weight.data.copy_(
            tf_conv_to_pt(tf_vars['primary_location/Variable']))
        net.primary_loc.bias.data.zero_()

        net.primary_pegs.weight.data.copy_(
            tf_conv_to_pt(tf_vars['primary_pegs/Variable']))
        net.primary_pegs.bias.data.zero_()

        net.primary_links.weight.data.copy_(
            tf_conv_to_pt(tf_vars['primary_links/Variable']))
        net.primary_links.bias.data.zero_()

        # Primary BN — TF names it "BatchNorm" (no numeric suffix)
        set_bn(net.primary_bn, tf_vars, 'BatchNorm')

        # ------------------------------------------------------------------
        # Residual blocks
        # Block i uses:
        #   conv1 weight: block{i}/Variable
        #   BN1:          BatchNorm_{1 + 2*i}
        #   conv2 weight: block{i}/Variable_1
        #   BN2:          BatchNorm_{2 + 2*i}
        # ------------------------------------------------------------------
        for i, block in enumerate(net.blocks):
            block.conv1.weight.data.copy_(
                tf_conv_to_pt(tf_vars[f'block{i}/Variable']))
            block.conv1.bias.data.zero_()

            bn1_name = f'BatchNorm_{1 + 2*i}'
            set_bn(block.bn1, tf_vars, bn1_name)

            block.conv2.weight.data.copy_(
                tf_conv_to_pt(tf_vars[f'block{i}/Variable_1']))
            block.conv2.bias.data.zero_()

            bn2_name = f'BatchNorm_{2 + 2*i}'
            set_bn(block.bn2, tf_vars, bn2_name)

        # ------------------------------------------------------------------
        # Value head
        # value_convs is a Sequential of (Conv, BN, Act) × value_reductions
        # TF BN indices: 1 + 2*num_blocks, …, 1 + 2*num_blocks + R-1
        # value FC BN:  1 + 2*num_blocks + R
        # policy BN:    1 + 2*num_blocks + R + 1
        # ------------------------------------------------------------------
        base = 1 + 2 * args.num_blocks   # = 41 for 20 blocks

        for r in range(args.value_reductions):
            group_start = r * 3     # (Conv=0, BN=1, Act=2) per group
            conv_module = net.value_convs[group_start]   # Conv or _SamePadConv2d
            bn_module   = net.value_convs[group_start + 1]

            # The actual nn.Conv2d is either direct or inside _SamePadConv2d
            actual_conv = conv_module.conv if hasattr(conv_module, 'conv') else conv_module
            # TF1 names first variable "pwin/Variable" (no _0), rest "pwin/Variable_N"
            tf_conv_key = 'pwin/Variable' if r == 0 else f'pwin/Variable_{r}'
            actual_conv.weight.data.copy_(
                tf_conv_to_pt(tf_vars[tf_conv_key]))

            bn_name = f'BatchNorm_{base + r}'
            set_bn(bn_module, tf_vars, bn_name)

        # Value FC layers — TF1 uses plain matmul (no bias); PyTorch nn.Linear
        # has bias; we set it to zero so the following BN absorbs any offset.
        fc_in_size  = tf_vars[f'pwin/Variable_{args.value_reductions}'].shape[0]
        fc_hid_size = tf_vars[f'pwin/Variable_{args.value_reductions}'].shape[1]
        assert net.value_fc1.in_features == fc_in_size, \
            f'FC1 size mismatch: model={net.value_fc1.in_features}, ckpt={fc_in_size}'

        net.value_fc1.weight.data.copy_(
            torch.from_numpy(tf_vars[f'pwin/Variable_{args.value_reductions}'].T.copy()))
        net.value_fc1.bias.data.zero_()

        set_bn(net.value_bn, tf_vars, f'BatchNorm_{base + args.value_reductions}')

        net.value_fc2.weight.data.copy_(
            torch.from_numpy(tf_vars[f'pwin/Variable_{args.value_reductions + 1}'].T.copy()))
        net.value_fc2.bias.data.zero_()

        # ------------------------------------------------------------------
        # Policy head
        # TF BN index: base + value_reductions + 1
        # ------------------------------------------------------------------
        policy_bn_idx = base + args.value_reductions + 1

        net.policy_conv1.weight.data.copy_(
            tf_conv_to_pt(tf_vars['movelogits/Variable']))
        net.policy_conv1.bias.data.zero_()

        set_bn(net.policy_bn, tf_vars, f'BatchNorm_{policy_bn_idx}')

        net.policy_conv2.weight.data.copy_(
            tf_conv_to_pt(tf_vars['movelogits/Variable_1']))
        net.policy_conv2.bias.data.zero_()

    # ------------------------------------------------------------------
    # Quick sanity check: run a zero-input forward pass
    # ------------------------------------------------------------------
    with torch.no_grad():
        dummy_pegs  = torch.zeros(1, 2, 24, 24)
        dummy_links = torch.zeros(1, 8, 24, 24)
        dummy_locs  = torch.zeros(1, 2, 24, 24)
        policy, value = net(dummy_pegs, dummy_links, dummy_locs)
    assert policy.shape == (1, 528), f'Unexpected policy shape: {policy.shape}'
    assert value.shape  == (1, 3),   f'Unexpected value shape: {value.shape}'
    print(f'Sanity check passed — policy {tuple(policy.shape)}, value {tuple(value.shape)}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(net, args.out)
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f'Saved to {args.out}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
