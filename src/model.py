"""
TwixNet — PyTorch implementation of the TwixBot neural network.

Mirrors the architecture from mkbig.py (TensorFlow 1) with these changes:
  - Activation: GELU (default) instead of abs()
  - Policy head: correct 528-move output (fixes the 529 off-by-one in original)
  - Value head: always 3-class (Loss/Draw/Win) logits
  - Uses PyTorch BatchNorm instead of tf.contrib.layers.batch_norm

Architecture overview:
  Primary layer: three input streams (pegs 5×5, links 4×4, location 1×1)
                 summed (not concatenated) → BN → activation
  Residual tower: N blocks of (5×5 conv → BN → act → 5×5 conv → BN → act + skip)
  Policy head:   1×1 conv → BN → act → 1×1 conv → slice border rows → flatten (528)
  Value head:    N× strided conv (5×5 s2) → flatten → FC → BN → act → FC(3)
                 value_padding='valid' (default) or 'same' (matches TF1 original)
"""
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

import naf
import twixt

SIZE = twixt.Game.SIZE
NUM_MOVES = SIZE * (SIZE - 2)   # 528 = 24 * 22


def _make_activation(name):
    if name == 'gelu':
        return nn.GELU()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'abs':
        return _AbsActivation()
    else:
        raise ValueError(f"Unknown activation: {name!r}. Choose gelu/silu/relu/abs.")


class _AbsActivation(nn.Module):
    """abs() activation — matches the original TF model exactly."""
    def forward(self, x):
        return x.abs()


class _SamePadConv2d(nn.Module):
    """Conv2d with TensorFlow-style SAME padding (works with stride > 1)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=False)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        _, _, h, w = x.shape
        out_h = math.ceil(h / self.stride)
        out_w = math.ceil(w / self.stride)
        pad_h = max(0, (out_h - 1) * self.stride + self.kernel_size - h)
        pad_w = max(0, (out_w - 1) * self.stride + self.kernel_size - w)
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                       pad_h // 2, pad_h - pad_h // 2])
        return self.conv(x)


class ResidualBlock(nn.Module):
    """One residual block: conv → BN → act → conv → BN → act, plus skip."""

    def __init__(self, num_filters, activation='gelu'):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding='same')
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding='same')
        self.bn2   = nn.BatchNorm2d(num_filters)
        self.act   = _make_activation(activation)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.act(self.bn2(self.conv2(h)))
        return h + x


class TwixNet(nn.Module):
    """Full TwixBot network.

    Args:
        num_filters (int):       Convolutional filter count throughout. Default 40.
        num_blocks (int):        Number of residual blocks. Default 12.
        num_value_hidden (int):  Hidden units in value FC layer. Default 80.
        value_reductions (int):  Strided-conv steps in value head. Default 2.
        activation (str):        One of 'gelu' (default), 'silu', 'relu', 'abs'.

    Inputs (NCHW float32 tensors):
        pegs  [B, 2, H, W]   — current player's and opponent's peg planes
        links [B, 8, H, W]   — 8 link-direction planes (shifted toward endpoints)
        locs  [B, 2, H, W]   — x and y coordinate ramps in [0, 1)

    Outputs:
        policy_logits [B, 528]   — un-normalised move logits
        value_logits  [B, 3]     — un-normalised (Loss, Draw, Win) logits
    """

    def __init__(self, num_filters=40, num_blocks=12, num_value_hidden=80,
                 value_reductions=2, value_padding='valid', activation='gelu'):
        super().__init__()
        F = num_filters

        # ------------------------------------------------------------------
        # Primary layer — three separate input convolutions, then summed
        # ------------------------------------------------------------------
        self.primary_loc   = nn.Conv2d(2, F, kernel_size=1, padding='same')
        self.primary_pegs  = nn.Conv2d(2, F, kernel_size=5, padding='same')
        self.primary_links = nn.Conv2d(8, F, kernel_size=4, padding='same')
        self.primary_bn    = nn.BatchNorm2d(F)
        self.primary_act   = _make_activation(activation)

        # ------------------------------------------------------------------
        # Residual tower
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            ResidualBlock(F, activation=activation)
            for _ in range(num_blocks)
        ])

        # ------------------------------------------------------------------
        # Policy head: 1×1 → BN → act → 1×1 → slice → reshape
        # ------------------------------------------------------------------
        self.policy_conv1 = nn.Conv2d(F, 2, kernel_size=1)
        self.policy_bn    = nn.BatchNorm2d(2)
        self.policy_act   = _make_activation(activation)
        self.policy_conv2 = nn.Conv2d(2, 1, kernel_size=1)

        # ------------------------------------------------------------------
        # Value head: strided convs → flatten → FC → BN → act → FC
        # ------------------------------------------------------------------
        spatial = SIZE
        value_conv_layers = []
        in_ch = F
        for _ in range(value_reductions):
            if value_padding == 'same':
                value_conv_layers.append(_SamePadConv2d(in_ch, F, kernel_size=5, stride=2))
                spatial = math.ceil(spatial / 2)   # SAME padding formula
            else:
                value_conv_layers.append(nn.Conv2d(in_ch, F, kernel_size=5, stride=2, padding=0, bias=False))
                spatial = (spatial - 5) // 2 + 1   # VALID padding formula
            value_conv_layers.append(nn.BatchNorm2d(F))
            value_conv_layers.append(_make_activation(activation))
            in_ch = F
        self.value_convs = nn.Sequential(*value_conv_layers)

        flat_size = F * spatial * spatial
        self.value_fc1  = nn.Linear(flat_size, num_value_hidden)
        self.value_bn   = nn.BatchNorm1d(num_value_hidden)
        self.value_act  = _make_activation(activation)
        self.value_fc2  = nn.Linear(num_value_hidden, 3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pegs, links, locs):
        """
        pegs:  [B, 2, H, W]
        links: [B, 8, H, W]
        locs:  [B, 2, H, W]
        Returns: (policy_logits [B, 528], value_logits [B, 3])
        """
        # Primary: sum three streams, BN, activate
        h = (self.primary_pegs(pegs.float())
             + self.primary_links(links.float())
             + self.primary_loc(locs.float()))
        h = self.primary_act(self.primary_bn(h))

        # Residual tower
        for block in self.blocks:
            h = block(h)

        # Policy head
        p = self.policy_act(self.policy_bn(self.policy_conv1(h)))
        p = self.policy_conv2(p)           # [B, 1, H, W]
        # Slice away the two border rows (row 0 and row H-1), then flatten.
        # This gives the 22 playable rows × 24 cols = 528 legal positions.
        p = p[:, 0, 1:-1, :]              # [B, 22, W]
        policy_logits = p.reshape(p.shape[0], -1)   # [B, 528]

        # Value head
        v = self.value_convs(h)
        v = v.flatten(start_dim=1)
        v = self.value_act(self.value_bn(self.value_fc1(v)))
        value_logits = self.value_fc2(v)   # [B, 3]

        return policy_logits, value_logits


# ---------------------------------------------------------------------------
# sap factory — wraps TwixNet as the score-and-policy callable for NeuralMCTS
# ---------------------------------------------------------------------------

def make_sap(model, device='cpu'):
    """Return a sap(game) → (score: float, policy: np.ndarray[528]) callable.

    The returned function:
      1. Encodes the game position via naf.NetInputs.to_input_arrays()
      2. Runs a single forward pass through model (no_grad, eval mode)
      3. Returns:
           score  — scalar in (-1, 1) via naf.three_to_one(value_logits)
           policy — raw policy logits as numpy array of shape (528,)
    """
    model.eval()

    def sap(game):
        ni = naf.NetInputs(game)
        pegs_np, links_np, locs_np = ni.to_input_arrays()

        # naf returns HWC numpy arrays; convert to NCHW float32 tensors
        def to_tensor(arr):
            # arr shape: (H, W, C) or already float
            t = torch.from_numpy(numpy.array(arr, dtype=numpy.float32))
            return t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)

        pegs_t  = to_tensor(pegs_np)
        links_t = to_tensor(links_np)
        locs_t  = to_tensor(locs_np)

        with torch.no_grad():
            policy_logits, value_logits = model(pegs_t, links_t, locs_t)

        policy_np = policy_logits[0].cpu().numpy()                # shape (528,)
        value_np  = value_logits[0].cpu().numpy().astype(float)   # shape (3,) as float

        # Convert 3-class (Loss, Draw, Win) logits to a scalar score
        score = naf.three_to_one(value_np)

        return float(score), policy_np

    return sap
