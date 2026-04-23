#!/usr/bin/env python
"""
Training loop for TwixNet — PyTorch replacement of the TF1 original.

Usage:
    python train.py --model path/to/model.pt [options] selfplay_file...

The model file is loaded with torch.load() and saved back to the same path
after training. Use model.py's TwixNet + torch.save(model, path) to create
initial model files.
"""
import collections
import numpy
import os
import random
import re
import sys
import time

import naf
import twixt
import wrs

import torch
import torch.nn.functional as F

from model import TwixNet


def when():
    return time.strftime("%Y%m%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Data preparation helpers (importable for testing)
# ---------------------------------------------------------------------------

def make_policy_target(N, temperature):
    """Convert MCTS visit counts to a softmax policy target.

    Args:
        N (np.ndarray): visit counts, shape (NUM_MOVES,)
        temperature (float): 0.0 (argmax), 0.5 (square root), 1.0 (linear)

    Returns:
        np.ndarray float32, sums to 1.0
    """
    if temperature == 0.5:
        nup = numpy.square(N.astype(numpy.float32))
    elif temperature == 0.0:
        nup = numpy.where(N == N.max(), 1.0, 0.0).astype(numpy.float32)
    elif temperature == 1.0:
        nup = N.astype(numpy.float32)
    else:
        raise ValueError(f"Bad temperature: {temperature}")
    nsum = nup.sum()
    assert nsum > 0, nsum
    return nup / nsum


def make_value_target(z):
    """Convert outcome z ∈ {-1, 0, 1} to integer class index ∈ {0, 1, 2}."""
    return int(z) + 1


# ---------------------------------------------------------------------------
# File / selector helpers
# ---------------------------------------------------------------------------

class FileInfo:
    """Metadata + open handle for a self-play binary file."""
    def __init__(self, filename):
        self.name = filename
        stat = os.stat(filename)
        self.count = stat.st_size // naf.LearningState.NUM_BYTES
        self.f = open(filename, 'rb')


_weight_re = re.compile(r'w=([0-9]*\.?[0-9]+)')


def load_selector(selector, name):
    """Recursively add a file or directory of self-play data to *selector*.

    A basename matching 'w=<float>' changes the default basket weight for
    subsequent adds in this subtree.

    Returns:
        (num_files, num_rows)
    """
    num_files = 0
    num_rows = 0
    mo = _weight_re.match(os.path.basename(name))
    if mo:
        selector.set_default_weight(float(mo.group(1)))
        return 0, 0
    if os.path.isdir(name):
        for sub in sorted(os.listdir(name)):
            f, r = load_selector(selector, os.path.join(name, sub))
            num_files += f
            num_rows += r
    elif os.path.isfile(name):
        fi = FileInfo(name)
        selector.add_basket(fi.count, obj=fi)
        num_files += 1
        num_rows += fi.count
    return num_files, num_rows


def sample_learning_state(selector):
    """Draw one valid LearningState uniformly from *selector*.

    Retries silently on corrupt records or all-zero visit counts.
    """
    while True:
        _, y, fi = selector.random_item()
        assert 0 <= y < fi.count, (y, fi)
        fi.f.seek(y * naf.LearningState.NUM_BYTES)
        b = fi.f.read(naf.LearningState.NUM_BYTES)
        assert len(b) == naf.LearningState.NUM_BYTES
        try:
            ts = naf.LearningState.from_bytes(b, f'{fi.name}:{y}')
        except ValueError:
            print(f'Errored on {fi.name}:{y}!!', file=sys.stderr)
            continue
        if ts.N.any():
            r = random.randint(0, 3)
            ts.nips.rotate(r)
            ts.N = naf.rotate_policy_array(ts.N, r)
            return ts


def read_from_holdout(name):
    """Yield every LearningState from a holdout file or directory tree."""
    if os.path.isdir(name):
        for sub in sorted(os.listdir(name)):
            yield from read_from_holdout(os.path.join(name, sub))
    elif os.path.isfile(name):
        lsnb = naf.LearningState.NUM_BYTES
        with open(name, 'rb') as f:
            all_bytes = f.read()
        for i in range(len(all_bytes) // lsnb):
            yield naf.LearningState.from_bytes(
                all_bytes[i*lsnb:(i+1)*lsnb], f'{name}:{i}')


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def prepare_batch(learning_states, temperature, policy_epsilon=0.0, device='cpu'):
    """Convert a list of LearningStates into model-ready tensors.

    Returns:
        pegs_t   [B, 2, H, W]   float32
        links_t  [B, 8, H, W]   float32
        locs_t   [B, 2, H, W]   float32
        p_target [B, NUM_MOVES] float32  (normalised policy distribution)
        v_target [B]            long     (class index: 0=Loss, 1=Draw, 2=Win)
    """
    S = twixt.Game.SIZE
    B = len(learning_states)
    pegs_arr  = numpy.zeros((B, S, S, 2), dtype=numpy.float32)
    links_arr = numpy.zeros((B, S, S, 8), dtype=numpy.float32)
    locs_arr  = numpy.zeros((B, S, S, 2), dtype=numpy.float32)
    p_arr     = numpy.zeros((B, naf.LearningState.NUM_MOVES), dtype=numpy.float32)
    v_arr     = numpy.zeros(B, dtype=numpy.int64)

    for i, ls in enumerate(learning_states):
        pegs, links, locs = ls.nips.to_input_arrays()
        pegs_arr[i]  = pegs
        links_arr[i] = links
        locs_arr[i]  = locs
        p_arr[i] = make_policy_target(ls.N, temperature)
        if policy_epsilon > 0 and temperature == 0.0:
            p_arr[i] += policy_epsilon
            p_arr[i] /= p_arr[i].sum()
        v_arr[i] = make_value_target(ls.z)

    def nhwc_to_nchw(arr):
        t = torch.from_numpy(arr)
        return t.permute(0, 3, 1, 2).to(device)

    return (nhwc_to_nchw(pegs_arr),
            nhwc_to_nchw(links_arr),
            nhwc_to_nchw(locs_arr),
            torch.from_numpy(p_arr).to(device),
            torch.from_numpy(v_arr).to(device))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Wraps model + optimiser for one training or evaluation step.

    Args:
        model:         TwixNet (or compatible nn.Module)
        learning_rate: initial SGD learning rate
        device:        torch device string
    """

    def __init__(self, model, learning_rate=0.01, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=0.9, weight_decay=1e-4)
        self.step = 0

    def set_learning_rate(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def train_step(self, pegs, links, locs, p_target, v_target):
        """One gradient update.

        Returns:
            (total_loss, policy_loss, value_loss) — Python floats
        """
        self.model.train()
        self.optimizer.zero_grad()
        policy_logits, value_logits = self.model(pegs, links, locs)
        l1 = _policy_loss(policy_logits, p_target)
        l2 = F.cross_entropy(value_logits, v_target)
        total = l1 + l2
        total.backward()
        self.optimizer.step()
        self.step += 1
        return total.item(), l1.item(), l2.item()

    def eval_step(self, pegs, links, locs, p_target, v_target):
        """Forward pass only (no gradient update).

        Returns:
            (total_loss, policy_loss, value_loss) — Python floats
        """
        self.model.eval()
        with torch.no_grad():
            policy_logits, value_logits = self.model(pegs, links, locs)
        l1 = _policy_loss(policy_logits, p_target)
        l2 = F.cross_entropy(value_logits, v_target)
        return float(l1 + l2), float(l1), float(l2)


def _policy_loss(logits, target):
    """Soft-target cross-entropy: -∑ target * log_softmax(logits), averaged over batch."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Train TwixNet on self-play data')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to model .pt file (loaded and saved in-place)')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                        help='torch device string, e.g. "cpu" or "cuda:0"')
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--num_batches', '-n', type=int, default=1000)
    parser.add_argument('--learning_rate', '-L', type=float, default=0.01)
    parser.add_argument('--decay_rate', '-D', type=float, default=1.0)
    parser.add_argument('--temperature', '-t', type=float, default=0.5)
    parser.add_argument('--policy_epsilon', '-P', type=float, default=0.01)
    parser.add_argument('--save_after', '-S', type=int, default=0)
    parser.add_argument('--holdout', '-H', type=str, required=False)
    parser.add_argument('--holdout_fraction', '-F', type=float, default=1.0)
    parser.add_argument('spfiles', metavar='S', type=str, nargs='*',
                        help='self-play binary log files')
    args = parser.parse_args(argv)

    print(f'Loading model: {args.model}')
    model = torch.load(args.model, weights_only=False, map_location=args.device)
    trainer = Trainer(model, learning_rate=args.learning_rate, device=args.device)

    if args.num_batches > 0:
        selector = wrs.WeightedRandomSelector()
        num_files = num_rows = 0
        for fn in args.spfiles:
            f, r = load_selector(selector, fn)
            num_files += f
            num_rows += r
        print(f'Scanned: #files={num_files}  #rows={num_rows}')
    else:
        print('no files scanned, #batches = 0')

    def run_holdout():
        rng = random.Random(int.from_bytes(os.urandom(4), 'big'))
        mean_total = mean_l1 = mean_l2 = 0.0
        n = 0
        for ls in read_from_holdout(args.holdout):
            if rng.random() < args.holdout_fraction:
                continue
            batch = prepare_batch([ls], args.temperature, args.policy_epsilon)
            _, l1, l2 = trainer.eval_step(*batch)
            n += 1
            mean_total += ((l1 + l2) - mean_total) / n
            mean_l1 += (l1 - mean_l1) / n
            mean_l2 += (l2 - mean_l2) / n
        print(f'holdout {n} batch{"es" if n != 1 else ""}')
        print(f'loss={mean_total:.4g}  (policy={mean_l1:.4g}  value={mean_l2:.4g})')

    if args.holdout:
        run_holdout()

    XX = numpy.zeros((2, 2))
    XY = numpy.zeros(2)
    prev_loss = None

    start_time = time.time()
    last_progress = start_time
    PROGRESS_INTERVAL = 30.0
    print(f'{when()} training start: {args.num_batches} batches, batch_size={args.batch_size}, device={args.device}')

    for b in range(args.num_batches):
        print(f'batch {b}')
        batch_states = [sample_learning_state(selector)
                        for _ in range(args.batch_size)]
        batch = prepare_batch(batch_states, args.temperature, args.policy_epsilon, device=args.device)
        total, l1, l2 = trainer.train_step(*batch)

        x = numpy.array([1, b])
        XX += numpy.outer(x, x)
        XY += x * total
        if b > 2:
            betas = numpy.linalg.solve(XX, XY)
            print(f'loss={total:.4g}  slope={betas[1]:.4g}')
        else:
            print(f'loss={total:.4g}')
        print(f'policy={l1:.4g}  value={l2:.4g}')

        if prev_loss is not None and total > prev_loss:
            new_lr = trainer.get_learning_rate() * args.decay_rate
            trainer.set_learning_rate(new_lr)
            print(f'reduce learning rate to {new_lr:.4g}')
        prev_loss = total

        if args.save_after and (b + 1) % args.save_after == 0:
            print('save it')
            torch.save(model, args.model)

        now = time.time()
        if now - last_progress >= PROGRESS_INTERVAL:
            elapsed = now - start_time
            done = b + 1
            rate = done / elapsed if elapsed > 0 else 0
            remaining = args.num_batches - done
            eta = remaining / rate if rate > 0 else 0
            print(f'{when()} progress {done}/{args.num_batches} | elapsed {elapsed:.0f}s | {rate:.2f} batch/s | ETA {eta:.0f}s | loss={total:.4g}')
            last_progress = now

        sys.stdout.flush()

    if args.num_batches > 0:
        print('save it')
        torch.save(model, args.model)
        elapsed = time.time() - start_time
        rate = args.num_batches / elapsed if elapsed > 0 else 0
        samples = args.num_batches * args.batch_size
        print(f'{when()} training done: {args.num_batches} batches in {elapsed:.1f}s ({elapsed/60:.1f} min) | {rate:.2f} batch/s | {samples/elapsed:.0f} samples/s')
        if args.holdout:
            run_holdout()


if __name__ == '__main__':
    main()
