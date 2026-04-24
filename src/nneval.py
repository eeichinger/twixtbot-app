"""
NNEvaluater — PyTorch replacement for the original TF1 nneval.py.

Wraps a TwixNet model and provides the batch-evaluation interface expected by
nns.py and other callers:

    eval_many_prepare(nips)         -> (pegs, links, locs)  [NHWC numpy]
    eval_many_doit(pegs, links, locs) -> (pws, mls)          [numpy arrays]
    eval_many(nips)                 -> (pws, mls)
    eval_one(nip)                   -> (pws, mls)
    pwin_size()                     -> 3  (Loss/Draw/Win)

All inputs arrive as NetInputs objects (HWC layout, uint8); internally
converted to NCHW float32 tensors for the model.
"""
import numpy
import torch

import naf
import model as mdl


class NNEvaluater:
    """Evaluate game positions with a TwixNet model.

    Args:
        model_or_path: Either a TwixNet instance or a path (str) to a
                       state-dict file saved with torch.save().
        device (str):  Torch device string, e.g. 'cpu' or 'cuda'.
    """

    def __init__(self, model_or_path, device='cpu', compiled=False, fp16=False):
        self.device = device
        self.fp16 = fp16 and device != 'cpu'
        if isinstance(model_or_path, str):
            net = torch.load(model_or_path, map_location=device,
                             weights_only=False)
        else:
            net = model_or_path
        self.model = net.eval().to(device)
        if compiled:
            self.model = torch.compile(self.model, mode='default')

    def pwin_size(self):
        """Return the number of value outputs — always 3 (Loss/Draw/Win)."""
        return 3

    def eval_many_prepare(self, nips):
        """Convert an iterable of NetInputs to stacked NHWC numpy arrays.

        Returns:
            pegs  (N, H, W, 2)  uint8
            links (N, H, W, 8)  uint8
            locs  (N, H, W, 2)  float32
        """
        pegs_list, links_list, locs_list = [], [], []
        for n in nips:
            p, l, lx = n.to_input_arrays()
            pegs_list.append(p)
            links_list.append(l)
            locs_list.append(lx)
        return (numpy.array(pegs_list),
                numpy.array(links_list),
                numpy.array(locs_list))

    def eval_many_doit(self, pegs, links, locs):
        """Run inference on pre-prepared NHWC numpy arrays.

        Args:
            pegs  (N, H, W, 2)
            links (N, H, W, 8)
            locs  (N, H, W, 2)

        Returns:
            pws (N, 3)     float32 — value logits (Loss, Draw, Win)
            mls (N, 528)   float32 — policy logits
        """
        def to_tensor(arr):
            t = torch.from_numpy(numpy.asarray(arr, dtype=numpy.float32))
            return t.permute(0, 3, 1, 2).to(self.device)  # NHWC → NCHW

        pegs_t  = to_tensor(pegs)
        links_t = to_tensor(links)
        locs_t  = to_tensor(locs)

        with torch.no_grad():
            if self.fp16:
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    policy_logits, value_logits = self.model(pegs_t, links_t, locs_t)
            else:
                policy_logits, value_logits = self.model(pegs_t, links_t, locs_t)

        mls = policy_logits.cpu().numpy().astype(numpy.float32)  # (N, 528)
        pws = value_logits.cpu().numpy().astype(numpy.float32)   # (N, 3)
        return pws, mls

    def eval_many(self, nips):
        """Evaluate a list of NetInputs end-to-end.

        Returns:
            pws (N, 3)    float32
            mls (N, 528)  float32
        """
        pegs, links, locs = self.eval_many_prepare(nips)
        return self.eval_many_doit(pegs, links, locs)

    def eval_one(self, nip):
        """Evaluate a single NetInputs.

        Returns:
            pws (1, 3)    float32
            mls (1, 528)  float32
        """
        p, l, lx = nip.to_input_arrays()
        pegs  = numpy.array([p])
        links = numpy.array([l])
        locs  = numpy.array([lx])
        return self.eval_many_doit(pegs, links, locs)
