#! /usr/bin/env python
""" Neural Net Server """
# python
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
import argparse
import numpy
import sys
import os
import torch
torch.set_num_threads(os.cpu_count())  # or a specific core count

# mine
import naf
import nneval
import smmpp
import twixt

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--location", type=str, required=True)
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-m", "--model", type=str, required=False)
parser.add_argument("-k", "--kill", action='store_true')
parser.add_argument("-c", "--capacity", type=int, default=200)
parser.add_argument("--milestone_step", type=int, default=10000)
parser.add_argument("--compile", action='store_true', dest='compile_model')
parser.add_argument("--fp16", action='store_true')
args = parser.parse_args()

if args.kill:
    c = smmpp.Client(args.location, smmpp.SUICIDE_CODE)
    sys.exit(0)

if args.model is None:
    with open("/data/twixt/models/best", "r") as f:
        model = f.read().strip()
        print("Model is:", model)
else:
    model = args.model


ne = nneval.NNEvaluater(model, device=args.device, compiled=args.compile_model, fp16=args.fp16)

class NNServer(smmpp.Server):
    def run_jobs(self, jobs):
        nips = list(map(naf.NetInputs, jobs))
        pegs, links, locs = ne.eval_many_prepare(nips)
        pws, mls = ne.eval_many_doit(pegs, links, locs)
        outs = []
        for i in range(len(jobs)):
            b = numpy.array([pws[i]], dtype=numpy.float32).tobytes()
            b += mls[i].astype(numpy.float32).tobytes()
            outs.append(b)
        return outs

num_outputs = twixt.Game.SIZE * (twixt.Game.SIZE-2) + ne.pwin_size()

server = NNServer(args.location, args.capacity, naf.NetInputs.EXPANDED_SIZE,
    4*num_outputs, args.milestone_step)
server.run()
