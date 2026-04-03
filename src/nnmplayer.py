#! /usr/bin/env python
import argparse
import collections
import numpy
import random

import naf
import nnclient
import nneval
import nnmcts
import swapmodel
import twixt


class Player:
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', None)
        self.resource = kwargs.get('resource', None)
        self.add_noise = float(kwargs.get('add_noise', 0))
        self.temperature = float(kwargs.get('temperature', 0))
        self.num_trials = int(kwargs.get('trials', 100))
        self.verbosity = int(kwargs.get('verbosity', 0))
        self.random_rotation = int(kwargs.get('random_rotation', 1))
        self.smart_root = int(kwargs.get('smart_root', 1))
        self.use_swap = int(kwargs.get('use_swap', 0))

        if self.temperature not in (0.0, 0.5, 1.0):
            raise ValueError("Unsupported temperature")

        if self.model:
            nneval_ = nneval.NNEvaluater(self.model)
            def nnfunc(game):
                nips = naf.NetInputs(game)
                if self.random_rotation:
                    rot = random.randint(0, 3)
                    nips.rotate(rot)
                else:
                    rot = 0
                pws, mls = nneval_.eval_one(nips)
                # eval_one returns (1,3) and (1,528) — index into batch dim
                pw = naf.three_to_one(pws[0])
                ml = mls[0]
                return pw, naf.rotate_policy_array(ml, rot)
        elif self.resource:
            nncli = kwargs.get("resources").get(self.resource)
            assert nncli is not None
            def nnfunc(game):
                nips = naf.NetInputs(game)
                rot = random.randint(0, 3) if self.random_rotation else 0
                nips.rotate(rot)
                pw, ml_r = nncli.eval(nips)
                if not isinstance(pw, numpy.floating):
                    assert pw.shape[0] == 3
                    pw = naf.three_to_one(pw)
                ml_0 = naf.rotate_policy_array(ml_r, rot)
                return pw, ml_0
        else:
            raise Exception("Specify model or resource")

        self.nm = nnmcts.NeuralMCTS(nnfunc, add_noise=self.add_noise, smart_root=self.smart_root, verbosity=self.verbosity)

    def pick_move(self, game):
        if self.use_swap and len(game.history) < 2:
            if len(game.history) == 0:
                self.report = "swapmodel"
                return swapmodel.choose_first_move()
            elif swapmodel.want_swap(game.history[0]):
                self.report = "swapmodel"
                return "swap"
            # else didn't want swap so compute a regular move

        N = self.nm.mcts(game, self.num_trials)
        self.report = self.nm.report

        # When a forcing win or forcing draw move is found, there's no policy array
        if isinstance(N, (str, twixt.Point)):
            return N

        if self.temperature == 0.0:
            mx = N.max()
            weights = numpy.where(N == mx, 1.0, 0.0)
        elif self.temperature == 1.0:
            weights = N
        elif self.temperature == 0.5:
            weights = N ** 2
        if self.verbosity >= 2:
            print("weights=", weights)
        index = numpy.random.choice(numpy.arange(len(weights)), p=weights/weights.sum())

        return naf.policy_index_point(game, index), N
