#!/usr/bin/env python3
"""Which sampling temperature plays most like Andrew, across ALL behaviours?

Careful about the metric. Mean probability on his played move is maximised as
T -> 0, because a deterministic argmax scores 1 whenever it is right; that
would "prove" the bot should never vary, which is obviously wrong. Two honest
measures instead:

  LOG-LIKELIHOOD of his actual moves. This is a proper scoring rule - it
  punishes both over- and under-confidence - and it is what the model was
  fitted to maximise at T=1.

  FEATURE-MOMENT MISMATCH across all 58 features. For each feature, compare
  the average value over the moves HE played against the average the bot's
  distribution puts on it. This is the direct answer to "does it do the things
  I do, as often as I do them" - captures, checks, retreats, king walks, edge
  pawns, castling, all of it at once. A fitted conditional logit matches these
  moments exactly at T=1 (the moment-matching property of the exponential
  family), so this measurement doubles as a check that the fit is sound.

Also reports expected centipawn loss at each T, since that is the carefulness
he is trading against.

Run from the repo root: python3 pipeline/temp_sweep.py
"""

import json
import os
import sys

import numpy as np

from features_v9 import FEATURE_NAMES, N_FEATURES
from train_v9 import game_folds, lbfgs, nll_and_grad

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO_ROOT, "pipeline", "examples-v9.npz")
RAW = os.path.join(REPO_ROOT, "pipeline", "raw-cache.json")
TEMPS = (0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25)


def softmax(z, M, temp):
    z = np.where(M, z / temp, -1e9)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    e = np.where(M, e, 0.0)
    return e / e.sum(axis=1, keepdims=True)


def main():
    d = np.load(EXAMPLES)
    X, M, y, g = d["X"].astype(np.float64), d["M"], d["y"].astype(int), d["g"]
    N, K, D = X.shape

    # per-candidate centipawn loss, for the carefulness column
    raw = json.load(open(RAW))["positions"]
    loss = np.zeros((N, K))
    i = 0
    for p in raw:
        ucis = [c["uci"] for c in p["cands"]][:K]
        if p["played"] not in ucis:
            continue
        cps = [c["cp"] for c in p["cands"][:K]]
        best = max(cps)
        for k, cp in enumerate(cps):
            loss[i, k] = min(max(0, best - cp), 1000)
        i += 1
    assert i == N, f"{i} vs {N} - raw cache and examples are out of step"

    flat = X[M]
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = X / scale
    folds = game_folds(g, k=5)

    # out-of-fold predictions, so nothing is scored on data it was fitted to
    z_oof = np.zeros((N, K))
    for tr, te in folds:
        w, _, _, _ = lbfgs(lambda v: nll_and_grad(v, Xs[tr], M[tr], y[tr], 3e-5), np.zeros(D))
        z_oof[te] = Xs[te] @ w

    idx = np.arange(N)
    his_moments = X[idx, y].mean(axis=0)
    his_acpl = loss[idx, y].mean()
    print(f"{N} held-out decisions. His own average loss: {his_acpl:.0f}cp\n")
    print(f"{'PLAY_TEMP':>10}{'log-lik':>10}{'mean p':>9}{'top-1':>8}"
          f"{'feature mismatch':>18}{'bot acpl':>10}")
    rows = []
    for T in TEMPS:
        p = softmax(z_oof, M, T)
        ph = p[idx, y]
        ll = float(np.log(np.maximum(ph, 1e-12)).mean())
        bot_moments = np.einsum("nk,nkd->d", p, X) / N
        # scale-free: how far each behaviour's rate is from his, in units of
        # that behaviour's own spread
        mismatch = float(np.mean(np.abs(bot_moments - his_moments) / scale))
        acpl = float((p * loss).sum() / N)
        rows.append((T, ll, float(ph.mean()), float((p.argmax(1) == y).mean()), mismatch, acpl))
        print(f"{T:>10}{ll:>10.4f}{ph.mean():>9.4f}{(p.argmax(1) == y).mean():>8.2%}"
              f"{mismatch:>18.4f}{acpl:>9.0f}cp")

    best_ll = max(rows, key=lambda r: r[1])
    best_mm = min(rows, key=lambda r: r[4])
    print(f"\nbest log-likelihood : T = {best_ll[0]}")
    print(f"closest behaviour    : T = {best_mm[0]}")
    print(f"his own loss {his_acpl:.0f}cp; closest bot loss at "
          f"T = {min(rows, key=lambda r: abs(r[5] - his_acpl))[0]}")

    # which behaviours move most between the current setting and T=1
    p065 = softmax(z_oof, M, 0.65)
    p1 = softmax(z_oof, M, 1.0)
    m065 = np.einsum("nk,nkd->d", p065, X) / N
    m1 = np.einsum("nk,nkd->d", p1, X) / N
    print("\nbehaviours furthest from him at T=0.65, and where T=1.0 puts them")
    print(f"  {'feature':<32}{'him':>9}{'T=0.65':>9}{'T=1.0':>9}")
    worst = np.argsort(-np.abs(m065 - his_moments) / scale)[:10]
    for k in worst:
        print(f"  {FEATURE_NAMES[k]:<32}{his_moments[k]:>9.4f}{m065[k]:>9.4f}{m1[k]:>9.4f}")


if __name__ == "__main__":
    main()
