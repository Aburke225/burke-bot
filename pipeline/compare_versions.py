#!/usr/bin/env python3
"""Two honest measurements, both on ONE data set so nothing is apples-to-oranges.

1. CALIBRATION. For a behaviour like "purposeless edge-pawn push", does the bot
   spend the same share of its probability on it that Andrew spends of his
   actual moves? Per-position agreement is NOT the goal and would be wrong -
   in a sharp position the rate should be near zero and in a dead quiet one it
   should be higher. What must match is the AGGREGATE.

2. v8 vs v9, same examples, same folds, same optimiser, same L2 - only the
   feature set differs. Comparing v8-as-shipped (10-wide pool) against v9
   (20-wide) would flatter v9, because probabilities fall as the pool widens.

Run from the repo root: python3 pipeline/compare_versions.py
"""

import json
import os
import sys

import chess
import numpy as np

from features_v9 import Context, features as f9, horizon_scores, N_FEATURES
from train_style import features as f8, position_tension
from train_v9 import game_folds, lbfgs, nll_and_grad, policy_metrics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "pipeline", "raw-cache.json")
V9_N = N_FEATURES
V8_N = 30
PLAY_TEMP = 0.65


def build(positions, pool):
    """Feature matrices for both contracts over the same decision points."""
    X9 = np.zeros((len(positions), pool, V9_N), dtype=np.float32)
    X8 = np.zeros((len(positions), pool, V8_N), dtype=np.float32)
    M = np.zeros((len(positions), pool), dtype=bool)
    y = np.zeros(len(positions), dtype=int)
    g = np.zeros(len(positions), dtype=int)
    aimless = np.zeros((len(positions), pool), dtype=bool)
    keep = []
    for n, p in enumerate(positions):
        board = chess.Board(p["fen"])
        ucis = [c["uci"] for c in p["cands"]][:pool]
        if p["played"] not in ucis:
            continue
        ctx = Context(board, p["prev_my_to"], p["prev_opp_cap_to"], p["last_move_to"])
        sh, best_sh = horizon_scores(ucis, {c["uci"]: c["sh"] for c in p["cands"][:pool]})
        cps = [c["cp"] for c in p["cands"][:pool]]
        best_cp = max(cps)
        tension = position_tension(board)
        i = len(keep)
        for k, u in enumerate(ucis):
            mv = chess.Move.from_uci(u)
            v9 = f9(board, mv, ctx, sh[k], best_sh, k)
            X9[i, k] = v9
            X8[i, k] = f8(board, mv, (best_cp - cps[k]) / 100.0, k,
                          p["prev_my_to"], p["prev_opp_cap_to"], p["prev_my_from"], tension)
            M[i, k] = True
            aimless[i, k] = bool(v9[57])
        y[i] = ucis.index(p["played"])
        g[i] = p["gi"]
        keep.append(n)
        if (n + 1) % 1500 == 0:
            print(f"  {n + 1}/{len(positions)}", flush=True)
    s = len(keep)
    return X9[:s], X8[:s], M[:s], y[:s], g[:s], aimless[:s]


def fit_and_score(X, M, y, g, folds, lam=3e-5):
    flat = X[M]
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = X / scale
    ps, accs, lls = [], [], []
    for tr, te in folds:
        w, _, _, _ = lbfgs(lambda v: nll_and_grad(v, Xs[tr], M[tr], y[tr], lam),
                           np.zeros(X.shape[2]))
        p, a, ll = policy_metrics(w, Xs[te], M[te], y[te])
        ps.append(p); accs.append(a); lls.append(ll)
    w, _, _, _ = lbfgs(lambda v: nll_and_grad(v, Xs, M, y, lam), np.zeros(X.shape[2]))
    return np.mean(ps), np.mean(accs), np.mean(lls), w / scale


def main():
    positions = json.load(open(RAW))["positions"]
    print(f"building both feature sets over {len(positions)} decision points "
          f"(20-wide pool, identical examples)...")
    X9, X8, M, y, g, aimless = build(positions, 20)
    folds = game_folds(g, k=5)
    print(f"{len(y)} usable examples, {len(np.unique(g))} games\n")

    p9, a9, l9, w9 = fit_and_score(X9, M, y, g, folds)
    p8, a8, l8, _ = fit_and_score(X8, M, y, g, folds)
    print("SAME examples, SAME folds, SAME optimiser - only the features differ:")
    print(f"{'':>26}{'mean p(his move)':>18}{'top-1':>9}{'log-lik':>10}")
    print(f"{'v8 contract (30 feats)':>26}{p8:>18.4f}{a8:>9.2%}{l8:>10.4f}")
    print(f"{'v9 contract (58 feats)':>26}{p9:>18.4f}{a9:>9.2%}{l9:>10.4f}")
    print(f"{'improvement':>26}{p9 - p8:>+18.4f}{a9 - a8:>+9.2%}{l9 - l8:>+10.4f}")
    print(f"{'always engine-best':>26}{'':>18}{np.mean(y == 0):>9.2%}")

    # --- calibration of the aimless edge push, in aggregate ---
    z = X9 @ w9
    z = np.where(M, z / PLAY_TEMP, -1e9)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    e = np.where(M, e, 0.0)
    prob = e / e.sum(axis=1, keepdims=True)
    offered = aimless.any(axis=1)
    bot_rate = float(prob[aimless].sum() / len(y))
    his_rate = float(aimless[np.arange(len(y)), y].mean())
    print(f"\nAIMLESS EDGE PUSH - aggregate calibration over all {len(y)} positions")
    print(f"  offered at all (in the 20-wide pool): {offered.mean():.1%} of positions")
    print(f"  HIS rate   (he actually played one):  {100 * his_rate:.2f}% of his moves")
    print(f"  BOT rate   (probability mass):        {100 * bot_rate:.2f}% of its moves")
    print(f"  ratio bot/him: {bot_rate / his_rate:.2f}x  (1.00 = perfectly calibrated)")
    o = offered
    print(f"  conditional on one being offered: him {100 * aimless[np.arange(len(y)), y][o].mean():.2f}%"
          f"  bot {100 * prob[aimless].sum() / o.sum():.2f}%")


if __name__ == "__main__":
    main()
