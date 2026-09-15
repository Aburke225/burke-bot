#!/usr/bin/env python3
"""Fit the v9 conditional logit over the full 55-feature contract.

Two deliberate departures from v8, both measured rather than assumed:

1. NO SUBSET SEARCH. v8 tried all 4,096 subsets of its 12 variable features
   and kept the best by validation top-1 accuracy. That was not merely
   infeasible at 55 features - it was HARMFUL at 12: its winner scored 26.24
   test mean-probability against 27.03 for simply keeping everything, because
   it discriminated 4,096 candidates on a metric whose noise was three times
   the spread between them. Every feature that varies within a position is
   kept, and L2 does the shrinking.

2. L-BFGS, NOT 400 FIXED GRADIENT STEPS. v8's optimiser never converged and
   nobody checked: the gradient at its "solution" was still four times larger
   than a converged fit's. No scipy here, so this is a compact two-loop
   recursion with Armijo backtracking.

THE ACCEPTANCE GATE, and why it is what it is:

  PRIMARY - out-of-fold LOG-LIKELIHOOD of the moves he actually played, folds
  split by GAME (positions from one game are not independent). The log score
  is a PROPER scoring rule: its expectation is maximised by reporting true
  probabilities, so it cannot be gamed by confidence.

  SECONDARY - FEATURE-MOMENT MISMATCH. For each feature, the average over the
  moves he played versus the average the model's distribution puts on it. This
  asks "does it do the things I do, as often as I do them" - and it catches
  changes that are behaviourally right but statistically tiny. The castle-safety
  pair is the worked example: worth only +0.0002 log-likelihood, which reads as
  noise, while halving a move he had explicitly flagged.

NOT the gate: mean probability on his played move. That is the LINEAR score,
and it is IMPROPER - it rises monotonically as a distribution sharpens, so
tuning anything on it drives you toward a deterministic bot. It is printed
below as a human-legible number only. Top-1 accuracy is likewise not the gate;
several features knowingly cost top-1 while earning likelihood, which is what
a suppression term is supposed to do.

Run from the repo root: python3 pipeline/train_v9.py
"""

import json
import os
import sys

import numpy as np

from features_v9 import FEATURE_NAMES, N_FEATURES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO_ROOT, "pipeline", "examples-v9.npz")
STYLE = os.path.join(REPO_ROOT, "docs", "style-v9.json")
NEG = -1e9  # logit for a padded candidate slot
# Shown on the site next to the wordmark. MAJOR is the feature contract (v9);
# bump MINOR whenever the contract changes - it went 55 features, then 57 with
# the castle-safety pair, then 58 with aimless_edge_pawn. A plain retrain on new
# games does NOT bump it: same contract, same version, new weights.
VERSION = "9.3"


def nll_and_grad(w, X, M, y, lam):
    """Conditional-logit loss and gradient. X is (N, K, D), y picks a column."""
    N = X.shape[0]
    z = X @ w
    z = np.where(M, z, NEG)
    zmax = z.max(axis=1, keepdims=True)
    e = np.exp(z - zmax)
    e = np.where(M, e, 0.0)
    s = e.sum(axis=1, keepdims=True)
    logZ = (zmax + np.log(s)).ravel()
    idx = np.arange(N)
    chosen = z[idx, y]
    loss = float(np.mean(logZ - chosen)) + 0.5 * lam * float(w @ w)
    p = e / s
    p[idx, y] -= 1.0                      # d/dz of (logZ - z_y)
    grad = np.einsum("nk,nkd->d", p, X) / N + lam * w
    return loss, grad


def lbfgs(f, w0, iters=400, m=10, tol=1e-7):
    """Limited-memory BFGS with Armijo backtracking; pure numpy."""
    w = w0.copy()
    loss, g = f(w)
    S, Y, rho = [], [], []
    for it in range(iters):
        if np.max(np.abs(g)) < tol:
            break
        q = g.copy()
        alpha = []
        for s_i, y_i, r_i in zip(reversed(S), reversed(Y), reversed(rho)):
            a = r_i * (s_i @ q)
            alpha.append(a)
            q -= a * y_i
        if S:
            gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
            q *= gamma
        for (s_i, y_i, r_i), a in zip(zip(S, Y, rho), reversed(alpha)):
            b = r_i * (y_i @ q)
            q += s_i * (a - b)
        d = -q
        slope = g @ d
        if slope >= 0:                     # curvature broke down; reset
            d = -g
            slope = g @ d
            S, Y, rho = [], [], []
        t = 1.0
        for _ in range(25):                # Armijo
            w_new = w + t * d
            loss_new, g_new = f(w_new)
            if loss_new <= loss + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            break                          # no acceptable step; done
        s_vec, y_vec = w_new - w, g_new - g
        sy = s_vec @ y_vec
        if sy > 1e-12:
            S.append(s_vec); Y.append(y_vec); rho.append(1.0 / sy)
            if len(S) > m:
                S.pop(0); Y.pop(0); rho.pop(0)
        w, loss, g = w_new, loss_new, g_new
    return w, loss, float(np.max(np.abs(g))), it + 1


def policy_metrics(w, X, M, y, temp=1.0):
    """(mean probability on his real move, top-1 agreement, mean log-lik)."""
    z = (X @ w) / temp
    z = np.where(M, z, NEG)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    e = np.where(M, e, 0.0)
    p = e / e.sum(axis=1, keepdims=True)
    idx = np.arange(len(y))
    ph = p[idx, y]
    return float(ph.mean()), float((p.argmax(axis=1) == y).mean()), float(np.log(ph).mean())


def game_folds(g, k=5, seed=11):
    games = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    buckets = np.array_split(games, k)
    return [(np.isin(g, b, invert=True), np.isin(g, b)) for b in buckets]


def main():
    if not os.path.exists(EXAMPLES):
        sys.exit(f"no {EXAMPLES} - run pipeline/build_examples_v9.py first")
    d = np.load(EXAMPLES)
    X, M, y, g = d["X"].astype(np.float64), d["M"], d["y"].astype(int), d["g"]
    N, K, D = X.shape
    assert D == N_FEATURES, f"{D} features in the cache, {N_FEATURES} in the contract"
    print(f"{N} examples, {K}-wide pool, {D} features, "
          f"{len(np.unique(g))} games")

    # scale for the optimiser only; the shipped weights apply to RAW features,
    # so the browser never has to know this happened
    flat = X[M]
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = X / scale

    folds = game_folds(g, k=5)
    print("\nL2 sweep, 5 folds split by GAME (positions in one game are not independent):")
    print(f"{'lambda':>9}{'mean p(his move)':>19}{'top-1':>9}{'log-lik':>10}")
    results = {}
    for lam in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        ps, accs, lls = [], [], []
        for tr, te in folds:
            w, _, _, _ = lbfgs(lambda v: nll_and_grad(v, Xs[tr], M[tr], y[tr], lam),
                               np.zeros(D))
            p, a, ll = policy_metrics(w, Xs[te], M[te], y[te])
            ps.append(p); accs.append(a); lls.append(ll)
        results[lam] = (np.mean(ps), np.mean(accs), np.mean(lls))
        print(f"{lam:>9.0e}{np.mean(ps):>18.4f} {np.mean(accs):>8.2%}{np.mean(lls):>10.4f}")

    # selected on LOG-LIKELIHOOD (proper), not mean probability (improper)
    best_lam = max(results, key=lambda k_: results[k_][2])
    bp, ba, bll = results[best_lam]
    print(f"\nbest L2 = {best_lam:.0e}  (mean p {bp:.4f}, top-1 {ba:.2%}, log-lik {bll:.4f})")

    # baselines on the same folds, for context
    rank_only = np.zeros(D);
    rank_idx = FEATURE_NAMES.index("engine_rank")
    base_ps = []
    for tr, te in folds:
        mask = np.zeros(D, dtype=bool); mask[rank_idx] = True
        w, _, _, _ = lbfgs(lambda v: nll_and_grad(v, Xs[tr][:, :, mask], M[tr], y[tr], best_lam),
                           np.zeros(1))
        p, _, _ = policy_metrics(w, Xs[te][:, :, mask], M[te], y[te])
        base_ps.append(p)
    print(f"engine-rank-only baseline: mean p {np.mean(base_ps):.4f}")
    print(f"always-play-engine-best  : {np.mean(y == 0):.2%} of his moves")

    # final fit on everything
    w_s, loss, gmax, iters = lbfgs(lambda v: nll_and_grad(v, Xs, M, y, best_lam), np.zeros(D))
    w = w_s / scale
    print(f"\nfinal fit: {iters} L-BFGS iterations, loss {loss:.5f}, "
          f"|grad|_inf {gmax:.2e}")
    p_in, a_in, _ = policy_metrics(w_s, Xs, M, y)
    print(f"in-sample mean p {p_in:.4f} vs out-of-fold {bp:.4f} "
          f"(gap {p_in - bp:+.4f} - small gap means nothing to select against)")

    # the behavioural gate: does it do what he does, as often as he does it?
    idx = np.arange(N)
    zz = np.where(M, Xs @ w_s, NEG)
    ee = np.exp(zz - zz.max(axis=1, keepdims=True))
    ee = np.where(M, ee, 0.0)
    pp = ee / ee.sum(axis=1, keepdims=True)
    his_m = X[idx, y].mean(axis=0)
    bot_m = np.einsum("nk,nkd->d", pp, X) / N
    mismatch = np.abs(bot_m - his_m) / scale
    print(f"\nbehaviour match: mean |bot - him| = {mismatch.mean():.5f} "
          f"feature std (0 = he and it do everything at the same rate)")
    worst = np.argsort(-mismatch)[:5]
    print("  furthest apart: " + ", ".join(
        f"{FEATURE_NAMES[k]} {his_m[k]:.3f} vs {bot_m[k]:.3f}" for k in worst))

    order = np.argsort(-np.abs(w))
    print("\nweights, largest first:")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<34} {w[i]:+.4f}")

    json.dump({
        "features": "v9",
        "version": VERSION,
        "n_features": N_FEATURES,
        "names": FEATURE_NAMES,
        "active": list(range(N_FEATURES)),
        "weights": [float(v) for v in w],
        "l2": best_lam,
        "examples": int(N),
        "pool": int(K),
        "out_of_fold_log_lik": float(bll),
        "out_of_fold_mean_p": float(bp),
        "out_of_fold_top1": float(ba),
        "behaviour_mismatch": float(mismatch.mean()),
    }, open(STYLE, "w"), indent=1)
    print(f"\nwrote {STYLE}")


if __name__ == "__main__":
    main()
