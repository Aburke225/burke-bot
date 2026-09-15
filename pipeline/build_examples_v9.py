#!/usr/bin/env python3
"""Turn pipeline/raw-cache.json into the v9 training matrices.

The engine pass (pipeline/analyse.py) is the expensive, cacheable half; this is
the cheap, re-runnable half. Changing a feature means re-running THIS only -
seconds of work instead of re-analysing 357 games.

Writes pipeline/examples-v9.npz:
  X  (N, 20, 55) float32   feature vectors, padded to the 20-wide pool
  M  (N, 20)     bool      which candidate slots are real
  y  (N,)        int16     index of the move he actually played
  g  (N,)        int32     game index, so CV folds can split by GAME - two
                           positions from one game are not independent
Only decision points where his move IS among the candidates can train a
conditional logit, so the rest are dropped and counted.

Run from the repo root: python3 pipeline/build_examples_v9.py
"""

import json
import os
import sys

import chess
import numpy as np

from features_v9 import Context, features, horizon_scores, N_FEATURES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "pipeline", "raw-cache.json")
OUT = os.path.join(REPO_ROOT, "pipeline", "examples-v9.npz")
MAX_CANDS = 20


def main():
    if not os.path.exists(RAW):
        sys.exit(f"no {RAW} - run pipeline/analyse.py first")
    raw = json.load(open(RAW))
    positions = raw["positions"]
    print(f"{len(positions)} decision points from {raw['multipv']}-wide "
          f"depth-{raw['pool_depth']} pools (horizon depth {raw['horizon_depth']})")

    X, M, y, g = [], [], [], []
    skipped_not_in_pool = 0
    for n, p in enumerate(positions):
        board = chess.Board(p["fen"])
        ctx = Context(board, p["prev_my_to"], p["prev_opp_cap_to"], p["last_move_to"])
        ucis = [c["uci"] for c in p["cands"]][:MAX_CANDS]
        chosen = ucis.index(p["played"]) if p["played"] in ucis else None
        if chosen is None:
            skipped_not_in_pool += 1
            continue
        sh_by = {c["uci"]: c["sh"] for c in p["cands"][:MAX_CANDS]}
        sh_list, best_sh = horizon_scores(ucis, sh_by)
        rows = np.zeros((MAX_CANDS, N_FEATURES), dtype=np.float32)
        mask = np.zeros(MAX_CANDS, dtype=bool)
        for i, (u, sh) in enumerate(zip(ucis, sh_list)):
            rows[i] = features(board, chess.Move.from_uci(u), ctx, sh, best_sh, i)
            mask[i] = True
        X.append(rows); M.append(mask); y.append(chosen); g.append(p["gi"])
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(positions)} positions", flush=True)

    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=bool)
    y = np.asarray(y, dtype=np.int16)
    g = np.asarray(g, dtype=np.int32)
    np.savez_compressed(OUT, X=X, M=M, y=y, g=g)
    print(f"\n{len(y)} usable examples across {len(set(g.tolist()))} games "
          f"({skipped_not_in_pool} dropped - his move was outside the pool)")
    print(f"his move's rank in the pool: mean {y.mean():.2f}, "
          f"rank 0 {(y == 0).mean():.1%}, rank >= 10 {(y >= 10).mean():.1%}")
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
