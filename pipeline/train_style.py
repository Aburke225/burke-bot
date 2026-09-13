#!/usr/bin/env python3
"""Train the style model: which of Stockfish's candidate moves would Andrew play?

This is imitation learning (the specific method is behavior cloning): every
position from his games past the opening becomes a training example -
Stockfish proposes its top 5 moves, and the label is the one he actually
played. A conditional-logit model (softmax over per-move feature scores)
learns his preferences; the browser ships the weights and samples from them.

CANONICAL FEATURES (a CONTRACT with docs/app.js - change both or neither):
  0 cp_loss (pawns, clamped 0..5), 1 rank/4, 2 capture, 3 gives_check,
  4 promotion, 5 castle, 6..11 moved piece one-hot P N B R Q K,
  12 center distance (0..1), 13 forward,
  --- the candidate extras ---
  14 retreat, 15 same piece as my previous move, 16 toward the enemy king,
  17 recapture, 18 from-square attacked, 19 to-square attacked (both judged
  pre-move), 20 capture while ahead on material, 21 move distance (cheb/7).
(No bias feature: softmax over a shared candidate set cancels any constant.)

Feature selection: features 0-13 are always in; ALL 256 subsets of the 8
extras are trained on a train split and compete on a validation split; the
winner is retrained on train+validation and reported on an untouched test
split. style.json carries {features:"v3", active:[...canonical indices...],
weights:[...]} so the browser can score exactly the chosen subset.

Run after build.py. Stockfish analysis is cached in pipeline/examples-cache.json
(rebuilt whenever games-cache.json is newer).
"""

import itertools
import json
import os
import random
import shutil
import sys

import chess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO_ROOT, "pipeline", "games-cache.json")
EX_CACHE = os.path.join(REPO_ROOT, "pipeline", "examples-cache.json")
OUT = os.path.join(REPO_ROOT, "docs", "style.json")
MIN_PLY = 8
MULTIPV = 5
DEPTH = 8
N_CANON = 22
BASE = list(range(14))
EXTRAS = list(range(14, 22))
PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}

try:
    import numpy as np
except ImportError:
    print("numpy not available - skipping style training")
    sys.exit(0)

import chess.engine

ENGINE_PATH = os.environ.get("STOCKFISH") or shutil.which("stockfish")


def cheb(a, b):
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))


def material(board, color):
    return sum(PIECE_VALUES[p.piece_type]
               for p in board.piece_map().values() if p.color == color)


def features(board, move, cp_gap_pawns, rank, prev_my_to=None, prev_opp_capture_to=None):
    x = [0.0] * N_CANON
    x[0] = min(max(cp_gap_pawns, 0.0), 5.0)
    x[1] = rank / 4.0
    is_cap = board.is_capture(move)
    x[2] = 1.0 if is_cap else 0.0
    x[3] = 1.0 if board.gives_check(move) else 0.0
    x[4] = 1.0 if move.promotion else 0.0
    x[5] = 1.0 if board.is_castling(move) else 0.0
    piece = board.piece_type_at(move.from_square)  # 1..6 = P N B R Q K
    if piece:
        x[5 + piece] = 1.0
    tf, tr = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    x[12] = (abs(tf - 3.5) + abs(tr - 3.5)) / 7.0
    fr = chess.square_rank(move.from_square)
    white = board.turn == chess.WHITE
    x[13] = 1.0 if (tr > fr if white else tr < fr) else 0.0
    x[14] = 1.0 if (tr < fr if white else tr > fr) else 0.0
    x[15] = 1.0 if prev_my_to is not None and prev_my_to == move.from_square else 0.0
    ek = board.king(not board.turn)
    if ek is not None:
        x[16] = 1.0 if cheb(move.to_square, ek) < cheb(move.from_square, ek) else 0.0
    x[17] = 1.0 if is_cap and prev_opp_capture_to == move.to_square else 0.0
    x[18] = 1.0 if board.is_attacked_by(not board.turn, move.from_square) else 0.0
    x[19] = 1.0 if board.is_attacked_by(not board.turn, move.to_square) else 0.0
    x[20] = 1.0 if is_cap and material(board, board.turn) > material(board, not board.turn) else 0.0
    x[21] = cheb(move.from_square, move.to_square) / 7.0
    return x


def collect_examples(games, engine):
    examples = []  # (game_idx, [feature vectors], chosen_idx)
    for gi, g in enumerate(games):
        color = chess.WHITE if g["color"] == "w" else chess.BLACK
        board = chess.Board()
        prev_my_to = None
        prev_opp_capture_to = None
        for ply, uci in enumerate(g["moves"]):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            if ply >= MIN_PLY and board.turn == color and not board.is_game_over():
                infos = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=MULTIPV)
                cands, best_cp = [], None
                for rank, info in enumerate(infos):
                    if "pv" not in info or not info["pv"]:
                        continue
                    cand = info["pv"][0]
                    cp = info["score"].pov(board.turn).score(mate_score=10000)
                    if best_cp is None:
                        best_cp = cp
                    gap = (best_cp - cp) / 100.0
                    cands.append((cand, features(board, cand, gap, rank,
                                                 prev_my_to, prev_opp_capture_to)))
                chosen = next((i for i, (c, _) in enumerate(cands) if c == move), None)
                if chosen is not None and len(cands) >= 2:
                    examples.append((gi, [f for _, f in cands], chosen))
            if board.turn == color:
                prev_my_to = move.to_square
            else:
                prev_opp_capture_to = move.to_square if board.is_capture(move) else None
            board.push(move)
        if (gi + 1) % 50 == 0:
            print(f"  analysed {gi + 1}/{len(games)} games, {len(examples)} examples")
    return examples


def get_examples():
    if (os.path.exists(EX_CACHE) and
            os.path.getmtime(EX_CACHE) > os.path.getmtime(CACHE)):
        d = json.load(open(EX_CACHE))
        if d.get("fmt") == "c22":
            print(f"examples cache hit: {len(d['examples'])} examples")
            return d["examples"]
    if not ENGINE_PATH:
        print("no stockfish binary - skipping style training")
        sys.exit(0)
    games = json.load(open(CACHE))
    print(f"games in cache: {len(games)}")
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        examples = collect_examples(games, engine)
    json.dump({"fmt": "c22", "examples": examples}, open(EX_CACHE, "w"))
    return examples


def to_arrays(examples):
    N = len(examples)
    X = np.zeros((N, MULTIPV, N_CANON))
    M = np.zeros((N, MULTIPV), dtype=bool)
    y = np.zeros(N, dtype=int)
    g = np.zeros(N, dtype=int)
    for i, (gi, feats, chosen) in enumerate(examples):
        for k, f in enumerate(feats[:MULTIPV]):
            X[i, k] = f
            M[i, k] = True
        y[i] = chosen
        g[i] = gi
    return X, M, y, g


def train_vec(X, M, y, cols, iters=400, lr=0.5, l2=1e-3):
    Xc = X[:, :, cols]
    N, K, d = Xc.shape
    w = np.zeros(d)
    onehot = np.zeros((N, K))
    onehot[np.arange(N), y] = 1.0
    for it in range(iters):
        z = Xc @ w
        z = np.where(M, z, -1e9)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad = np.einsum("nk,nkd->d", onehot - p, Xc) / N
        w += lr * (grad - l2 * w)
        if it in (100, 250):
            lr *= 0.5
    return w


def accuracy(X, M, y, cols, w):
    z = X[:, :, cols] @ w
    z = np.where(M, z, -1e9)
    return float((z.argmax(axis=1) == y).mean())


EXTRA_NAMES = {14: "retreat", 15: "same-piece", 16: "toward-king", 17: "recapture",
               18: "from-attacked", 19: "to-attacked", 20: "capture-ahead", 21: "distance"}


def main():
    examples = get_examples()
    X, M, y, g = to_arrays(examples)

    rng = random.Random(7)
    game_ids = sorted(set(g.tolist()))
    rng.shuffle(game_ids)
    n = len(game_ids)
    test_g = set(game_ids[: n * 15 // 100])
    val_g = set(game_ids[n * 15 // 100: n * 30 // 100])
    test_m = np.isin(g, list(test_g))
    val_m = np.isin(g, list(val_g))
    train_m = ~(test_m | val_m)
    print(f"examples: train {train_m.sum()}, val {val_m.sum()}, test {test_m.sum()}")

    results = []
    for r in range(len(EXTRAS) + 1):
        for combo in itertools.combinations(EXTRAS, r):
            cols = BASE + list(combo)
            w = train_vec(X[train_m], M[train_m], y[train_m], cols)
            va = accuracy(X[val_m], M[val_m], y[val_m], cols, w)
            results.append((va, len(cols), combo))
    results.sort(key=lambda t: (-t[0], t[1]))

    base_val = next(va for va, _, combo in results if combo == ())
    best_val, _, best_combo = results[0]
    print(f"validation: base(14) {base_val:.2%} | best {best_val:.2%} "
          f"with extras {[EXTRA_NAMES[i] for i in best_combo] or 'none'}")
    for va, _, combo in results[:5]:
        print(f"  {va:.2%}  + {[EXTRA_NAMES[i] for i in combo] or ['(base only)']}")

    # stability: the incumbent subset keeps its seat unless a challenger beats
    # it on validation by a real margin - near-ties should not churn nightly
    SWITCH_MARGIN = 0.0025
    chosen_combo = best_combo
    try:
        cur = json.load(open(OUT))
        if cur.get("features") == "v3":
            incumbent = tuple(sorted(i for i in cur.get("active", []) if i in EXTRAS))
            inc_val = next((va for va, _, combo in results
                            if tuple(sorted(combo)) == incumbent), None)
            if inc_val is not None and best_val - inc_val < SWITCH_MARGIN:
                chosen_combo = incumbent
                if incumbent != tuple(sorted(best_combo)):
                    print(f"keeping incumbent extras {[EXTRA_NAMES[i] for i in incumbent] or 'none'} "
                          f"(challenger led by only {best_val - inc_val:.2%})")
    except FileNotFoundError:
        pass

    cols = BASE + sorted(chosen_combo)
    fit_m = train_m | val_m
    w = train_vec(X[fit_m], M[fit_m], y[fit_m], cols)
    test_acc = accuracy(X[test_m], M[test_m], y[test_m], cols, w)
    base_w = train_vec(X[fit_m], M[fit_m], y[fit_m], BASE)
    base_test = accuracy(X[test_m], M[test_m], y[test_m], BASE, base_w)
    engine_test = float((y[test_m] == 0).mean())
    print(f"untouched test: chosen {test_acc:.2%} | base-14 {base_test:.2%} | "
          f"engine-best {engine_test:.2%} ({int(test_m.sum())} examples)")

    json.dump({
        "features": "v3",
        "active": cols,
        "weights": [round(float(x), 5) for x in w.tolist()],
        "examples": len(examples),
        "test_accuracy": round(test_acc, 4),
        "engine_best_accuracy": round(engine_test, 4),
    }, open(OUT, "w"), indent=2)
    print(f"wrote {OUT} ({len(cols)} active features)")


if __name__ == "__main__":
    main()
