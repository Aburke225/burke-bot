#!/usr/bin/env python3
"""Train the style model: which of Stockfish's candidate moves would Andrew play?

This is imitation learning (the specific method is behavior cloning): every
position from his games past the opening becomes a training example -
Stockfish proposes its top 5 moves, and the label is the one he actually
played. A conditional-logit model (softmax over per-move feature scores)
learns his preferences; the browser ships the weights and samples from them.

CANONICAL FEATURES (a CONTRACT with docs/app.js - change both or neither):
  --- 15 static features, always active ---
  0 cp_loss (pawns, clamped 0..5), 1 rank/4, 2 capture, 3 gives_check,
  4 promotion, 5 castle, 6..11 moved piece one-hot P N B R Q K,
  12 center distance (0..1), 13 forward, 14 capture value (victim/9),
  --- 15 variable features, competing for a seat ---
  15 retreat, 16 same piece as my previous move, 17 toward the enemy king,
  18 recapture, 19 from-square attacked, 20 to-square attacked (both judged
  pre-move), 21 capture while ahead on material, 22 move distance (cheb/7),
  23 destination in the enemy half, 24 destination's distance to the enemy
  king (cheb/7), 25 capture of a defended piece, 26 escape (attacked piece
  moves somewhere safe), 27 landing square defended (judged post-move),
  28 from my back rank, 29 undo (same piece returns to where it just was).
(No bias feature: softmax over a shared candidate set cancels any constant.)

Feature selection: the 15 static features are always in; ALL 32,768 subsets
of the 15 variable features are trained on a train split and compete on a
validation split (batched - every subset in a chunk shares two big matrix
multiplies per iteration, with inactive columns zero-masked, which is exactly
equivalent to training each subset alone). The winner is retrained on
train+validation and reported on an untouched test split. style.json carries
{features:"v4", active:[...canonical indices...], weights:[...], extras:[names]}
so the browser can score exactly the chosen subset.

Stability: the incumbent subset (read from docs/style.json) keeps its seat
unless a challenger beats it on validation by SWITCH_MARGIN.

Run after build.py. Analysis runs on the SITE'S OWN engine (the vendored WASM
Stockfish under docs/vendor/stockfish/, driven headless through node) and is
cached in pipeline/examples-cache.json (rebuilt whenever games-cache.json is
newer or the engine changes). The daily GitHub Action only runs this when
pipeline/games-fingerprint.txt says the game set changed.
"""

import itertools
import json
import os
import random
import shutil
import sys
import time

import chess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO_ROOT, "pipeline", "games-cache.json")
EX_CACHE = os.path.join(REPO_ROOT, "pipeline", "examples-cache.json")
OUT = os.path.join(REPO_ROOT, "docs", "style.json")
MIN_PLY = 8
MULTIPV = 5
DEPTH = 8
N_CANON = 30
BASE = list(range(15))
EXTRAS = list(range(15, 30))
CHUNK = 1024  # subsets trained per batched fit
SWITCH_MARGIN = 0.0025
PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}

try:
    import numpy as np
except ImportError:
    print("numpy not available - skipping style training")
    sys.exit(0)

import chess.engine

# the trainer analyses with the SAME engine the site plays with: the vendored
# WASM build, run headless through node. That kills train/play skew - the
# model always chooses among candidates ranked exactly like its training data
# - and upgrading the vendored engine upgrades both sides at once.
# STOCKFISH=<path> still overrides with a native binary for experiments.
VENDOR_JS = os.path.join(REPO_ROOT, "docs", "vendor", "stockfish",
                         "stockfish-18-lite-single.js")


def engine_cmd():
    if os.environ.get("STOCKFISH"):
        return [os.environ["STOCKFISH"]], os.path.basename(os.environ["STOCKFISH"])
    node = shutil.which("node")
    if node and os.path.exists(VENDOR_JS):
        return [node, VENDOR_JS], os.path.basename(VENDOR_JS)
    native = shutil.which("stockfish")
    if native:
        return [native], "stockfish-native"
    return None, None


ENGINE_CMD, ENGINE_TAG = engine_cmd()

EXTRA_NAMES = {15: "retreat", 16: "same-piece", 17: "toward-king", 18: "recapture",
               19: "from-attacked", 20: "to-attacked", 21: "capture-ahead",
               22: "distance", 23: "enemy-half", 24: "king-dist",
               25: "capture-defended", 26: "escape", 27: "defended-to",
               28: "back-rank", 29: "undo-move"}
NAME_TO_EXTRA = {v: k for k, v in EXTRA_NAMES.items()}
V3_NAMES = {14: "retreat", 15: "same-piece", 16: "toward-king", 17: "recapture",
            18: "from-attacked", 19: "to-attacked", 20: "capture-ahead", 21: "distance"}


def cheb(a, b):
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))


def material(board, color):
    return sum(PIECE_VALUES[p.piece_type]
               for p in board.piece_map().values() if p.color == color)


def features(board, move, cp_gap_pawns, rank,
             prev_my_to=None, prev_opp_capture_to=None, prev_my_from=None):
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
    if is_cap:
        victim = board.piece_type_at(move.to_square)  # None means en passant
        x[14] = PIECE_VALUES[victim or chess.PAWN] / 9.0
    x[15] = 1.0 if (tr < fr if white else tr > fr) else 0.0
    x[16] = 1.0 if prev_my_to is not None and prev_my_to == move.from_square else 0.0
    ek = board.king(not board.turn)
    if ek is not None:
        x[17] = 1.0 if cheb(move.to_square, ek) < cheb(move.from_square, ek) else 0.0
        x[24] = cheb(move.to_square, ek) / 7.0
    x[18] = 1.0 if is_cap and prev_opp_capture_to == move.to_square else 0.0
    from_att = board.is_attacked_by(not board.turn, move.from_square)
    to_att = board.is_attacked_by(not board.turn, move.to_square)
    x[19] = 1.0 if from_att else 0.0
    x[20] = 1.0 if to_att else 0.0
    x[21] = 1.0 if is_cap and material(board, board.turn) > material(board, not board.turn) else 0.0
    x[22] = cheb(move.from_square, move.to_square) / 7.0
    x[23] = 1.0 if (tr >= 4 if white else tr <= 3) else 0.0
    x[25] = 1.0 if is_cap and to_att else 0.0
    x[26] = 1.0 if from_att and not to_att else 0.0
    mover = board.turn
    board.push(move)
    x[27] = 1.0 if board.is_attacked_by(mover, move.to_square) else 0.0
    board.pop()
    x[28] = 1.0 if (fr == 0 if white else fr == 7) else 0.0
    x[29] = 1.0 if (prev_my_to is not None and prev_my_to == move.from_square and
                    prev_my_from is not None and prev_my_from == move.to_square) else 0.0
    return x


def collect_examples(games, engine):
    examples = []  # (game_idx, [feature vectors], chosen_idx)
    for gi, g in enumerate(games):
        color = chess.WHITE if g["color"] == "w" else chess.BLACK
        board = chess.Board()
        prev_my_to = None
        prev_my_from = None
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
                                                 prev_my_to, prev_opp_capture_to,
                                                 prev_my_from)))
                chosen = next((i for i, (c, _) in enumerate(cands) if c == move), None)
                if chosen is not None and len(cands) >= 2:
                    examples.append((gi, [f for _, f in cands], chosen))
            if board.turn == color:
                prev_my_to = move.to_square
                prev_my_from = move.from_square
            else:
                prev_opp_capture_to = move.to_square if board.is_capture(move) else None
            board.push(move)
        if (gi + 1) % 50 == 0:
            print(f"  analysed {gi + 1}/{len(games)} games, {len(examples)} examples", flush=True)
    return examples


def get_examples():
    if (os.path.exists(EX_CACHE) and
            os.path.getmtime(EX_CACHE) > os.path.getmtime(CACHE)):
        d = json.load(open(EX_CACHE))
        if d.get("fmt") == "c30" and d.get("engine") == ENGINE_TAG:
            print(f"examples cache hit: {len(d['examples'])} examples")
            return d["examples"]
    if not ENGINE_CMD:
        print("no engine available (need node + the vendored build, or a "
              "stockfish binary) - skipping style training")
        sys.exit(0)
    games = json.load(open(CACHE))
    print(f"games in cache: {len(games)} (engine: {ENGINE_TAG})", flush=True)
    with chess.engine.SimpleEngine.popen_uci(ENGINE_CMD) as engine:
        examples = collect_examples(games, engine)
    json.dump({"fmt": "c30", "engine": ENGINE_TAG, "examples": examples},
              open(EX_CACHE, "w"))
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


def train_many(X, M, y, combos, iters=400, lr=0.5, l2=1e-3):
    """One conditional logit per subset, all trained at once.

    Inactive columns start at zero weight and are re-zeroed after every
    update, which is exactly equivalent to slicing them out (they add nothing
    to the logits and never move) - but it lets every subset in the chunk
    share two big matrix multiplies per iteration instead of running
    thousands of separate fits.
    """
    N, K, D = X.shape
    S = len(combos)
    Xf = np.ascontiguousarray(X.reshape(N * K, D), dtype=np.float32)
    colmask = np.zeros((S, D), dtype=np.float32)
    for si, combo in enumerate(combos):
        colmask[si, BASE] = 1.0
        if combo:
            colmask[si, list(combo)] = 1.0
    W = np.zeros((S, D), dtype=np.float32)
    onehot = np.zeros((N, K), dtype=np.float32)
    onehot[np.arange(N), y] = 1.0
    dead = ~M  # padded candidate slots
    for it in range(iters):
        Z = (Xf @ W.T).reshape(N, K, S)
        Z[dead] = -1e9
        Z -= Z.max(axis=1, keepdims=True)
        np.exp(Z, out=Z)
        Z /= Z.sum(axis=1, keepdims=True)
        E = onehot[:, :, None] - Z
        E[dead] = 0.0
        G = E.reshape(N * K, S).T @ Xf
        W += lr * (G / N - l2 * W)
        W *= colmask
        if it in (100, 250):
            lr *= 0.5
    return W


def accuracy(X, M, y, cols, w):
    z = X[:, :, cols] @ w
    z = np.where(M, z, -1e9)
    return float((z.argmax(axis=1) == y).mean())


def accuracy_many(X, M, y, W):
    N, K, D = X.shape
    Z = (X.reshape(N * K, D).astype(np.float32) @ W.T).reshape(N, K, -1)
    Z[~M] = -1e9
    return (Z.argmax(axis=1) == y[:, None]).mean(axis=0)


def incumbent_extras():
    """The variable subset the live model currently uses, as new-scheme indices."""
    try:
        cur = json.load(open(OUT))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if cur.get("features") == "v4":
        names = cur.get("extras", [])
    elif cur.get("features") == "v3":
        names = [V3_NAMES[i] for i in cur.get("active", []) if i in V3_NAMES]
    else:
        return None
    if not all(n in NAME_TO_EXTRA for n in names):
        return None
    return tuple(sorted(NAME_TO_EXTRA[n] for n in names))


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
    print(f"examples: train {train_m.sum()}, val {val_m.sum()}, test {test_m.sum()}", flush=True)

    subsets = [combo for r in range(len(EXTRAS) + 1)
               for combo in itertools.combinations(EXTRAS, r)]
    results = []
    t0 = time.time()
    for c0 in range(0, len(subsets), CHUNK):
        chunk = subsets[c0:c0 + CHUNK]
        W = train_many(X[train_m], M[train_m], y[train_m], chunk)
        vas = accuracy_many(X[val_m], M[val_m], y[val_m], W)
        for combo, va in zip(chunk, vas):
            results.append((float(va), len(combo), combo))
        done = c0 + len(chunk)
        el = time.time() - t0
        print(f"  {done}/{len(subsets)} subsets "
              f"({el / 60:.1f} min elapsed, ~{el / done * (len(subsets) - done) / 60:.1f} min left)",
              flush=True)
    results.sort(key=lambda t: (-t[0], t[1], t[2]))

    base_val = next(va for va, _, combo in results if combo == ())
    best_val, _, best_combo = results[0]
    print(f"validation: base(15) {base_val:.2%} | best {best_val:.2%} "
          f"with extras {[EXTRA_NAMES[i] for i in best_combo] or 'none'}")
    for va, _, combo in results[:5]:
        print(f"  {va:.2%}  + {[EXTRA_NAMES[i] for i in combo] or ['(static only)']}")

    # stability: the incumbent subset keeps its seat unless a challenger beats
    # it on validation by a real margin - near-ties should not churn the model
    chosen_combo = best_combo
    incumbent = incumbent_extras()
    if incumbent is not None:
        inc_val = next((va for va, _, combo in results
                        if tuple(sorted(combo)) == incumbent), None)
        if inc_val is not None and best_val - inc_val < SWITCH_MARGIN:
            chosen_combo = incumbent
            if incumbent != tuple(sorted(best_combo)):
                print(f"keeping incumbent extras {[EXTRA_NAMES[i] for i in incumbent] or 'none'} "
                      f"(challenger led by only {best_val - inc_val:.2%})")

    cols = BASE + sorted(chosen_combo)
    fit_m = train_m | val_m
    w = train_vec(X[fit_m], M[fit_m], y[fit_m], cols)
    test_acc = accuracy(X[test_m], M[test_m], y[test_m], cols, w)
    base_w = train_vec(X[fit_m], M[fit_m], y[fit_m], BASE)
    base_test = accuracy(X[test_m], M[test_m], y[test_m], BASE, base_w)
    engine_test = float((y[test_m] == 0).mean())
    print(f"untouched test: chosen {test_acc:.2%} | static-15 {base_test:.2%} | "
          f"engine-best {engine_test:.2%} ({int(test_m.sum())} examples)")

    json.dump({
        "features": "v4",
        "active": cols,
        "weights": [round(float(x), 5) for x in w.tolist()],
        "extras": [EXTRA_NAMES[i] for i in sorted(chosen_combo)],
        "examples": len(examples),
        "test_accuracy": round(test_acc, 4),
        "engine_best_accuracy": round(engine_test, 4),
    }, open(OUT, "w"), indent=2)
    print(f"wrote {OUT} ({len(cols)} active features)")


if __name__ == "__main__":
    main()
