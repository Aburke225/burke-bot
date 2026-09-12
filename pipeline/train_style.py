#!/usr/bin/env python3
"""Train the style model: which of Stockfish's candidate moves would Andrew play?

This is behavioral cloning (imitation learning): every position from his games
past the opening becomes a training example — Stockfish proposes its top 5
moves, and the label is the one he actually played. A conditional-logit model
(softmax over per-move feature scores) learns his preferences. The browser
ships the weights and, out of book, samples the bot's move from the model's
probabilities over the live engine's top 5.

Feature order is a CONTRACT with docs/app.js — change both or neither:
  0 bias, 1 eval_gap_pawns (clamped 0..5), 2 rank/4, 3 capture, 4 gives_check,
  5 promotion, 6 castle, 7..12 moved piece one-hot P N B R Q K,
  13 to-square center distance (0..1), 14 forward move.

Run after build.py (it needs pipeline/games-cache.json):
  python3 pipeline/train_style.py
Skips gracefully when no stockfish binary or no numpy is available.
"""

import json
import os
import random
import shutil
import sys

import chess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO_ROOT, "pipeline", "games-cache.json")
OUT = os.path.join(REPO_ROOT, "docs", "style.json")
MIN_PLY = 8          # the opening book owns earlier plies
MULTIPV = 5
DEPTH = 8
N_FEATURES = 15

try:
    import numpy as np
except ImportError:
    print("numpy not available - skipping style training")
    sys.exit(0)

import chess.engine

ENGINE_PATH = os.environ.get("STOCKFISH") or shutil.which("stockfish")
if not ENGINE_PATH:
    print("no stockfish binary - skipping style training")
    sys.exit(0)


def features(board, move, cp_gap_pawns, rank):
    x = [0.0] * N_FEATURES
    x[0] = 1.0
    x[1] = min(max(cp_gap_pawns, 0.0), 5.0)
    x[2] = rank / 4.0
    x[3] = 1.0 if board.is_capture(move) else 0.0
    x[4] = 1.0 if board.gives_check(move) else 0.0
    x[5] = 1.0 if move.promotion else 0.0
    x[6] = 1.0 if board.is_castling(move) else 0.0
    piece = board.piece_type_at(move.from_square)  # 1..6 = P N B R Q K
    if piece:
        x[6 + piece] = 1.0
    tf, tr = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    x[13] = (abs(tf - 3.5) + abs(tr - 3.5)) / 7.0
    fr = chess.square_rank(move.from_square)
    x[14] = 1.0 if (tr > fr if board.turn == chess.WHITE else tr < fr) else 0.0
    return x


def collect_examples(games, engine):
    examples = []  # (game_idx, [feature vectors], chosen_idx)
    for gi, g in enumerate(games):
        color = chess.WHITE if g["color"] == "w" else chess.BLACK
        board = chess.Board()
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
                    cands.append((cand, features(board, cand, gap, rank)))
                chosen = next((i for i, (c, _) in enumerate(cands) if c == move), None)
                if chosen is not None and len(cands) >= 2:
                    examples.append((gi, [f for _, f in cands], chosen))
            board.push(move)
        if (gi + 1) % 50 == 0:
            print(f"  analysed {gi + 1}/{len(games)} games, {len(examples)} examples")
    return examples


def train(train_ex, test_ex):
    w = np.zeros(N_FEATURES)
    lr, l2 = 0.5, 1e-3
    for it in range(400):
        grad = np.zeros(N_FEATURES)
        for _, feats, chosen in train_ex:
            X = np.array(feats)
            z = X @ w
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            grad += X[chosen] - p @ X
        w += lr * (grad / len(train_ex) - l2 * w)
        if it in (100, 250):
            lr *= 0.5

    def accuracy(exs, pick):
        return sum(1 for _, f, c in exs if pick(np.array(f)) == c) / len(exs)

    model_acc = accuracy(test_ex, lambda X: int(np.argmax(X @ w)))
    engine_acc = accuracy(test_ex, lambda X: 0)  # always take the engine's best
    return w, model_acc, engine_acc


def main():
    with open(CACHE) as f:
        games = json.load(f)
    print(f"games in cache: {len(games)}")

    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        examples = collect_examples(games, engine)
    print(f"training examples: {len(examples)}")
    if len(examples) < 500:
        print("not enough examples - skipping")
        return

    rng = random.Random(7)
    game_ids = sorted({gi for gi, _, _ in examples})
    rng.shuffle(game_ids)
    test_games = set(game_ids[: max(1, len(game_ids) // 7)])
    train_ex = [e for e in examples if e[0] not in test_games]
    test_ex = [e for e in examples if e[0] in test_games]

    w, model_acc, engine_acc = train(train_ex, test_ex)
    print(f"held-out top-1 accuracy: model {model_acc:.1%} vs engine-best {engine_acc:.1%} "
          f"({len(test_ex)} test examples)")

    with open(OUT, "w") as f:
        json.dump({
            "features": "v1",
            "weights": [round(x, 5) for x in w.tolist()],
            "examples": len(examples),
            "test_accuracy": round(model_acc, 4),
            "engine_best_accuracy": round(engine_acc, 4),
        }, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
