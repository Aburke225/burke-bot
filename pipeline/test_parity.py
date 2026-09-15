#!/usr/bin/env python3
"""The two halves of the v9 feature contract must agree, exactly.

pipeline/features_v9.py trains the model; the v9 block in docs/app.js plays it.
If they disagree the bot plays a different game than the one it was fitted to,
and NOTHING in the training metrics notices - the loss is computed on the
python side and looks perfect either way. So this is the one check that has to
run on every commit.

Self-contained on purpose: it builds its own positions from move lists rather
than reading pipeline/raw-cache.json, which is gitignored and absent in CI.

It also asserts COVERAGE. An earlier parity run sampled 2,112 real vectors and
still never exercised feature 7 (promotion), so "0 mismatches" was quietly
weaker than it sounded. Every feature must be non-zero somewhere here.

    python3 pipeline/test_parity.py     # exits non-zero on failure
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chess  # noqa: E402

from features_v9 import (Context, FEATURE_NAMES, N_FEATURES, features,  # noqa: E402
                         horizon_scores)

# chosen to exercise the whole contract, not just the common path: castling
# both sides, promotion, en passant, checks, a king walk with rights intact,
# edge pawns, a queenless endgame, pieces hanging
GAMES = [
    # italian: castling, quiet development, home squares, fianchetto-free
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Bg5 h6 Bh4 g5 Bg3 a6 a4 Bg4",
    # sicilian with early queen trade -> queenless middlegame, king walks
    "e4 c5 d4 cxd4 Qxd4 Nc6 Qd1 g6 f4 Bg7 Nf3 Nf6 Nc3 Qa5 Bd3 O-O a3 d6 h3 Bd7",
    # a pawn race into promotion, and en passant on the way
    "e4 d5 exd5 c6 dxc6 Nxc6 d4 e5 d5 Nb4 c4 b5 cxb5 Nd3+ Bxd3 Qa5+ Nc3 Qxb5",
    # queenside castling into pressure, rooks on open files
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Qc2 h6 Bh4 b6 cxd5 exd5 O-O-O c5",
    # a bare endgame: queens off, king activity, passed pawns, edge pawns
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Qf6 Nb5 Bc5 Qe2 Qxf2+ Qxf2 Bxf2+ Kxf2 Kd8 a4 a5 h4 h5",
    # a pawn on the seventh with the straight push blocked, so both
    # capture-promotions are legal and NOT while in check (an earlier attempt
    # ended in check, where promoting is not a legal evasion and never fired)
    "e4 d5 exd5 c6 dxc6 Nf6 c7 e6 Nf3 Be7",
]

# Positions that would take forty moves of SAN to reach. No history, so the
# context's previous-move fields are null on both sides - which the harness
# mirrors by loading the FEN instead of replaying moves.
FENS = [
    # king-and-pawn endgame: 6 pieces, so the endgame ramp is fully on
    "8/5pk1/8/6P1/5K2/8/7P/8 w - - 0 40",
    # rook endgame, 8 pieces, edge pawns and an active king
    "8/1R3pk1/p7/8/5K2/7P/r5P1/8 w - - 2 35",
    # queens off, minor pieces left, a passed pawn to push
    "8/3k4/1p3n2/p1p5/P1P5/1PB2K2/8/8 w - - 0 42",
]


def build_cases():
    cases = []
    for moves in GAMES:
        board = chess.Board()
        played = []
        for san in moves.split():
            try:
                mv = board.parse_san(san)
            except ValueError:
                break
            played.append(mv.uci())
            board.push(mv)
            if len(played) < 8 or board.is_game_over():
                continue
            # derive the context exactly as the real pipeline does
            me = board.turn
            pmt = poc = None
            tmp = chess.Board()
            for u in played:
                m = chess.Move.from_uci(u)
                if tmp.turn == me:
                    pmt = m.to_square
                else:
                    poc = m.to_square if tmp.is_capture(m) else None
                tmp.push(m)
            last = chess.Move.from_uci(played[-1]).to_square
            ctx = Context(board, pmt, poc, last)
            # every legal move, not a 20-move slice: castling and promotion
            # sort late alphabetically and the slice was hiding them
            ucis = [m.uci() for m in board.legal_moves]
            if len(ucis) < 2:
                continue
            # synthetic horizon scores: the parity check is about the feature
            # arithmetic, not about what the engine happens to say
            sh_by = {u: 40 - 7 * i for i, u in enumerate(ucis)}
            sh_by[ucis[-1]] = None          # exercise the imputation branch
            sh, best_sh = horizon_scores(ucis, sh_by)
            for i, (u, s) in enumerate(zip(ucis, sh)):
                cases.append({
                    "moves": list(played), "uci": u, "rank": i,
                    "sh": s, "bestSh": best_sh,
                    "x": [round(float(v), 9) for v in
                          features(board, chess.Move.from_uci(u), ctx, s, best_sh, i)],
                })
    for fen in FENS:
        board = chess.Board(fen)
        ctx = Context(board, None, None, None)
        ucis = [m.uci() for m in board.legal_moves]
        if len(ucis) < 2:
            continue
        sh_by = {u: 30 - 5 * i for i, u in enumerate(ucis)}
        sh, best_sh = horizon_scores(ucis, sh_by)
        for i, (u, s_) in enumerate(zip(ucis, sh)):
            cases.append({
                "fen": fen, "uci": u, "rank": i, "sh": s_, "bestSh": best_sh,
                "x": [round(float(v), 9) for v in
                      features(board, chess.Move.from_uci(u), ctx, s_, best_sh, i)],
            })
    return cases


def main():
    cases = build_cases()
    nz = [sum(1 for c in cases if c["x"][k]) for k in range(N_FEATURES)]
    never = [FEATURE_NAMES[k] for k, n in enumerate(nz) if n == 0]
    print(f"{len(cases)} candidate vectors from {len(GAMES)} games, "
          f"{N_FEATURES} features")
    if never:
        print("COVERAGE FAILURE - these features are never exercised, so parity "
              "over them proves nothing:")
        for name in never:
            print("  -", name)
        return 1

    path = os.path.join(tempfile.mkdtemp(prefix="bb-parity-"), "cases.json")
    with open(path, "w") as f:
        json.dump(cases, f)
    node = subprocess.run(["node", os.path.join(HERE, "parity_v9.mjs"), path],
                          capture_output=True, text=True)
    sys.stdout.write(node.stdout)
    if node.returncode != 0 or "0 mismatching" not in node.stdout:
        sys.stderr.write(node.stderr)
        print("PARITY FAILED - features_v9.py and docs/app.js disagree")
        return 1
    print("parity OK: every feature exercised and every vector agrees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
