#!/usr/bin/env python3
"""Audit move quality: does Burke Bot make Andrew's KINDS of mistakes?

Two modes, meant to be compared side by side:
  python3 pipeline/audit.py me        # profile Andrew's real games
  python3 pipeline/audit.py bot [N]   # profile the bot across N self-play games

Both profile every move from ply 8 on: the loss versus the engine's best move
(same vendored engine, same depth as training and play), a severity bucket,
and - for mistakes and blunders - a type tag computed with the same board
logic the model's mistake features use. If the bot plays like Andrew, the two
profiles should look alike.
"""

import collections
import json
import random
import sys

import chess
import chess.engine

from train_style import (CACHE, DEPTH, ENGINE_CMD, MIN_PLY, MULTIPV, REPO_ROOT,
                         features, position_tension)
import os

BOOK = os.path.join(REPO_ROOT, "docs", "book.json")
STYLE = os.path.join(REPO_ROOT, "docs", "style.json")

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required")

AVG_CAP = 1000  # cap per-move loss when averaging, standard ACPL practice


def book_key(board):
    return " ".join(board.fen().split(" ")[:3])


def severity(loss):
    if loss < 50: return "fine"
    if loss < 150: return "inaccuracy"
    if loss < 300: return "mistake"
    return "blunder"


class Profile:
    def __init__(self):
        self.n = 0
        self.loss_sum = 0
        self.sev = collections.Counter()
        self.tags = collections.Counter()

    def add(self, loss, x, mate_allowed):
        self.n += 1
        self.loss_sum += min(loss, AVG_CAP)
        self.sev[severity(loss)] += 1
        if loss >= 150:
            if mate_allowed: self.tags["allowed-mate"] += 1
            if x[25]: self.tags["hung-piece"] += 1
            if x[26]: self.tags["left-hanging"] += 1
            if x[27]: self.tags["bad-trade"] += 1
            if not (mate_allowed or x[25] or x[26] or x[27]):
                self.tags["other"] += 1

    def report(self, label):
        print(f"\n{label}: {self.n} moves, avg loss {self.loss_sum / self.n:.0f}cp")
        for s in ("fine", "inaccuracy", "mistake", "blunder"):
            print(f"  {s:>10}: {100 * self.sev[s] / self.n:5.1f}%")
        print("  mistake+blunder types per 100 moves:")
        for t in ("hung-piece", "left-hanging", "bad-trade", "allowed-mate", "other"):
            print(f"  {t:>13}: {100 * self.tags[t] / self.n:5.2f}")


def move_loss(engine, board, move):
    """(loss cp, feature vector, mate_allowed, cand list) for a move about to be played."""
    infos = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=MULTIPV)
    cands = {}
    best_cp = None
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        cp = info["score"].pov(board.turn).score(mate_score=10000)
        if best_cp is None:
            best_cp = cp
        cands[info["pv"][0]] = cp
    if best_cp is None:
        return None
    if move in cands:
        my_cp = cands[move]
    else:
        board.push(move)
        if board.is_game_over():
            board.pop()
            return None
        info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
        board.pop()
        my_cp = -info["score"].pov(not board.turn).score(mate_score=10000)
    loss = max(0, best_cp - my_cp)
    mate_allowed = my_cp <= -9000 <= best_cp  # walked from survivable into mated
    return loss, my_cp, mate_allowed, cands


def profile_me(engine):
    games = json.load(open(CACHE))
    prof = Profile()
    for gi, g in enumerate(games):
        color = chess.WHITE if g["color"] == "w" else chess.BLACK
        board = chess.Board()
        prev_my_to = prev_my_from = prev_opp_cap_to = None
        for ply, uci in enumerate(g["moves"]):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            if ply >= MIN_PLY and board.turn == color and not board.is_game_over():
                r = move_loss(engine, board, move)
                if r:
                    loss, _, mate_allowed, _ = r
                    x = features(board, move, loss / 100.0, 0, prev_my_to,
                                 prev_opp_cap_to, prev_my_from, position_tension(board))
                    prof.add(loss, x, mate_allowed)
            if board.turn == color:
                prev_my_to, prev_my_from = move.to_square, move.from_square
            else:
                prev_opp_cap_to = move.to_square if board.is_capture(move) else None
            board.push(move)
        if (gi + 1) % 50 == 0:
            print(f"  {gi + 1}/{len(games)} games, {prof.n} moves profiled", flush=True)
    return prof


def profile_bot(engine, n_games):
    book = json.load(open(BOOK))
    style = json.load(open(STYLE))
    w = np.array(style["weights"])
    active = style["active"]
    rng = random.Random(7)
    prof = Profile()
    for game_no in range(n_games):
        board = chess.Board()
        prev = {chess.WHITE: (None, None), chess.BLACK: (None, None)}  # (to, from)
        prev_cap = {chess.WHITE: None, chess.BLACK: None}  # last capture square BY that side
        for ply in range(160):
            if board.is_game_over():
                break
            me = board.turn
            move = None
            entry = book.get(book_key(board))
            if entry:
                legal = {m.uci(): m for m in board.legal_moves}
                cands = [(u, v["n"]) for u, v in entry.items() if u in legal]
                if cands:
                    total = sum(n for _, n in cands)
                    r = rng.random() * total
                    for u, n in cands:
                        r -= n
                        if r <= 0:
                            move = legal[u]
                            break
            analysed = None
            if move is None:
                infos = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=MULTIPV)
                tension = position_tension(board)
                prev_my_to, prev_my_from = prev[me]
                cands, best_cp = [], None
                for rank, info in enumerate(infos):
                    if "pv" not in info or not info["pv"]:
                        continue
                    cand = info["pv"][0]
                    cp = info["score"].pov(me).score(mate_score=10000)
                    if best_cp is None:
                        best_cp = cp
                    x = features(board, cand, (best_cp - cp) / 100.0, rank,
                                 prev_my_to, prev_cap[not me], prev_my_from, tension)
                    cands.append((cand, cp, x))
                if not cands:
                    break
                z = np.array([float(np.dot(w, [x[i] for i in active])) for _, _, x in cands])
                p = np.exp(z - z.max()); p /= p.sum()
                pick = rng.choices(range(len(cands)), weights=p.tolist())[0]
                move, my_cp, x = cands[pick]
                analysed = (max(0, best_cp - my_cp), x,
                            my_cp <= -9000 <= best_cp)
            if ply >= MIN_PLY:
                if analysed:
                    prof.add(*analysed)
                else:
                    r = move_loss(engine, board, move)
                    if r:
                        loss, _, mate_allowed, _ = r
                        prev_my_to, prev_my_from = prev[me]
                        x = features(board, move, loss / 100.0, 0, prev_my_to,
                                     prev_cap[not me], prev_my_from, position_tension(board))
                        prof.add(loss, x, mate_allowed)
            prev[me] = (move.to_square, move.from_square)
            prev_cap[me] = move.to_square if board.is_capture(move) else None
            board.push(move)
        if (game_no + 1) % 10 == 0:
            print(f"  {game_no + 1}/{n_games} games, {prof.n} moves profiled", flush=True)
    return prof


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "me"
    if not ENGINE_CMD:
        sys.exit("no engine available")
    with chess.engine.SimpleEngine.popen_uci(ENGINE_CMD) as engine:
        if mode == "me":
            profile_me(engine).report("ANDREW (real games)")
        elif mode == "bot":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            profile_bot(engine, n).report(f"BURKE BOT ({n} self-play games)")
        else:
            sys.exit("usage: audit.py me | bot [N]")


if __name__ == "__main__":
    main()
