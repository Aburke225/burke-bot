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
                         en_prise, features, position_tension)
import os

BOOK = os.path.join(REPO_ROOT, "docs", "book.json")
STYLE = os.path.join(REPO_ROOT, "docs", "style.json")

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required")

AVG_CAP = 1000  # cap per-move loss when averaging, standard ACPL practice
PLAY_TEMP = 0.8  # sampling temperature, matching docs/app.js


def book_key(board):
    return " ".join(board.fen().split(" ")[:3])


def aimless_edge(board, move):
    """Mirrors aimlessEdgePawn in docs/app.js: a quiet edge-pawn push with no
    job (kicks nothing off the square it now attacks, defends nothing, not a
    passer, no storm at their king) while the queens are still on."""
    me = board.turn
    if not (board.pieces(chess.QUEEN, chess.WHITE) and board.pieces(chess.QUEEN, chess.BLACK)):
        return False
    ff = chess.square_file(move.from_square)
    tr = chess.square_rank(move.to_square)
    ahead = range(tr + 1, 8) if me == chess.WHITE else range(0, tr)
    passed = True
    for f in {max(0, ff - 1), ff, min(7, ff + 1)}:
        for r in ahead:
            pc = board.piece_at(chess.square(f, r))
            if pc and pc.piece_type == chess.PAWN and pc.color != me:
                passed = False
    if passed:
        return False
    board.push(move)
    try:
        pawn_atk = chess.BB_PAWN_ATTACKS[me][move.to_square]
        for sq in chess.scan_forward(pawn_atk):
            pc = board.piece_at(sq)
            if pc and pc.color != me:
                return False  # kicks an enemy piece
            if pc and pc.color == me and board.attackers(not me, sq):
                return False  # defends an attacked friend
        # no prophylaxis exemption (covering a square an enemy minor merely eyes
        # is available in 29% of his positions and he takes it 1.8% of the time,
        # against 0.7% for admittedly aimless pushes) and no luft exemption (a
        # step in front of his own castled king: available 10%, played 0.4% -
        # BELOW the aimless baseline, because it weakens the king it "helps")
        my_k, opp_k = board.king(me), board.king(not me)
        if opp_k is not None and my_k is not None and \
                abs(chess.square_file(opp_k) - ff) <= 2 and abs(chess.square_file(my_k) - ff) >= 3:
            return False  # pawn storm at their king
        return True
    finally:
        board.pop()


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
        last_move_to = None  # destination of the previous ply's move
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
                # salience guard, mirroring docs/app.js stylePick: drop candidates
                # that ignore the threat the opponent JUST made, and voluntary
                # king-walks - unless engine-best, or fewer than 2 would remain
                fresh = []
                if last_move_to is not None:
                    for sq, pc in board.piece_map().items():
                        if (pc.color == me and pc.piece_type in
                                (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                                and last_move_to in board.attackers(not me, sq)):
                            fresh.append(sq)
                def guarded(cand, x):
                    if x[20] == 1.0:
                        return True
                    if x[22] == 1.0 and cand.promotion is None and aimless_edge(board, cand):
                        return True
                    board.push(cand)
                    bad = any(sq != cand.from_square and board.piece_at(sq)
                              and board.piece_at(sq).color == me
                              and en_prise(board, sq, me) for sq in fresh)
                    board.pop()
                    return bad
                # capture salience, mirroring docs/app.js: a near-best grab of
                # a queen with a lesser piece pre-empts everything else
                grabs = [i for i, (c, cp, x) in enumerate(cands)
                         if x[2] == 1 and x[14] == 1.0 and x[28] < 1.0
                         and best_cp - cp <= 50]
                if grabs:
                    keep = sorted(set([0] + grabs))
                else:
                    # tail guard: in a healthy position, howlers (>=250cp) are
                    # out; rank-0 always survives so keep is never empty
                    keep = [i for i, (c, cp, x) in enumerate(cands)
                            if i == 0 or ((best_cp <= -200 or best_cp - cp < 250)
                                          and not guarded(c, x))]
                z = np.array([float(np.dot(w, [cands[i][2][j] for j in active])) for i in keep])
                z = z / PLAY_TEMP  # keep in sync with PLAY_TEMP in docs/app.js
                p = np.exp(z - z.max()); p /= p.sum()
                pick = keep[rng.choices(range(len(keep)), weights=p.tolist())[0]]
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
            last_move_to = move.to_square
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
