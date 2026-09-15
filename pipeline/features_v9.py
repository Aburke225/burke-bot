#!/usr/bin/env python3
"""The v9 feature contract: 55 numbers per candidate move.

This file is one half of a CONTRACT. The other half is moveFeaturesV9() in
docs/app.js, and the two must produce identical vectors for the same position
or the bot plays a different game than the one it was trained on. Nothing here
may depend on python-chess behaviour that chess.js does not share.

The central change from v8: the model is only allowed to see what Andrew can
see. Every score-derived feature is built from the DEPTH-2 evaluation (his
horizon), never the depth-5 one. Measured over his games, his choices respond
to how bad a move LOOKS at depth 2 (coefficient -0.62) and not at all to the
badness only a deeper search reveals (-0.029, sd 0.020; adding it moves
held-out likelihood by +0.0000). That gap is the whole point: 1.4% of his moves
carry a pawn or more of hidden badness and account for 13.5% of his total
error - the quiet move that only loses material three or four moves later.
Feed the model the deep score and it avoids exactly the mistakes he makes.

Feature order IS the contract. Never renumber; append only.
"""

import math

import chess

N_FEATURES = 55

# P N B R Q K - the king is 0 as a VICTIM (it is never captured)...
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
# ...but 100 as an ATTACKER. v8 used 0 here, which made every defended piece
# standing next to the enemy king look hanging (the king was the "cheapest
# attacker" at value 0). Indices 35-42 all depend on this fix.
ATTACKER_VAL = dict(PIECE_VAL)
ATTACKER_VAL[chess.KING] = 100

HOME_SQUARES = {
    (chess.WHITE, chess.KNIGHT): {chess.B1, chess.G1},
    (chess.WHITE, chess.BISHOP): {chess.C1, chess.F1},
    (chess.WHITE, chess.ROOK): {chess.A1, chess.H1},
    (chess.BLACK, chess.KNIGHT): {chess.B8, chess.G8},
    (chess.BLACK, chess.BISHOP): {chess.C8, chess.F8},
    (chess.BLACK, chess.ROOK): {chess.A8, chess.H8},
}
FIANCHETTO = {chess.WHITE: {chess.B2, chess.G2}, chess.BLACK: {chess.B7, chess.G7}}


def en_prise(board, sq, owner):
    """Attacked, and not adequately defended - the king-fixed definition.

    True when an enemy piece attacks the square AND either nothing of mine
    defends it, or the cheapest enemy attacker is worth less than the piece
    standing there. A king attacker counts as 100, so a defended piece beside
    the enemy king is NOT hanging.
    """
    attackers = board.attackers(not owner, sq)
    if not attackers:
        return False
    if not board.attackers(owner, sq):
        return True
    piece = board.piece_at(sq)
    if piece is None:
        return False
    cheapest = min(ATTACKER_VAL[board.piece_at(a).piece_type] for a in attackers)
    return cheapest < PIECE_VAL[piece.piece_type]


def chebyshev(a, b):
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))


class Context:
    """Everything that is true of the POSITION rather than of one candidate.

    Computed once per decision point; the per-candidate loop then does almost
    no board scanning. docs/app.js must build the same thing in
    decisionContext().
    """

    def __init__(self, board, prev_my_to=None, prev_opp_cap_to=None, last_move_to=None):
        me = board.turn
        opp = not me
        self.me = me
        self.prev_my_to = prev_my_to
        self.prev_opp_cap_to = prev_opp_cap_to
        self.last_move_to = last_move_to
        self.in_check = board.is_check()
        self.can_castle = bool(board.castling_rights & (
            chess.BB_RANK_1 if me == chess.WHITE else chess.BB_RANK_8))

        pieces = board.piece_map()
        self.piece_count = len(pieces)
        npm = 0
        mine = theirs = 0
        for sq, pc in pieces.items():
            v = PIECE_VAL[pc.piece_type]
            if pc.piece_type not in (chess.PAWN, chess.KING):
                npm += v
            if pc.color == me:
                mine += v
            else:
                theirs += v
        self.npm = npm
        self.phase = 1.0 - self.piece_count / 32.0
        self.imbalance = math.sqrt(min(abs(mine - theirs), 9) / 9.0)

        # most advanced own pawn per file (index 31) and enemy pawn ranks per
        # file (index 32)
        self.own_pawn_ranks = {f: [] for f in range(8)}
        self.enemy_pawn_ranks = {f: [] for f in range(8)}
        for sq in board.pieces(chess.PAWN, me):
            self.own_pawn_ranks[chess.square_file(sq)].append(chess.square_rank(sq))
        for sq in board.pieces(chess.PAWN, opp):
            self.enemy_pawn_ranks[chess.square_file(sq)].append(chess.square_rank(sq))

        self.my_king = board.king(me)
        self.enemy_king = board.king(opp)

        # index 47: distinct enemy pieces bearing on my king's neighbourhood
        self.king_exposure = 0
        if self.my_king is not None:
            seen = set()
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[self.my_king]):
                seen |= set(board.attackers(opp, sq))
            self.king_exposure = len(seen)

        # index 48: the six squares in front of a castled king
        self.shield = set()
        self.castled = False
        if self.my_king is not None:
            kf, kr = chess.square_file(self.my_king), chess.square_rank(self.my_king)
            home_rank = 0 if me == chess.WHITE else 7
            second = 1 if me == chess.WHITE else 6
            self.castled = kr in (home_rank, second) and kf in (0, 1, 2, 5, 6, 7)
            step = 1 if me == chess.WHITE else -1
            for f in (kf - 1, kf, kf + 1):
                if not 0 <= f <= 7:
                    continue
                for d in (1, 2):
                    r = kr + step * d
                    if 0 <= r <= 7:
                        self.shield.add(chess.square(f, r))

        # index 50: the enemy king's zone, and who already bears on it
        self.zone = []
        self.zone_before = 0
        if self.enemy_king is not None:
            self.zone = list(chess.SquareSet(chess.BB_KING_ATTACKS[self.enemy_king]))
            seen = set()
            for sq in self.zone:
                seen |= set(board.attackers(me, sq))
            self.zone_before = len(seen)

        # index 40: pieces of mine the opponent's LAST move put en prise
        self.fresh = []
        if last_move_to is not None:
            for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                for sq in board.pieces(pt, me):
                    if last_move_to in board.attackers(opp, sq) and en_prise(board, sq, me):
                        self.fresh.append(sq)

        # index 13: is a cheapest-recapture available, and is it forced?
        self.recap_n = 0
        self.recap_min = None
        if prev_opp_cap_to is not None:
            vals = [PIECE_VAL[board.piece_type_at(m.from_square)]
                    for m in board.legal_moves
                    if m.to_square == prev_opp_cap_to and board.is_capture(m)]
            self.recap_n = len(vals)
            self.recap_min = min(vals) if vals else None


def horizon_scores(cands, sh_by_uci):
    """Impute missing depth-2 scores, then return (sh list, best_sh).

    A candidate the depth-2 search did not report is treated as looking at
    least as bad as the worst move it did report. Imputation happens BEFORE
    best_sh is taken - doing it after diverges on the 3.3% of candidates that
    need it.
    """
    present = [sh_by_uci[u] for u in cands if sh_by_uci.get(u) is not None]
    floor = min(present) if present else None
    out = []
    for u in cands:
        v = sh_by_uci.get(u)
        out.append(floor if v is None else v)
    usable = [v for v in out if v is not None]
    return out, (max(usable) if usable else None)


def features(board, move, ctx, sh, best_sh, rank):
    """The 55-vector for one candidate. `board` is the PRE-move position."""
    x = [0.0] * N_FEATURES
    me = ctx.me
    opp = not me

    mover = board.piece_type_at(move.from_square)
    is_capture = board.is_capture(move)
    is_castle = board.is_castling(move)
    gives_check = board.gives_check(move)
    to_sq, from_sq = move.to_square, move.from_square
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    forward_sign = 1 if me == chess.WHITE else -1

    # ---- 0-2: perception. His horizon, never the deep score. ----
    if best_sh is not None and sh is not None:
        raw = max(0.0, (best_sh - sh) / 100.0)
        x[0] = math.log1p(min(raw, 5.0))          # clamp BEFORE the log
        x[2] = 1.0 / (1.0 + math.exp(-max(-2000, min(2000, sh)) / 150.0))
    quiet = 0.0 if (is_capture or gives_check) else 1.0
    x[1] = x[0] * quiet

    # ---- 3-13: the engine's ordering, and the capture family ----
    x[3] = rank / 4.0
    forcing = 1.0 if (is_capture or gives_check or move.promotion) else 0.0
    x[4] = forcing
    x[5] = 1.0 if is_capture else 0.0
    x[6] = 1.0 if gives_check else 0.0
    x[7] = 1.0 if move.promotion else 0.0
    x[8] = 1.0 if is_castle else 0.0
    if is_capture:
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(to_sq)
        x[9] = PIECE_VAL[victim] / 9.0
    to_attacked = board.is_attacked_by(opp, to_sq)
    x[10] = 1.0 if (is_capture and to_attacked) else 0.0
    if is_capture and to_attacked:
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(to_sq)
        x[11] = 1.0 if PIECE_VAL[victim] < PIECE_VAL[mover] else 0.0
    x[12] = 1.0 if (is_capture and ctx.prev_opp_cap_to == to_sq) else 0.0
    if (is_capture and ctx.prev_opp_cap_to == to_sq and ctx.recap_n > 1
            and ctx.recap_min is not None and PIECE_VAL[mover] == ctx.recap_min):
        x[13] = 1.0

    # ---- 14-22: who moved, and which way ----
    x[14 + [chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING].index(mover)] = 1.0
    x[20] = (abs(tf - 3.5) + abs(tr - 3.5)) / 7.0
    forward = (tr - fr) * forward_sign
    x[21] = 1.0 if forward > 0 else 0.0
    x[22] = 1.0 if forward < 0 else 0.0

    # ---- 23-32: shape of the square and the structure ----
    x[23] = x[22] if mover in (chess.KNIGHT, chess.BISHOP) else 0.0
    x[24] = 1.0 if (mover == chess.KNIGHT and (tf in (0, 7) or tr in (0, 7))) else 0.0
    x[25] = 1.0 if (mover == chess.PAWN and abs(tr - fr) == 2) else 0.0
    if mover == chess.PAWN and not is_capture:
        back = tr - forward_sign
        if 0 <= back <= 7:
            for f in (tf - 1, tf + 1):
                if 0 <= f <= 7 and board.piece_at(chess.square(f, back)) == \
                        chess.Piece(chess.PAWN, me):
                    x[26] = 1.0
                    break
    x[27] = 1.0 if (mover == chess.PAWN and not is_capture and ff in (0, 7)) else 0.0
    ramp = max(0.0, (12 - ctx.piece_count) / 10.0)
    x[28] = ramp * (1.0 if (mover == chess.PAWN and not is_capture) else 0.0)
    x[29] = 1.0 if from_sq in HOME_SQUARES.get((me, mover), ()) else 0.0
    x[30] = 1.0 if (mover == chess.BISHOP and to_sq in FIANCHETTO[me]) else 0.0
    if mover == chess.ROOK:
        # a rook stepping onto a file its OWN pawn still blocks ahead of it
        for r in ctx.own_pawn_ranks[tf]:
            if (r - tr) * forward_sign > 0:
                x[31] = 1.0
                break
    # a square no enemy pawn attacks now or could ever attack
    if not any(to_sq in board.attacks(s) for s in board.pieces(chess.PAWN, opp)):
        reachable = False
        for f in (tf - 1, tf + 1):
            if not 0 <= f <= 7:
                continue
            for r in ctx.enemy_pawn_ranks[f]:
                if (r - tr) * forward_sign > 0:
                    reachable = True
                    break
            if reachable:
                break
        x[32] = 0.0 if reachable else 1.0

    # ---- 33-42: danger, given and taken ----
    x[33] = 1.0 if board.is_attacked_by(opp, from_sq) else 0.0
    x[34] = 1.0 if to_attacked else 0.0
    danger = en_prise(board, from_sq, me)
    mover_val = PIECE_VAL[mover]

    board.push(move)
    try:
        landed_hanging = en_prise(board, to_sq, me)
        x[35] = 1.0 if landed_hanging else 0.0
        count = 1 if landed_hanging else 0
        others = False
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(pt, me):
                if sq == to_sq:
                    continue
                if en_prise(board, sq, me):
                    others = True
                    count += 1
        x[36] = 1.0 if others else 0.0
        x[37] = float(count)
        x[39] = (mover_val / 9.0) if (danger and not landed_hanging) else 0.0
        for sq in ctx.fresh:
            if sq == from_sq:
                continue
            pc = board.piece_at(sq)
            if pc is not None and pc.color == me and en_prise(board, sq, me):
                x[40] = 1.0
                break
        # what the piece now attacks from where it landed
        best_any = 0
        best_loose = 0
        for sq in board.attacks(to_sq):
            pc = board.piece_at(sq)
            if pc is None or pc.color != opp or pc.piece_type == chess.KING:
                continue
            v = PIECE_VAL[pc.piece_type]
            best_any = max(best_any, v)
            if not board.attackers(opp, sq) or v > mover_val:
                best_loose = max(best_loose, v)
        x[41] = 1.0 if (best_any >= 3 and not landed_hanging) else 0.0
        x[42] = best_loose / 9.0
        if gives_check and ctx.enemy_king is not None:
            if (ctx.enemy_king in board.attackers(opp, to_sq)
                    and not board.is_attacked_by(me, to_sq)):
                x[49] = 1.0
        if ctx.zone:
            seen = set()
            for sq in ctx.zone:
                seen |= set(board.attackers(me, sq))
            x[50] = 1.0 if len(seen) > ctx.zone_before else 0.0
    finally:
        board.pop()

    if mover in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) and not is_capture:
        for s in board.pieces(chess.PAWN, opp):
            if to_sq in board.attacks(s):
                x[38] = 1.0
                break

    # ---- 43-44: where the action was ----
    if ctx.last_move_to is not None:
        x[43] = (6 - min(chebyshev(to_sq, ctx.last_move_to), 6)) / 6.0
    if ctx.prev_my_to is not None:
        x[44] = (4 - min(chebyshev(to_sq, ctx.prev_my_to), 4)) / 4.0

    # ---- 45-50: kings ----
    king_move = mover == chess.KING and not is_castle
    if king_move and not ctx.in_check:
        if ctx.can_castle:
            x[45] = 1.0
        else:
            x[46] = (ctx.npm / 62.0)
            x[47] = min(ctx.king_exposure, 3) / 3.0
    x[48] = 1.0 if (mover == chess.PAWN and not is_capture
                    and from_sq in ctx.shield and ctx.castled) else 0.0

    # ---- 51-54: the interactions that survive the softmax ----
    x[51] = ctx.phase * forcing
    x[52] = ctx.imbalance * forcing
    x[53] = ctx.phase * (1.0 if king_move else 0.0)
    x[54] = ctx.imbalance * x[0]
    return x


FEATURE_NAMES = [
    "horizon_loss_log", "horizon_loss_log_quiet", "horizon_winprob", "engine_rank",
    "forcing", "capture", "gives_check", "promotion", "castle", "captured_value",
    "capture_of_defended", "loses_exchange", "recapture", "recapture_with_cheapest",
    "is_pawn", "is_knight", "is_bishop", "is_rook", "is_queen", "is_king",
    "center_distance", "forward", "retreat", "minor_piece_retreat", "knight_to_edge",
    "pawn_double_step", "pawn_supported_push", "quiet_edge_pawn",
    "endgame_ramp_x_pawn_push", "home_on_start_square", "bishop_to_fianchetto",
    "rook_blocked_file", "square_no_pawn_can_reach", "from_attacked", "to_attacked",
    "hangs_mover", "leaves_another_hanging", "hangs_count", "hangs_to_pawn",
    "escapes_with_value", "ignores_fresh_threat", "attacks_minor_safely",
    "attacks_undefended_value", "prox_to_last_move", "prox_to_my_last_move",
    "king_walk", "king_walk_free_x_phase", "king_walk_free_x_king_exposure",
    "shield_push_castled", "unsafe_contact_check", "enemy_king_zone_pressure",
    "phase_x_forcing", "imbalance_x_forcing", "phase_x_king_move",
    "imbalance_x_horizon_loss",
]
assert len(FEATURE_NAMES) == N_FEATURES
