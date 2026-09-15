#!/usr/bin/env python3
"""Engine pass for the style model: cache RAW analysis, not features.

The v8 pipeline baked feature extraction into the engine pass, so changing a
feature meant re-running the engine over every game. Measured, that was the
wrong split: the engine costs ~11ms per position and the python feature loop
costs far more, so this stage now stores only what the engine knows and
feature extraction became a separate, cheap, re-runnable pass.

Two evaluations are recorded per candidate:
  cp  - at POOL_DEPTH, the honest strength of the move
  sh  - at HORIZON_DEPTH, roughly what I can see over the board
The gap between them is the part of a move's badness that is invisible to me,
which is exactly the class of mistake the bot has never been able to make:
a quiet move that only loses material three to five moves later.

Writes pipeline/raw-cache.json. Run from the repo root.
"""
import json, os, sys
import chess, chess.engine
from train_style import CACHE, ENGINE_CMD, MIN_PLY, REPO_ROOT

POOL_DEPTH = 5      # candidate strength, and the depth the site plays at
HORIZON_DEPTH = 2   # my perceptual horizon - what a move LOOKS like to me
MULTIPV = 20        # wide enough to contain the moves I really play: at 10 the
                    # pool misses 9% of my moves, at 20 it misses 2%
RAW = os.path.join(REPO_ROOT, "pipeline", "raw-cache.json")


def main():
    if not ENGINE_CMD:
        sys.exit("no engine available")
    games = json.load(open(CACHE))
    out = []
    with chess.engine.SimpleEngine.popen_uci(ENGINE_CMD) as eng:
        for gi, g in enumerate(games):
            color = chess.WHITE if g["color"] == "w" else chess.BLACK
            board = chess.Board()
            prev_my_to = prev_my_from = prev_opp_cap = last_to = None
            for ply, uci in enumerate(g["moves"]):
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    break
                if move not in board.legal_moves:
                    break
                if ply >= MIN_PLY and board.turn == color and not board.is_game_over():
                    deep = eng.analyse(board, chess.engine.Limit(depth=POOL_DEPTH), multipv=MULTIPV)
                    shal = eng.analyse(board, chess.engine.Limit(depth=HORIZON_DEPTH), multipv=MULTIPV)
                    sh_by = {}
                    for info in shal:
                        if info.get("pv"):
                            sh_by[info["pv"][0].uci()] = info["score"].pov(board.turn).score(mate_score=10000)
                    cands = []
                    for info in deep:
                        if not info.get("pv"):
                            continue
                        u = info["pv"][0].uci()
                        cands.append({"uci": u,
                                      "cp": info["score"].pov(board.turn).score(mate_score=10000),
                                      "sh": sh_by.get(u)})
                    if len(cands) >= 2:
                        out.append({"gi": gi, "ply": ply, "fen": board.fen(), "played": uci,
                                    "prev_my_to": prev_my_to, "prev_my_from": prev_my_from,
                                    "prev_opp_cap_to": prev_opp_cap, "last_move_to": last_to,
                                    "cands": cands})
                if board.turn == color:
                    prev_my_to, prev_my_from = move.to_square, move.from_square
                else:
                    prev_opp_cap = move.to_square if board.is_capture(move) else None
                last_to = move.to_square
                board.push(move)
            if (gi + 1) % 50 == 0:
                print(f"  {gi + 1}/{len(games)} games, {len(out)} decision points", flush=True)
    covered = sum(1 for p in out if any(c["uci"] == p["played"] for c in p["cands"]))
    print(f"\n{len(out)} decision points; my move is in the {MULTIPV}-wide pool "
          f"{100 * covered / len(out):.1f}% of the time")
    json.dump({"fmt": "raw1", "engine": os.path.basename(str(ENGINE_CMD[-1])),
               "pool_depth": POOL_DEPTH, "horizon_depth": HORIZON_DEPTH,
               "multipv": MULTIPV, "positions": out}, open(RAW, "w"))
    print(f"wrote {RAW}")


if __name__ == "__main__":
    main()
