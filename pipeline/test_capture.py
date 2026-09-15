#!/usr/bin/env python3
"""Guard the two data paths that only run when there is new data.

On 2026-09-15 the first real captured game crashed build.py with

    TypeError: list indices must be integers or slices, not str

because the opening counters were converted to lists with .most_common(6)
BEFORE the site-games loop, so note_opening_result() indexed a list with an
opening name. The bug had been latent for a week and every nightly run was
green - because site-games.json was always EMPTY, so the code path that breaks
was never executed. A green run proved nothing.

So this test does the one thing those runs never did: run the real build.main()
with a NON-EMPTY store. It stubs the network but redirects every output into a
temp directory; the ingestion, the opening bookkeeping and the stats
finalisation are the genuine article.

It covers the manual vs-computer PGN path for the same reason. Those games
never come from chess.com's API - he exports them by hand - so that path also
sits idle for weeks at a time, and it is the only source of the rating shown
on the site.

    python3 pipeline/test_capture.py     # exits non-zero on failure
"""

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build  # noqa: E402

# a real game of his, trimmed: 1.e4 c5 is a Sicilian, so the opening
# bookkeeping actually fires on both the api path and the site path
API_PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[White "Burkeley"]
[Black "someone"]
[Result "1-0"]
[UTCDate "2026.01.01"]
[Link "https://www.chess.com/game/live/test-api"]

1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 e5 1-0
"""

# The vs-computer export path. chess.com's public API never returns these, so
# they only arrive when he manually drops a PGN into manual-pgn/ - which gives
# this path exactly the same "only runs when there is new data" character that
# let the site-games bug hide for a week. It is also the ONLY source of
# bot_scale_strength, the rating shown on the site.
MANUAL_PGN = """[Event "Computer Match"]
[Site "Chess.com"]
[White "Burkeley"]
[Black "Wally-BOT"]
[BlackElo "1800"]
[Result "0-1"]
[UTCDate "2026.01.02"]
[Link "https://www.chess.com/game/computer/test-manual"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 0-1
"""

# the shape the Cloudflare Worker really returns - verified against the live
# store: a LIST of {color, id, moves, result, ts}
SITE_GAMES = [
    {"color": "b", "id": "t1", "result": "w", "ts": 1789410815375,
     "moves": ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6",
               "b1c3", "e7e5", "d4b5", "d7d6", "c1g5", "a7a6"]},
    {"color": "w", "id": "t2", "result": "l", "ts": 1789424304231,
     "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6",
               "d2d4", "e5d4", "c3d4", "c5b4"]},
]


def main():
    tmp = tempfile.mkdtemp(prefix="bb-capture-test-")
    web = os.path.join(tmp, "docs")
    pipe = os.path.join(tmp, "pipeline")
    os.makedirs(web)
    os.makedirs(pipe)
    # the opening map is READ from the web dir, and without it the opening
    # bookkeeping - the thing that actually broke - never runs
    shutil.copy(os.path.join(HERE, "..", "docs", "openings.json"),
                os.path.join(web, "openings.json"))
    with open(os.path.join(pipe, "site-games.json"), "w") as f:
        json.dump(SITE_GAMES, f)

    orig = (build.REPO_ROOT, build.WEB_DIR, build.api_games, build.manual_games)
    build.REPO_ROOT, build.WEB_DIR = tmp, web
    build.api_games = lambda: iter([(API_PGN, "api")])
    build.manual_games = lambda: build.split_pgn(MANUAL_PGN)
    try:
        build.main()
    finally:
        build.REPO_ROOT, build.WEB_DIR, build.api_games, build.manual_games = orig

    stats = json.load(open(os.path.join(web, "stats.json")))
    book = json.load(open(os.path.join(web, "book.json")))
    cache = json.load(open(os.path.join(pipe, "games-cache.json")))

    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)

    # 1. the games were ingested at all - the headline regression
    check(stats["site_games"] == 2,
          f"site_games is {stats['site_games']}, expected 2")
    check(stats["games"] == 4, f"total games is {stats['games']}, expected 4")

    # 1b. the manual vs-computer path - the other source that only runs when
    #     he has dropped in new data, and the only thing that feeds the rating
    check(stats["manual_games"] == 1,
          f"manual_games is {stats['manual_games']}, expected 1")
    check(stats["bot_scale_strength"] is not None,
          "bot_scale_strength is None - the manual PGN's Elo never reached the "
          "rating calculation, so the number on the site would go blank")

    # 2. they reached the TRAINING cache, not just the counters
    check(len(cache) == 4, f"games-cache has {len(cache)} games, expected 4")

    # 3. the ordering bug itself: counters must still be Counters while the
    #    site-games loop runs, and lists by the time they are written out
    check(isinstance(stats["openings_white"], list),
          "openings_white should be a list in the written stats")
    check(isinstance(stats["openings_black"], list),
          "openings_black should be a list in the written stats")
    black = dict((name, n) for name, n in stats["openings_black"])
    check(black.get("Sicilian Defense") == 1,
          f"the captured Sicilian did not reach openings_black: {stats['openings_black']}")

    # 4. book_positions counted AFTER the site games added to the book - the
    #    second half of the same ordering bug, which under-reported it
    check(stats["book_positions"] == len(book),
          f"book_positions {stats['book_positions']} != {len(book)} real entries")

    # 5. every source lands in the training cache as whole games
    check(all(len(g["moves"]) >= 6 for g in cache),
          "every cached game should carry its full move list")
    check(sorted(g["color"] for g in cache) == ["b", "w", "w", "w"],
          f"unexpected colours in the cache: {[g['color'] for g in cache]}")

    shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print("CAPTURE PATH TEST FAILED")
        for f in fails:
            print("  -", f)
        return 1
    print(f"capture path OK: {stats['site_games']} site games ingested, "
          f"{stats['games']} total, book {stats['book_positions']} positions, "
          "opening counters intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
