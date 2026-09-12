#!/usr/bin/env python3
"""Build the Burke Bot opening book from chess.com games.

Fetches every monthly archive for the account, keeps rapid games against
humans, merges any manually exported PGNs from manual-pgn/ (chess.com's
public API never includes vs-computer games, so bot games arrive that way),
and writes docs/book.json + docs/stats.json for the site to consume.

Run from the repo root: python3 pipeline/build.py
"""

import collections
import glob
import io
import json
import math
import os
import sys
import urllib.request

import chess
import chess.pgn

USERNAME = "burkeley"
API_BASE = f"https://api.chess.com/pub/player/{USERNAME}"
USER_AGENT = "burke-bot build script (+https://aburke225.github.io)"
MAX_BOOK_PLY = 30  # keep the book to the first 15 full moves
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(REPO_ROOT, "docs")
MANUAL_DIR = os.path.join(REPO_ROOT, "manual-pgn")


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def api_games():
    """Yield (pgn_text, headers_hint) for every qualifying chess.com game."""
    archives = fetch_json(f"{API_BASE}/games/archives")["archives"]
    for url in archives:
        month = fetch_json(url)
        for g in month.get("games", []):
            if g.get("rules") != "chess":
                continue
            if g.get("time_class") != "rapid":
                continue
            if "pgn" not in g:
                continue
            yield g["pgn"], "api"


def manual_games():
    """Yield games from manual-pgn/*.pgn (vs-computer exports, any speed)."""
    for path in sorted(glob.glob(os.path.join(MANUAL_DIR, "*.pgn"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        yield from split_pgn(text)


def split_pgn(text):
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        yield game, "manual"


def to_game(item, source):
    if source == "api":
        return chess.pgn.read_game(io.StringIO(item))
    return item  # manual path already yields parsed games


def my_color(game):
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    if white == USERNAME:
        return chess.WHITE
    if black == USERNAME:
        return chess.BLACK
    return None


def my_result(game, color):
    result = game.headers.get("Result", "*")
    if result == "1/2-1/2":
        return "d"
    if result == "1-0":
        return "w" if color == chess.WHITE else "l"
    if result == "0-1":
        return "l" if color == chess.WHITE else "w"
    return None


def book_key(board):
    """Transposition-aware position key: FEN board, turn, and castling only.

    The en-passant field is dropped because chess.js and python-chess write
    it differently (always-after-double-push vs only-when-capturable), which
    would make the frontend miss book positions. Book moves are re-checked
    for legality in the browser before they are played.
    """
    return " ".join(board.fen().split(" ")[:3])


def main():
    sources = [(pgn, src) for pgn, src in api_games()]
    manual = list(manual_games())
    print(f"api rapid games: {len(sources)}, manual games: {len(manual)}")

    book = collections.defaultdict(dict)  # key -> uci -> stats
    stats = {
        "username": "Burkeley",
        "games": 0,
        "as_white": 0,
        "as_black": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "manual_games": 0,
        "openings_white": collections.Counter(),
        "openings_black": collections.Counter(),
        "last_game": "",
    }

    seen_links = set()
    skipped = 0
    bot_games = []  # (opponent elo, my score) from the vs-computer exports
    cache = []  # every ingested game as {color, moves:[uci]} for the style trainer

    for item, source in sources + manual:
        game = to_game(item, source)
        if game is None:
            skipped += 1
            continue
        link = game.headers.get("Link", "")
        if link and link in seen_links:
            continue  # manual export overlapping the API set
        if link:
            seen_links.add(link)

        color = my_color(game)
        result = my_result(game, color) if color is not None else None
        if color is None or result is None:
            skipped += 1
            continue

        stats["games"] += 1
        if source == "manual":
            stats["manual_games"] += 1
            opp_elo = game.headers.get("BlackElo" if color == chess.WHITE else "WhiteElo", "?")
            if opp_elo.isdigit():
                bot_games.append((int(opp_elo), {"w": 1.0, "d": 0.5, "l": 0.0}[result]))
        stats["as_white" if color == chess.WHITE else "as_black"] += 1
        stats["wins" if result == "w" else "losses" if result == "l" else "draws"] += 1

        eco_url = game.headers.get("ECOUrl", "")
        if eco_url:
            name = eco_url.rstrip("/").split("/")[-1].replace("-", " ")
            # keep the family name, drop the deep variation tail
            name = " ".join(name.split(" ")[:4]).split(" 1.")[0].split(" 2.")[0].split(" 3.")[0].strip()
            key = "openings_white" if color == chess.WHITE else "openings_black"
            stats[key][name] += 1

        date = game.headers.get("UTCDate", game.headers.get("Date", ""))
        if date and date.replace(".", "-") > stats["last_game"]:
            stats["last_game"] = date.replace(".", "-")

        cache.append({"color": "w" if color == chess.WHITE else "b",
                      "moves": [m.uci() for m in game.mainline_moves()]})

        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= MAX_BOOK_PLY:
                break
            if board.turn == color:
                key = book_key(board)
                uci = move.uci()
                entry = book[key].get(uci)
                if entry is None:
                    entry = {"san": board.san(move), "n": 0, "w": 0, "d": 0, "l": 0}
                    book[key][uci] = entry
                entry["n"] += 1
                entry[result] += 1
            try:
                board.push(move)
            except (ValueError, AssertionError):
                break

    stats["openings_white"] = stats["openings_white"].most_common(6)
    stats["openings_black"] = stats["openings_black"].most_common(6)
    stats["book_positions"] = len(book)

    # games played against the bot on the site (captured by the Worker; only
    # Andrew's own, flagged games are ever stored — and only HIS side's moves
    # go into the book, never the bot's replies)
    stats["site_games"] = 0
    site_path = os.path.join(REPO_ROOT, "pipeline", "site-games.json")
    if os.path.exists(site_path):
        with open(site_path) as f:
            site_games = json.load(f)
        for g in site_games:
            color = chess.WHITE if g.get("color") == "w" else chess.BLACK
            result = g.get("result")
            moves = g.get("moves", [])
            if result not in ("w", "l", "d") or len(moves) < 6:
                continue
            stats["games"] += 1
            stats["site_games"] += 1
            stats["as_white" if color == chess.WHITE else "as_black"] += 1
            stats["wins" if result == "w" else "losses" if result == "l" else "draws"] += 1
            cache.append({"color": g["color"], "moves": moves})
            board = chess.Board()
            for ply, uci in enumerate(moves):
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    break
                if move not in board.legal_moves:
                    break
                if ply < MAX_BOOK_PLY and board.turn == color:
                    key = book_key(board)
                    entry = book[key].get(uci)
                    if entry is None:
                        entry = {"san": board.san(move), "n": 0, "w": 0, "d": 0, "l": 0}
                        book[key][uci] = entry
                    entry["n"] += 1
                    entry[result] += 1
                board.push(move)
        print(f"site games ingested: {stats['site_games']}")

    # performance rating vs the chess.com bots (their scale runs hotter than
    # human ratings, so this is quoted on the site as "bot scale")
    if bot_games:
        n = len(bot_games)
        avg = sum(e for e, _ in bot_games) / n
        s = sum(sc for _, sc in bot_games) / n
        s = min(max(s, 1 / (2 * n)), 1 - 1 / (2 * n))  # keep the log finite
        stats["bot_scale_strength"] = int(round((avg - 400 * math.log10(1 / s - 1)) / 5) * 5)
    else:
        stats["bot_scale_strength"] = None

    os.makedirs(WEB_DIR, exist_ok=True)
    with open(os.path.join(WEB_DIR, "book.json"), "w") as f:
        json.dump({k: book[k] for k in sorted(book)}, f, separators=(",", ":"), sort_keys=True)
    with open(os.path.join(WEB_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    with open(os.path.join(REPO_ROOT, "pipeline", "games-cache.json"), "w") as f:
        json.dump(cache, f, separators=(",", ":"))

    print(f"games used: {stats['games']} (skipped {skipped}), "
          f"book positions: {len(book)}, last game: {stats['last_game']}")


if __name__ == "__main__":
    sys.exit(main())
