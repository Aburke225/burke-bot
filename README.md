# burkeley bot

A chess bot that plays like me — built from every game I've played on
[chess.com/Burkeley](https://www.chess.com/member/Burkeley).

**Play it:** https://aburke225.github.io/burkeley-bot/

## How it works

- `pipeline/build.py` downloads my full chess.com game history (public API,
  rapid games vs humans), merges any PGN files in `manual-pgn/` (vs-computer
  games — the API never includes those, so they're exported by hand), and
  builds `docs/book.json`: every position I've faced, with the moves I played
  there and how often.
- While the game is inside that book, the bot samples its move from my real
  frequencies — if I play 1. e4 87% of the time, so does the bot.
- Once the game leaves my games, a strength-capped Stockfish
  (single-threaded WASM, skill level 3, depth 6 — roughly my level) takes over.
- A daily GitHub Action re-runs the pipeline, so new games teach the bot
  automatically.

## Site

`docs/` is the whole site — static files on GitHub Pages, no server, no build
step. Cache-busted assets (`style.css?v=N`, `app.js?v=N`): bump N on every change.

## Rebuilding by hand

```bash
python3 -m pip install python-chess
python3 pipeline/build.py
```

## Vendored libraries

- [cm-chessboard](https://github.com/shaack/cm-chessboard) (MIT) — the board UI
- [chess.js](https://github.com/jhlywa/chess.js) (BSD-2) — rules and legality
- [stockfish.js 10](https://github.com/nmrugg/stockfish.js) (GPLv3) — the
  out-of-book engine
