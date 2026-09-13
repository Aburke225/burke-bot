# Burke Bot

A chess bot that plays like me — built from every game I've played on
[chess.com/Burkeley](https://www.chess.com/member/Burkeley).

**Play it:** https://aburke225.github.io/burke-bot/

## How it works

- `pipeline/build.py` downloads my full chess.com game history (public API,
  rapid games vs humans), merges any PGN files in `manual-pgn/` (vs-computer
  games — the API never includes those, so they're exported by hand), and
  builds `docs/book.json`: every position I've faced, with the moves I played
  there and how often.
- While the game is inside that book, the bot samples its move from my real
  frequencies — if I play 1. e4 87% of the time, so does the bot.
- Once the game leaves my games, Stockfish (single-threaded WASM) proposes
  its top 5 moves and a style model picks among them: a conditional logit
  trained on my games via behavior cloning (`pipeline/train_style.py`).
  Every candidate move scores 30 features — 15 static ones that are always
  on, and 15 variable ones where the trainer tries all 32,768 subsets and
  keeps whichever predicts my moves best on held-out games (with a switching
  margin, so near-ties don't churn the model). `docs/style.json` ships the
  winner. If the model fails to load, a strength-capped engine (skill 4,
  depth 8 — roughly my level) takes over instead.
- Games played against the bot on the site (only mine, flagged by sign-in)
  are captured and folded back in.
- A daily GitHub Action checks for new games and, only when it finds any,
  re-runs the pipeline and retrains the model.

## Site

`docs/` is the whole site — static files on GitHub Pages, no server, no build
step. Cache-busted assets (`style.css?v=N`, `app.js?v=N`): bump N on every change.

## Rebuilding by hand

```bash
python3 -m pip install python-chess numpy
python3 pipeline/build.py
python3 pipeline/train_style.py  # needs a stockfish binary on PATH
```

## Vendored libraries

- [cm-chessboard](https://github.com/shaack/cm-chessboard) (MIT) — the board UI
- [chess.js](https://github.com/jhlywa/chess.js) (BSD-2) — rules and legality
- [stockfish.js 10](https://github.com/nmrugg/stockfish.js) (GPLv3) — the
  out-of-book engine
