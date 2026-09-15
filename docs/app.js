// Burke Bot — plays from Andrew's real games while the position is known,
// then hands over to a strength-capped engine once we leave them.
// Engine target: ~1200 human / ~1400 chess.com-bot scale, triangulated from
// his official 1002, his self-assessed 1200, and an 11.7% score vs 1600-1800
// chess.com bots (performance 1335 on their scale).

import { INPUT_EVENT_TYPE, COLOR, Chessboard, BORDER_TYPE } from "./vendor/cm-chessboard/src/Chessboard.js"
import { MARKER_TYPE, Markers } from "./vendor/cm-chessboard/src/extensions/markers/Markers.js"
import { PROMOTION_DIALOG_RESULT_TYPE, PromotionDialog } from "./vendor/cm-chessboard/src/extensions/promotion-dialog/PromotionDialog.js"
import { Accessibility } from "./vendor/cm-chessboard/src/extensions/accessibility/Accessibility.js"
import { Chess } from "./vendor/chess.js"

const statusDot = document.getElementById("status-dot")
const statusLabel = document.getElementById("status-label")
const statusLine = document.getElementById("status-line")
const movelistEl = document.getElementById("movelist")

let book = {}
let chess = new Chess()
let botColor = "w"
let inBook = true
let gameId = 0
let styleModel = null
let stdStart = true  // capture only games that began from the standard position
let gameActive = false
const CAPTURE_URL = "https://prompt-yourself-bot.andrewburke225.workers.dev/chess/game"

// search depth for candidates - a contract with pipeline/train_style.py
// (training must analyse at the depth the site plays at; retrain after changing)
const DEPTH = 5
// 20 candidates, not 10: 11.6% of his real moves rank 10th or worse, and those
// carry a disproportionate share of his error. A contract with analyse.py.
const MULTIPV = 20
// his perceptual horizon - a contract with analyse.py's HORIZON_DEPTH
const HORIZON_DEPTH = 2
// The eval bar is NOT the bot's view of the position - it is the honest one,
// so it searches far deeper than the bot plays. The model never touches it:
// the number is Stockfish's own score for the position, full stop. Depth 5
// would be the bot's shallow read, which is the wrong thing to show a player
// judging how the game actually stands. ~70ms per ply, once per move.
// chess.com's analysis eval settles around here; the score is converged by
// this point (depth 18 and 20 agree within a few centipawns in test positions)
const EVAL_DEPTH = 18
// Sampling temperature. 1 plays the fitted distribution exactly, and measured
// across all 58 behaviours at once that is not a compromise - it is the answer.
// Over 9,431 held-out decisions (pipeline/temp_sweep.py):
//   T     log-lik   feature mismatch   bot loss   (mine: 67cp)
//   0.65  -1.8657   0.0522             48cp
//   0.90  -1.7440   0.0149             61cp
//   1.00  -1.7363   0.0004             66cp
//   1.25  -1.7602   0.0345             79cp
// T=1 wins the likelihood, matches my rate on every behaviour to 4 decimals,
// and lands 1cp from my own error rate. That exactness is not luck: a
// conditional logit fitted by maximum likelihood matches the empirical feature
// means exactly at T=1, and any other temperature biases all 58 at once - at
// 0.65 the bot captured 31.5% of the time against my 27.4% and checked 12.2%
// against my 10.3%. Note mean-probability-on-my-move is NOT the metric to tune
// this with: it rises as T falls simply because argmax scores 1 when right.
// keep in sync with PLAY_TEMP in pipeline/audit.py
const PLAY_TEMP = 1.0

// ---------- engine (single-threaded Stockfish 18 lite WASM) ----------
// the same build the retrain pipeline analyses games with (run there via
// node), so the model always chooses among the candidates it trained on

const engine = (() => {
  const worker = new Worker("vendor/stockfish/stockfish-18-lite-single.js")
  let onBest = null
  let lines = {}
  let readyResolve
  const ready = new Promise(res => { readyResolve = res })
  worker.onmessage = (e) => {
    const line = typeof e.data === "string" ? e.data : ""
    if (line === "uciok") {
      worker.postMessage("setoption name Skill Level value 4")
      worker.postMessage("isready")
    } else if (line === "readyok") {
      readyResolve()
    } else if (line.startsWith("info ") && line.includes(" multipv ")) {
      const mpv = /\bmultipv (\d+)/.exec(line)
      const pv = / pv ([a-h][1-8][a-h][1-8][qrbn]?)/.exec(line)
      const cp = / score cp (-?\d+)/.exec(line)
      const mate = / score mate (-?\d+)/.exec(line)
      if (mpv && pv) {
        const m = parseInt(mate ? mate[1] : "0", 10)
        const score = cp ? parseInt(cp[1], 10) : mate ? (m > 0 ? 10000 - m : -10000 - m) : 0
        lines[parseInt(mpv[1], 10)] = { uci: pv[1], cp: score }
      }
    } else if (line.startsWith("info ") && line.includes(" score ") && line.includes(" pv ")) {
      const pv = / pv ([a-h][1-8][a-h][1-8][qrbn]?)/.exec(line)
      const cp = / score cp (-?\d+)/.exec(line)
      const mate = / score mate (-?\d+)/.exec(line)
      if (pv) {
        const m = parseInt(mate ? mate[1] : "0", 10)
        const score = cp ? parseInt(cp[1], 10) : mate ? (m > 0 ? 10000 - m : -10000 - m) : 0
        lines[1] = { uci: pv[1], cp: score }
      }
    } else if (line.startsWith("bestmove") && onBest) {
      const uci = line.split(" ")[1]
      const resolve = onBest
      onBest = null
      resolve({ uci, lines })
    }
  }
  worker.postMessage("uci")
  // one engine serves both move choice and the eval bar, so searches queue up
  let queue = Promise.resolve()
  return {
    ready,
    // with the style model in charge of humanness, search runs clean:
    // full strength, TEN candidate lines for the model to choose among -
    // wide enough to include the genuinely bad moves a human would play
    useMultipv() {
      worker.postMessage("setoption name Skill Level value 20")
      worker.postMessage("setoption name MultiPV value " + MULTIPV)
    },
    // the same position seen from HIS horizon. v9's three strongest features
    // are built from this, not from the depth-5 score: he responds to how bad
    // a move LOOKS two ply out and is measurably blind to the rest.
    horizon(fen) {
      const run = () => new Promise(res => {
        lines = {}
        onBest = res
        worker.postMessage("position fen " + fen)
        worker.postMessage("go depth " + HORIZON_DEPTH)
      })
      const p = queue.then(run)
      queue = p.then(() => {}, () => {})
      return p
    },
    bestMove(fen) {
      const run = () => new Promise(res => {
        lines = {}
        onBest = res
        worker.postMessage("position fen " + fen)
        worker.postMessage("go depth " + DEPTH)
      })
      const p = queue.then(run)
      queue = p.then(() => {}, () => {})
      return p
    },
  }
})()

// The eval bar gets its OWN engine. It searches far deeper than the bot plays
// (~1s per position in the browser), and sharing the move engine's queue would
// make every bot move wait for it. Loaded lazily, so a visitor who never starts
// a game never pays for it. Nothing here touches the style model: the number on
// the bar is Stockfish's own score for the position and nothing else.
const evalEngine = (() => {
  let worker = null, ready = null, onDone = null, score = null
  let queue = Promise.resolve()
  function boot() {
    if (ready) return ready
    worker = new Worker("vendor/stockfish/stockfish-18-lite-single.js")
    let resolveReady
    ready = new Promise(res => { resolveReady = res })
    worker.onmessage = (e) => {
      const line = typeof e.data === "string" ? e.data : ""
      if (line === "uciok") {
        worker.postMessage("setoption name MultiPV value 1")
        worker.postMessage("isready")
      } else if (line === "readyok") {
        resolveReady()
      } else if (line.startsWith("info ") && line.includes(" score ")) {
        const cp = / score cp (-?\d+)/.exec(line)
        const mate = / score mate (-?\d+)/.exec(line)
        if (cp) score = parseInt(cp[1], 10)
        else if (mate) {
          const m = parseInt(mate[1], 10)
          score = m > 0 ? 10000 - m : -10000 - m
        }
      } else if (line.startsWith("bestmove") && onDone) {
        const done = onDone
        onDone = null
        done(score)
      }
    }
    worker.postMessage("uci")
    return ready
  }
  return {
    async score(fen) {
      await boot()
      const run = () => new Promise(res => {
        score = null
        onDone = res
        worker.postMessage("position fen " + fen)
        worker.postMessage("go depth " + EVAL_DEPTH)
      })
      const p = queue.then(run)
      queue = p.then(() => {}, () => {})
      return p
    },
  }
})()

// ---------- helpers ----------

function tryMove(args) {
  try { return chess.move(args) } catch { return null }
}

function bookKey(fen) {
  return fen.split(" ").slice(0, 3).join(" ")
}

function pickBookMove() {
  const entry = book[bookKey(chess.fen())]
  if (!entry) return null
  const legal = new Set(chess.moves({ verbose: true }).map(m => m.from + m.to + (m.promotion || "")))
  const candidates = Object.entries(entry).filter(([uci]) => legal.has(uci))
  if (!candidates.length) return null
  const total = candidates.reduce((s, [, v]) => s + v.n, 0)
  let r = Math.random() * total
  for (const [uci, v] of candidates) {
    r -= v.n
    if (r <= 0) return { uci, san: v.san, n: v.n, total, wins: v.w, draws: v.d }
  }
  return null
}

// the chat browses along with the board: what was said is remembered per ply
// ("thinking" is transient and never remembered)
let statusByPly = {}
let lastLiveStatus = null

function setStatusRaw(kind, label, line) {
  statusDot.className = "dot " + kind
  statusLabel.textContent = label
  statusLine.textContent = line
  // the chat line can wrap to two lines, which grows the panel; the move list
  // has to give that height back or Resign drops below the board
  fitMoveList()
  revealCurrentMove()
}

function setStatus(kind, label, line) {
  setStatusRaw(kind, label, line)
  lastLiveStatus = { kind, label, line }
  if (kind !== "thinking") statusByPly[chess.history().length] = { kind, label, line }
}

// Resign sits under the stats card on a desktop layout, and under the board on
// a stacked one - which is where it has always been on a phone, and the stacked
// layout is deliberately left exactly as it was
const stacked = window.matchMedia("(max-width: 860px)")

function placeControls() {
  const controls = document.querySelector(".controls")
  if (!controls) return
  const host = document.querySelector(stacked.matches ? ".board-col" : ".panel")
  if (host && controls.parentElement !== host) host.appendChild(controls)
}

// The move list grows with the game, but never so far that Resign below it
// drops past the bottom of the board - which on a laptop would put the button
// under the fold. Past that point the list caps and scrolls instead, so rows
// keep their size and every move stays reachable.
const MOVELIST_MIN = 72
function fitMoveList() {
  const boardWrap = document.querySelector(".board-wrap")
  const controls = document.querySelector(".controls")
  const panel = document.querySelector(".panel")
  if (!boardWrap || !controls || !panel) return
  if (stacked.matches || controls.parentElement !== panel) {
    movelistEl.style.removeProperty("--movelist-max")  // stacked: CSS cap applies
    return
  }
  // measure the list unconstrained, then give back exactly the slack (or take
  // back exactly the overflow) between Resign's bottom and the board's bottom
  movelistEl.style.setProperty("--movelist-max", "100vh")
  const natural = movelistEl.getBoundingClientRect().height
  const spare = boardWrap.getBoundingClientRect().bottom - controls.getBoundingClientRect().bottom
  movelistEl.style.setProperty("--movelist-max",
    Math.max(MOVELIST_MIN, Math.floor(natural + spare)) + "px")
}

// scroll the highlighted move into view INSIDE the list - scrollIntoView would
// also scroll the page, which yanks the board around while browsing history
function revealCurrentMove() {
  const cur = movelistEl.querySelector(".cur")
  if (!cur) return
  const box = movelistEl.getBoundingClientRect()
  const item = cur.getBoundingClientRect()
  if (item.top < box.top) movelistEl.scrollTop -= box.top - item.top + 6
  else if (item.bottom > box.bottom) movelistEl.scrollTop += item.bottom - box.bottom + 6
}

// ---------- the two players, above and below the board ----------

// NOT PIECE_VAL - that name belongs to the v9 feature extractor further down,
// and redeclaring it killed the whole script before the board ever built.
const CAP_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9 }

// The challenger wears a person, drawn in the same 2px round-cap language as
// the rest of the site's icons. The bot wears a knight on a board square - the
// site's own mark, and the glyph is pulled from the board's sprite so it is
// literally the same artwork the pieces are drawn from.
const HUMAN_AVATAR =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<circle cx="12" cy="8" r="3.4"/><path d="M4.8 20a7.2 7.2 0 0 1 14.4 0"/></svg>'
const PIECE_VIEWBOX = "6 3 28 34"   // crops the sprite tile's padding
// Burke Bot's portrait: the Chequerhead's face wearing the Warden's frame.
// Rook battlements crown a head whose visor is eight squares of board, and it
// stands on the flared collar and plinth every chess piece stands on. The
// aerial rises out of the crown's middle notch, where the merlon is missing.
// Drawn in the board's own two square colours, so it reads as a piece sitting
// on a dark square.
const BOT_AVATAR = [
  '<svg viewBox="0 0 24 24" aria-hidden="true">',
  '<defs><clipPath id="bb-visor">',
  '<rect x="6.6" y="8.4" width="10.8" height="5.4" rx="1"/>',
  '</clipPath></defs>',
  '<g fill="#e9ecd6">',
  '<rect x="11.35" y="1.5" width="1.3" height="3.4" rx=".65"/>',
  '<path d="M4.6 7.4V4.1h2.5v1.6h1.6V4.1h2.5v1.6h1.6V4.1h2.5v1.6h1.6V4.1h2.5v3.3z"/>',
  '<rect x="4.6" y="6.8" width="14.8" height="10.6" rx="2.4"/>',
  '<rect x="2.3" y="10.2" width="2" height="3.6" rx="1"/>',
  '<rect x="19.7" y="10.2" width="2" height="3.6" rx="1"/>',
  '<path d="M8.2 17.4h7.6l1.1 2.3H7.1z"/>',
  '<path d="M5.9 20.4h12.2a1.1 1.1 0 0 1 1.1 1.1v1.4H4.8v-1.4a1.1 1.1 0 0 1 1.1-1.1z"/>',
  '</g>',
  '<rect x="6.6" y="8.4" width="10.8" height="5.4" rx="1" fill="#2f4a2a"/>',
  '<g fill="#8fc57f" clip-path="url(#bb-visor)">',
  '<rect x="6.6" y="8.4" width="2.7" height="2.7"/>',
  '<rect x="12" y="8.4" width="2.7" height="2.7"/>',
  '<rect x="9.3" y="11.1" width="2.7" height="2.7"/>',
  '<rect x="14.7" y="11.1" width="2.7" height="2.7"/>',
  '</g>',
  '<rect x="8.6" y="14.9" width="6.8" height="1.4" rx=".7" fill="#2f4a2a"/>',
  '<circle cx="12" cy="1.4" r="1.4" fill="#8fc57f"/>',
  '</svg>',
].join("")

function pieceGlyph(colour, type) {
  return '<svg viewBox="' + PIECE_VIEWBOX + '" aria-hidden="true">' +
         '<use href="#' + colour + type + '"/></svg>'
}

function renderPlayers() {
  const avTop = document.getElementById("av-top")
  const avBot = document.getElementById("av-bot")
  if (!avTop || !avBot) return
  let signedIn = false
  try { signedIn = !!localStorage.getItem("bb-key") } catch (e) {}

  // Always on screen, including before a colour has been picked - the board
  // starts in its white-at-the-bottom orientation, which puts the bot on the
  // black side, and that is the pairing the pick overlay is offering.
  if (avTop.dataset.set !== "bot") { avTop.className = "avatar bot"; avTop.innerHTML = BOT_AVATAR; avTop.dataset.set = "bot" }
  document.getElementById("nm-top").textContent = "Burke Bot"

  if (avBot.dataset.set !== "human") { avBot.className = "avatar human"; avBot.innerHTML = HUMAN_AVATAR; avBot.dataset.set = "human" }
  document.getElementById("nm-bot").textContent = signedIn ? "Burke" : "Challenger"
}

function renderCaptures() {
  const capTop = document.getElementById("cap-top")
  const capBottom = document.getElementById("cap-bot")
  if (!capTop || !capBottom) return

  // What each side has taken, counted from the move list rather than by
  // differencing the board: a promotion removes a pawn and adds a queen
  // without anyone capturing anything, and board-differencing would report a
  // phantom captured pawn for the rest of the game.
  const taken = { w: {}, b: {} }
  for (const m of chess.history({ verbose: true })) {
    if (!m.captured) continue
    const victim = m.color === "w" ? "b" : "w"
    taken[victim][m.captured] = (taken[victim][m.captured] || 0) + 1
  }

  // The +N comes from the pieces actually on the board, which IS what a player
  // means by being up material - and unlike the capture list it counts a
  // promotion, which changes the balance by eight points with no capture.
  let mat = 0
  for (const row of chess.board()) {
    for (const sq of row) {
      if (!sq || sq.type === "k") continue
      mat += (sq.color === "w" ? 1 : -1) * CAP_VAL[sq.type]
    }
  }

  const userColor = botColor === "w" ? "b" : "w"
  // a player's row shows what THEY captured, so it carries the enemy's colour
  paintCaptures(capTop, taken[userColor], userColor, botColor === "w" ? mat : -mat)
  paintCaptures(capBottom, taken[botColor], botColor, userColor === "w" ? mat : -mat)
}

function paintCaptures(el, counts, colour, advantage) {
  let html = ""
  for (const type of ["p", "n", "b", "r", "q"]) {
    const n = (counts && counts[type]) || 0
    if (!n) continue
    html += '<span class="cap-group cap-' + colour + '">' + pieceGlyph(colour, type).repeat(n) + "</span>"
  }
  // only the player who is ahead carries a badge, the way chess.com does it
  if (advantage > 0) html += '<span class="adv">+' + advantage + "</span>"
  if (el.innerHTML !== html) el.innerHTML = html
}

function renderMoves() {
  renderPlayers()
  renderCaptures()
  const hist = chess.history()
  movelistEl.innerHTML = ""
  for (let i = 0; i < hist.length; i += 2) {
    const li = document.createElement("li")
    const num = document.createElement("span"); num.className = "n"
    num.textContent = (i / 2 + 1) + "."
    li.appendChild(num)
    // the inner .t span lets the highlight hug the move text while the outer
    // span keeps its min-width so the columns stay aligned
    const w = document.createElement("span"); w.className = "w"; w.dataset.ply = i
    const wt = document.createElement("span"); wt.className = "t"; wt.textContent = hist[i]
    w.appendChild(wt)
    li.appendChild(w)
    if (hist[i + 1]) {
      const b = document.createElement("span"); b.className = "b"; b.dataset.ply = i + 1
      const bt = document.createElement("span"); bt.className = "t"; bt.textContent = hist[i + 1]
      b.appendChild(bt)
      li.appendChild(b)
    }
    movelistEl.appendChild(li)
  }
  // the freshest move wears the highlight (the chat no longer names moves)
  const last = movelistEl.querySelector('[data-ply="' + (hist.length - 1) + '"]')
  if (last) last.classList.add("cur")
  fitMoveList()
  revealCurrentMove()
  // any real move snaps history browsing back to the live position
  viewPly = -1
  endHiddenForBrowse = false
  try { if (boardRef) boardRef.removeMarkers(MARKER_TYPE.square) } catch (e) {}
}

function gameOverLine() {
  if (chess.isCheckmate()) {
    return chess.turn() === botColor
      ? "Checkmate — you got me. Rematch?"
      : "Checkmate — that one's mine."
  }
  if (chess.isStalemate()) return "Stalemate — nobody wins that one."
  if (chess.isThreefoldRepetition()) return "Draw by repetition."
  if (chess.isInsufficientMaterial()) return "Draw — not enough pieces left to win."
  if (chess.isDraw()) return "Draw."
  return "Game over."
}

function checkNote() {
  return chess.inCheck() ? " Check." : ""
}

const sleep = (ms) => new Promise(res => setTimeout(res, ms))

// ---------- sounds (synthesized with WebAudio - no files, no licenses) ----------

const sfx = (() => {
  // Straight WebAudio, synthesised on the fly - no audio files in the repo and
  // nothing to license. An earlier version rendered these to WAV and played
  // them through <audio> elements to get around the iPhone's ring/silent
  // switch; that turned out to be unreliable on a real phone, so it is gone.
  // The trade-off is the honest one: with the silent switch on, iOS mutes
  // WebAudio and the site is silent.
  let ctx = null
  const ac = () => ctx || (ctx = new (window.AudioContext || window.webkitAudioContext)())

  let muted = false
  try { muted = localStorage.getItem("bb-sound") === "off" } catch (e) {}

  // a piece landing, modeled on measurements of chess.com's real samples:
  // their move sound is PITCHED, not noise - a couple of damped tones near
  // 460+900Hz, ~1ms attack, dead within ~20ms, nothing below 300Hz or
  // above 2400Hz. So: a few exponentially damped sines, instantly muted.
  function tock(freqs, weights, gain, when = 0, tau = 0.004, dur = 0.05) {
    if (muted) return
    try {
      const c = ac(), t = c.currentTime + when
      const len = Math.floor(c.sampleRate * dur)
      const attack = Math.max(1, Math.floor(c.sampleRate * 0.001))
      const buf = c.createBuffer(1, len, c.sampleRate)
      const d = buf.getChannelData(0)
      const wsum = weights.reduce((a, b) => a + b, 0)
      for (let i = 0; i < len; i++) {
        const ts = i / c.sampleRate
        const env = Math.min(1, i / attack) * Math.exp(-ts / tau)
        let v = 0
        for (let j = 0; j < freqs.length; j++) v += weights[j] * Math.sin(2 * Math.PI * freqs[j] * ts)
        d[i] = (v / wsum) * env
      }
      const src = c.createBufferSource(); src.buffer = buf
      const g = c.createGain(); g.gain.value = gain
      src.connect(g); g.connect(c.destination)
      src.start(t)
    } catch (e) {}
  }
  function blip(freq, dur, gain, when = 0, slide = 0) {
    if (muted) return
    try {
      const c = ac(), t = c.currentTime + when
      const o = c.createOscillator(), g = c.createGain()
      o.type = "sine"
      o.frequency.setValueAtTime(freq, t)
      if (slide) o.frequency.exponentialRampToValueAtTime(slide, t + dur)
      g.gain.setValueAtTime(gain, t)
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur)
      o.connect(g); g.connect(c.destination)
      o.start(t); o.stop(t + dur + 0.02)
    } catch (e) {}
  }

  return {
    // iOS starts the context suspended, only honours resume() inside a real
    // gesture, and wants a buffer actually played in that gesture; it also
    // suspends again whenever the page goes to the background
    unlock() {
      try {
        const c = ac()
        if (c.state === "suspended") c.resume()
        const src = c.createBufferSource()
        src.buffer = c.createBuffer(1, 1, c.sampleRate)
        src.connect(c.destination)
        src.start(0)
      } catch (e) {}
    },
    isMuted() { return muted },
    setMuted(on) {
      muted = !!on
      try { localStorage.setItem("bb-sound", muted ? "off" : "on") } catch (e) {}
    },
    // frequencies/weights tuned so each synth's spectrum lands on the
    // measured band profile of the matching chess.com sample
    // his picks from the tasting panel: A move, B capture, A castle, D check
    move() { tock([460, 900, 1280], [0.75, 1, 0.4], 0.5) },
    capture() { tock([473, 938, 1103], [0.95, 1, 0.7], 0.63, 0, 0.0065, 0.06) },
    castle() { tock([200, 420, 750], [1, 0.75, 0.6], 0.5, 0, 0.005, 0.06); tock([230, 460, 780], [1, 0.75, 0.6], 0.5, 0.09, 0.005, 0.06) },
    check() { tock([882, 1260], [1, 0.6], 0.63, 0, 0.0042, 0.07) },
    // same damped-tock family as the moves, but a small two-step rise so
    // turning sound back on is obviously not a piece landing
    soundOn() { tock([540, 1030], [1, 0.5], 0.42, 0, 0.005, 0.05)
                tock([760, 1440], [1, 0.5], 0.42, 0.07, 0.005, 0.05) },
    promote() { blip(440, 0.09, 0.1); blip(660, 0.13, 0.1, 0.09) },
    start() { blip(392, 0.1, 0.1); blip(523, 0.15, 0.1, 0.1) },
    end() { blip(523, 0.1, 0.1); blip(392, 0.18, 0.1, 0.1) },
  }
})()

// one sound per move, chess.com priorities: check > promote > capture > castle > plain
function playMoveSound(m) {
  if (!m) return
  if (m.san.includes("+") || m.san.includes("#")) sfx.check()
  else if (m.promotion) sfx.promote()
  else if (m.captured) sfx.capture()
  else if (m.flags.includes("k") || m.flags.includes("q")) sfx.castle()
  else sfx.move()
}

// ---------- opening detection (lichess openings data, my own record) ----------

let openingsMap = null      // book-style position key -> opening name
let openingStats = {}       // family -> {n, w, d, l} from my real games
let favoriteFamily = null   // the family I've played most often
let announcedFamilies = new Set()
let pendingOpeningRemark = null

// called after every move: the deepest named position seen so far names the
// opening, and the first time a family appears the bot gets one remark to make
function noteOpening() {
  if (!openingsMap) return
  const name = openingsMap[bookKey(chess.fen())]
  if (!name) return
  const family = name.split(":")[0].trim()
  if (announcedFamilies.has(family)) return
  announcedFamilies.add(family)
  pendingOpeningRemark = openingRemark(name, family)
}

function openingRemark(name, family) {
  const s = openingStats[family]
  if (!s || !s.n) return "The " + name + "? I've never played this one."
  // plain win rate - it explains itself, unlike a chess "score"
  const pct = Math.round(100 * s.w / s.n)
  // "favorite" is about how often I reach for an opening, never how it goes -
  // you can love an opening and still be bad at it
  if (family === favoriteFamily && s.n >= 8) return "The " + name + "! My favorite — I've won " + pct + "% of my games with it."
  if (s.n >= 20) return "The " + name + "! One of my favorites — I've won " + pct + "% of my games with it."
  if (s.n >= 8 && pct <= 40) return "The " + name + "... I've only won " + pct + "% of my games with it. It's time to bump those numbers up!"
  if (s.n >= 4) return "The " + name + " — I've won " + pct + "% of my games with this one."
  return "The " + name + " — I've dabbled in it."
}

function takeOpeningRemark() {
  const r = pendingOpeningRemark
  pendingOpeningRemark = null
  return r
}

function resetOpening() {
  announcedFamilies = new Set()
  pendingOpeningRemark = null
}

// ---------- history browsing (arrow keys, like chess.com) ----------

let viewPly = -1  // -1 = the live position; otherwise "position after ply N"
let endHiddenForBrowse = false

function fenAtPly(k) {
  const hist = chess.history({ verbose: true })
  if (!hist.length) return chess.fen()
  return k <= 0 ? hist[0].before : hist[Math.min(k, hist.length) - 1].after
}

function browseTo(k) {
  const n = chess.history().length
  if (!n || !boardRef) return
  k = Math.max(0, Math.min(n, k))
  const prev = viewPly === -1 ? n : viewPly
  viewPly = k === n ? -1 : k
  const live = viewPly === -1
  // a single step replays that move's sound; a jump gets a plain tap
  if (Math.abs(k - prev) === 1) {
    playMoveSound(chess.history({ verbose: true })[Math.max(k, prev) - 1])
  } else if (k !== prev) {
    sfx.move()
  }
  const fen = live ? chess.fen() : fenAtPly(k)
  boardRef.setPosition(fen, true)
  // pieces only move at the live position, on the user's turn
  try { boardRef.disableMoveInput() } catch (e) {}
  if (live && gameActive && chess.turn() !== botColor) {
    boardRef.enableMoveInput(inputHandler, botColor === "w" ? COLOR.black : COLOR.white)
  }
  // the end screen steps aside while reviewing and returns at the end
  const end = document.getElementById("end")
  if (!live && !end.hidden) { end.hidden = true; endHiddenForBrowse = true }
  if (live && endHiddenForBrowse) { end.hidden = false; endHiddenForBrowse = false }
  // mark the viewed move in the list (at live, that's the freshest move)...
  movelistEl.querySelectorAll(".cur").forEach(s => s.classList.remove("cur"))
  const mark = live ? n - 1 : k - 1
  if (mark >= 0) {
    const span = movelistEl.querySelector('[data-ply="' + mark + '"]')
    if (span) { span.classList.add("cur"); revealCurrentMove() }
  } else {
    movelistEl.scrollTop = 0  // at the starting position, show the first move
  }
  // ...and highlight its from/to squares on the board
  try {
    boardRef.removeMarkers(MARKER_TYPE.square)
    if (mark >= 0) {
      const m = chess.history({ verbose: true })[mark]
      boardRef.addMarker(MARKER_TYPE.square, m.from)
      boardRef.addMarker(MARKER_TYPE.square, m.to)
    }
  } catch (e) {}
  // the chat matches the viewed position: replay what was said back then
  if (live) {
    if (lastLiveStatus) setStatusRaw(lastLiveStatus.kind, lastLiveStatus.label, lastLiveStatus.line)
  } else {
    for (let p = k; p >= 0; p--) {
      if (statusByPly[p]) { setStatusRaw(statusByPly[p].kind, statusByPly[p].label, statusByPly[p].line); break }
    }
  }
  updateEval(fen)
}

function browseKey(e) {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return
  const n = chess.history().length
  if (!n) return
  sfx.unlock()
  const cur = viewPly === -1 ? n : viewPly
  if (e.key === "ArrowLeft") browseTo(cur - 1)
  else if (e.key === "ArrowRight") browseTo(cur + 1)
  else if (e.key === "ArrowUp") browseTo(0)
  else if (e.key === "ArrowDown") browseTo(n)
  else return
  e.preventDefault()
}

// ---------- the eval bar (White's winning chances, chess.com style) ----------

let evalToken = 0

function renderEvalBar(cpWhite) {
  const bar = document.getElementById("eval-bar")
  const fill = document.getElementById("eval-fill")
  const num = document.getElementById("eval-num")
  if (!bar || !fill || !num) return
  const mate = Math.abs(cpWhite) >= 9000
  let p = mate ? (cpWhite > 0 ? 1 : 0) : 1 / (1 + Math.pow(10, -cpWhite / 400))
  if (!mate) p = Math.min(0.95, Math.max(0.05, p))
  fill.style.height = Math.round(p * 100) + "%"
  num.textContent = mate ? "#" : (cpWhite >= 0 ? "+" : "−") + Math.abs(cpWhite / 100).toFixed(1)
  let flipped = false
  try { flipped = boardRef.getOrientation() === COLOR.black } catch (e) {}
  bar.classList.toggle("flip", flipped)
}

function showEvalBar(on) {
  document.getElementById("eval-bar").hidden = !on
  document.getElementById("eval-num").hidden = !on
}

async function updateEval(fen) {
  if (!document.getElementById("eval-bar")) return
  const my = ++evalToken
  const cp = await evalEngine.score(fen)
  if (my !== evalToken) return  // a newer position took over
  if (cp === null || cp === undefined) return
  const cpWhite = fen.split(" ")[1] === "w" ? cp : -cp
  renderEvalBar(cpWhite)
}

// ---------- the style model (behavioral cloning over engine candidates) ----------

// feature order is a contract with pipeline/train_style.py - change both or neither:
// 0-14 static (always active), 15-29 variable (the daily retrain picks a subset)
const CHEB = (f1, r1, f2, r2) => Math.max(Math.abs(f1 - f2), Math.abs(r1 - r2))
const PIECE_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }

// facts about the position that are the same for all candidates
function decisionContext() {
  const hist = chess.history({ verbose: true })
  const oppLast = hist[hist.length - 1]
  const myLast = hist[hist.length - 2]
  const userColor = botColor === "w" ? "b" : "w"
  const mineLast = myLast && myLast.color === botColor ? myLast : null
  const castling = chess.fen().split(" ")[2]
  // squares of my minor+ pieces the opponent's LAST move just attacked -
  // humans miss stale threats, not the one that was played a second ago
  const freshThreats = []
  if (oppLast) {
    for (const row of chess.board()) {
      for (const sq of row) {
        if (sq && sq.color === botColor && "nbrq".includes(sq.type) &&
            chess.attackers(sq.square, userColor).includes(oppLast.to)) {
          freshThreats.push(sq.square)
        }
      }
    }
  }
  return {
    prevMyTo: mineLast ? mineLast.to : null,
    prevMyFrom: mineLast ? mineLast.from : null,
    prevOppCapTo: oppLast && oppLast.captured ? oppLast.to : null,
    canCastle: botColor === "w" ? /[KQ]/.test(castling) : /[kq]/.test(castling),
    inCheck: chess.inCheck(),
    freshThreats,
    tension: Math.min(1, chess.moves({ verbose: true }).filter(m => m.captured).length / 10),
    userColor,
  }
}

// attacked and (undefended, or the cheapest attacker is worth less than the
// piece) - a defended queen attacked by a knight is still lost. Judged with
// the candidate move applied to the board.
function enPrise(sq, owner, opp) {
  const atk = chess.attackers(sq, opp)
  if (!atk.length) return false
  if (!chess.isAttacked(sq, owner)) return true
  const victim = PIECE_VAL[chess.get(sq).type]
  let cheapest = 99
  for (const a of atk) cheapest = Math.min(cheapest, PIECE_VAL[chess.get(a).type])
  return cheapest < victim
}

// judged with the pawn push already applied: does this quiet edge-pawn move
// have a job? It counts as purposeful if it kicks an enemy piece, covers a
// square an enemy minor eyes, defends an attacked friend, is a passer on the
// march, gives a castled king luft, or storms the enemy king. A push that
// does NONE of those while the queens are still on is something I play on
// only 0.3% of my moves - the aggregate edge-pawn feature rations the rate,
// but it can't see purpose.
function aimlessEdgePawn(played, pf) {
  const me = played.color, opp = me === "w" ? "b" : "w"
  const tf = played.to.charCodeAt(0) - 97, tr = played.to.charCodeAt(1) - 49
  let myK = null, oppK = null
  const queens = { w: false, b: false }
  for (const row of chess.board()) {
    for (const sq of row) {
      if (!sq) continue
      if (sq.type === "q") queens[sq.color] = true
      if (sq.type === "k") { if (sq.color === me) myK = sq.square; else oppK = sq.square }
    }
  }
  if (!queens.w || !queens.b) return false // endgames: edge pushes are normal
  const step = me === "w" ? 1 : -1
  let passed = true
  for (const f of [pf - 1, pf, pf + 1]) {
    if (f < 0 || f > 7) continue
    for (let r = tr + step; r >= 0 && r <= 7; r += step) {
      const p = chess.get(String.fromCharCode(97 + f) + (r + 1))
      if (p && p.type === "p" && p.color === opp) { passed = false; break }
    }
  }
  if (passed) return false
  const ar = tr + step
  for (const f of [tf - 1, tf + 1]) {
    if (f < 0 || f > 7 || ar < 0 || ar > 7) continue
    const sq = String.fromCharCode(97 + f) + (ar + 1)
    const p = chess.get(sq)
    if (p && p.color === opp) return false // kicks an enemy piece
    if (p && p.color === me && chess.attackers(sq, opp).length) return false // defends it
  }
  // NO prophylaxis exemption: covering a square an enemy minor merely EYES is
  // available in 29% of my positions and I take it 1.8% of the time - barely
  // above the 0.7% I spend on admittedly aimless pushes. I kick a bishop that
  // is ALREADY there 17% of the time; preventing arrivals is not my game.
  // NO luft exemption either: a single step in front of my own castled king is
  // available in 10% of positions and I play it 0.4% of the time - BELOW the
  // aimless baseline, because it airs out the king it pretends to help.
  const kf = s => s.charCodeAt(0) - 97
  if (oppK && myK && Math.abs(kf(oppK) - pf) <= 2 &&
      Math.abs(kf(myK) - pf) >= 3) return false // pawn storm at their king
  return true
}

function moveFeatures(uci, cpGapPawns, rank, ctx) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4)
  // attack facts are judged in the pre-move position, so read them first
  const fromAtt = chess.isAttacked(from, ctx.userColor) ? 1 : 0
  const toAtt = chess.isAttacked(to, ctx.userColor) ? 1 : 0
  const played = tryMove({ from, to, promotion: uci.slice(4) || undefined })
  if (!played) return null
  // post-move facts, judged with the move on the board (value-aware en prise)
  const hangs = enPrise(to, played.color, ctx.userColor) ? 1 : 0
  let leaves = 0
  for (const row of chess.board()) {
    for (const sq of row) {
      if (!sq || sq.color !== played.color || sq.square === to) continue
      if (sq.type === "p" || sq.type === "k") continue
      if (enPrise(sq.square, played.color, ctx.userColor)) {
        leaves = 1
        break
      }
    }
    if (leaves) break
  }
  // sampling guard: does this move ignore the threat the opponent JUST made?
  let ignoresFresh = false
  for (const sq of ctx.freshThreats) {
    if (sq === from) continue  // the threatened piece itself moved away
    const p = chess.get(sq)
    if (p && p.color === played.color && enPrise(sq, played.color, ctx.userColor)) {
      ignoresFresh = true
      break
    }
  }
  // sampling guard #2: quiet edge-pawn pushes with no detectable purpose
  let aimlessEdge = false
  if (played.piece === "p" && !played.captured && !played.promotion) {
    const pf = played.from.charCodeAt(0) - 97
    if (pf === 0 || pf === 7) aimlessEdge = aimlessEdgePawn(played, pf)
  }
  chess.undo()
  const x = new Array(30).fill(0)
  x[0] = Math.min(Math.max(cpGapPawns, 0), 5)
  x[1] = rank / 4
  x[2] = played.captured ? 1 : 0
  x[3] = played.san.includes("+") || played.san.includes("#") ? 1 : 0
  x[4] = played.promotion ? 1 : 0
  x[5] = played.flags.includes("k") || played.flags.includes("q") ? 1 : 0
  x[6 + "pnbrqk".indexOf(played.piece)] = 1
  const ff = from.charCodeAt(0) - 97, fr = from.charCodeAt(1) - 49
  const tf = to.charCodeAt(0) - 97, tr = to.charCodeAt(1) - 49
  x[12] = (Math.abs(tf - 3.5) + Math.abs(tr - 3.5)) / 7
  x[13] = (played.color === "w" ? tr > fr : tr < fr) ? 1 : 0
  if (played.captured) x[14] = PIECE_VAL[played.captured] / 9
  x[15] = (played.color === "w" ? tr < fr : tr > fr) ? 1 : 0
  x[16] = ctx.prevMyTo === from ? 1 : 0
  x[17] = played.captured && ctx.prevOppCapTo === to ? 1 : 0
  x[18] = fromAtt
  x[19] = toAtt
  x[20] = played.piece === "k" && !x[5] && ctx.canCastle && !ctx.inCheck ? 1 : 0
  x[21] = played.captured && toAtt ? 1 : 0
  x[22] = played.piece === "p" && !played.captured && (ff === 0 || ff === 7) ? 1 : 0
  x[23] = (played.color === "w" ? fr === 0 : fr === 7) ? 1 : 0
  x[24] = x[2] * x[0]
  x[25] = hangs
  x[26] = leaves
  x[27] = played.captured && toAtt && PIECE_VAL[played.captured] < PIECE_VAL[played.piece] ? 1 : 0
  x[28] = PIECE_VAL[played.piece] / 9
  x[29] = x[0] * ctx.tension
  x.ignoresFresh = ignoresFresh  // guard flags, not model features
  x.aimlessEdge = aimlessEdge
  return x
}

// ---------- the v9 feature contract (57 numbers per candidate) ----------
// This is one half of a CONTRACT; the other half is pipeline/features_v9.py.
// The two must produce identical vectors or the bot plays a different game
// than the one it was trained on. pipeline/parity_v9.mjs checks that.
//
// The organising idea: the model only sees what Andrew can see. Every
// score-derived feature is built from the DEPTH-2 evaluation, never depth 5.
// Measured on his games, his choices track how bad a move LOOKS at his horizon
// (weight -0.78) and are statistically blind to the badness only a deeper
// search reveals - which is exactly why the bot can now make the quiet mistake
// that only costs material several moves later.

const V9_N = 58
const V9_PIECE_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }
// a king ATTACKER is worth 100, not 0. v8 used 0 and therefore called every
// defended piece beside the enemy king "hanging".
const V9_ATTACKER_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 100 }
const V9_ORDER = ["p", "n", "b", "r", "q", "k"]
const V9_HOME = {
  w: { n: ["b1", "g1"], b: ["c1", "f1"], r: ["a1", "h1"] },
  b: { n: ["b8", "g8"], b: ["c8", "f8"], r: ["a8", "h8"] },
}
const V9_FIANCHETTO = { w: ["b2", "g2"], b: ["b7", "g7"] }

const sqFile = (s) => s.charCodeAt(0) - 97
const sqRank = (s) => s.charCodeAt(1) - 49
const mkSq = (f, r) => String.fromCharCode(97 + f) + (r + 1)
const cheb = (a, b) => Math.max(Math.abs(sqFile(a) - sqFile(b)), Math.abs(sqRank(a) - sqRank(b)))

// attacked, and not adequately defended (king attacker counts 100)
function enPriseV9(sq, owner, opp) {
  const atk = chess.attackers(sq, opp)
  if (!atk.length) return false
  if (!chess.isAttacked(sq, owner)) return true
  const pc = chess.get(sq)
  if (!pc) return false
  let cheapest = 999
  for (const a of atk) {
    const ap = chess.get(a)
    if (ap) cheapest = Math.min(cheapest, V9_ATTACKER_VAL[ap.type])
  }
  return cheapest < V9_PIECE_VAL[pc.type]
}

function neighbours(sq) {
  const f = sqFile(sq), r = sqRank(sq), out = []
  for (let df = -1; df <= 1; df++) {
    for (let dr = -1; dr <= 1; dr++) {
      if (!df && !dr) continue
      const nf = f + df, nr = r + dr
      if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) out.push(mkSq(nf, nr))
    }
  }
  return out
}

// everything true of the POSITION rather than of one candidate, computed once
function decisionContextV9() {
  const hist = chess.history({ verbose: true })
  const oppLast = hist[hist.length - 1]
  const myLast = hist[hist.length - 2]
  const me = chess.turn()
  const opp = me === "w" ? "b" : "w"
  const mineLast = myLast && myLast.color === me ? myLast : null
  const castling = chess.fen().split(" ")[2]

  const cells = []
  for (const row of chess.board()) for (const c of row) if (c) cells.push(c)

  let pieceCount = 0, npm = 0, mine = 0, theirs = 0
  let myKing = null, enemyKing = null
  const ownPawnRanks = [[], [], [], [], [], [], [], []]
  const enemyPawnRanks = [[], [], [], [], [], [], [], []]
  for (const c of cells) {
    pieceCount++
    const v = V9_PIECE_VAL[c.type]
    if (c.type !== "p" && c.type !== "k") npm += v
    if (c.color === me) mine += v; else theirs += v
    if (c.type === "k") { if (c.color === me) myKing = c.square; else enemyKing = c.square }
    if (c.type === "p") {
      (c.color === me ? ownPawnRanks : enemyPawnRanks)[sqFile(c.square)].push(sqRank(c.square))
    }
  }

  let kingExposure = 0
  const shield = new Set()
  let castled = false
  if (myKing) {
    const seen = new Set()
    for (const sq of neighbours(myKing)) for (const a of chess.attackers(sq, opp)) seen.add(a)
    kingExposure = seen.size
    const kf = sqFile(myKing), kr = sqRank(myKing)
    const homeRank = me === "w" ? 0 : 7, second = me === "w" ? 1 : 6
    castled = (kr === homeRank || kr === second) && [0, 1, 2, 5, 6, 7].includes(kf)
    const step = me === "w" ? 1 : -1
    for (const f of [kf - 1, kf, kf + 1]) {
      if (f < 0 || f > 7) continue
      for (const d of [1, 2]) {
        const r = kr + step * d
        if (r >= 0 && r <= 7) shield.add(mkSq(f, r))
      }
    }
  }

  let zone = [], zoneBefore = 0
  if (enemyKing) {
    zone = neighbours(enemyKing)
    const seen = new Set()
    for (const sq of zone) for (const a of chess.attackers(sq, me)) seen.add(a)
    zoneBefore = seen.size
  }

  // pieces of mine the opponent's LAST move put en prise
  const fresh = []
  if (oppLast) {
    for (const c of cells) {
      if (c.color !== me || !"nbrq".includes(c.type)) continue
      if (chess.attackers(c.square, opp).includes(oppLast.to) && enPriseV9(c.square, me, opp)) {
        fresh.push(c.square)
      }
    }
  }

  const prevOppCapTo = oppLast && oppLast.captured ? oppLast.to : null
  let recapN = 0, recapMin = null
  const legal = chess.moves({ verbose: true })
  if (prevOppCapTo) {
    const vals = legal.filter(m => m.to === prevOppCapTo && m.captured)
                      .map(m => V9_PIECE_VAL[m.piece])
    recapN = vals.length
    if (vals.length) recapMin = Math.min(...vals)
  }

  return {
    me, opp,
    prevMyTo: mineLast ? mineLast.to : null,
    prevOppCapTo,
    lastMoveTo: oppLast ? oppLast.to : null,
    inCheck: chess.inCheck(),
    canCastle: me === "w" ? /[KQ]/.test(castling) : /[kq]/.test(castling),
    pieceCount, npm,
    phase: 1 - pieceCount / 32,
    imbalance: Math.sqrt(Math.min(Math.abs(mine - theirs), 9) / 9),
    ownPawnRanks, enemyPawnRanks,
    myKing, enemyKing, kingExposure, shield, castled,
    zone, zoneBefore, fresh, recapN, recapMin,
  }
}

// impute missing horizon scores, then take the best - in that order
function horizonScores(ucis, shByUci) {
  const present = ucis.map(u => shByUci[u]).filter(v => v !== undefined && v !== null)
  const floor = present.length ? Math.min(...present) : null
  const out = ucis.map(u => {
    const v = shByUci[u]
    return (v === undefined || v === null) ? floor : v
  })
  const usable = out.filter(v => v !== null)
  return { sh: out, bestSh: usable.length ? Math.max(...usable) : null }
}

// a quiet edge-pawn push with no job, judged on the PRE-move board. v8 carried
// this as a hard play-time guard; v9 carries it as a feature so the model fits
// his own measured rate (he kicks a bishop already on the square 17.1% of the
// time and pushes with no purpose at all 0.7%) instead of a hand-set rule.
function aimlessEdgePawnV9(me, opp, from, to) {
  // called with the move ALREADY on the board; `from`/`to` are its squares
  let whiteQ = false, blackQ = false
  const cells = []
  for (const row of chess.board()) for (const c of row) if (c) {
    cells.push(c)
    if (c.type === "q") { if (c.color === "w") whiteQ = true; else blackQ = true }
  }
  if (!whiteQ || !blackQ) return false        // endgames: edge pushes are normal
  const ff = sqFile(from), tr = sqRank(to), fwd = me === "w" ? 1 : -1
  let blocked = false
  for (const f of [Math.max(0, ff - 1), ff, Math.min(7, ff + 1)]) {
    for (const c of cells) {
      if (c.type === "p" && c.color === opp && sqFile(c.square) === f &&
          (sqRank(c.square) - tr) * fwd > 0) { blocked = true; break }
    }
    if (blocked) break
  }
  if (!blocked) return false                  // a passer on the march
  const ar = tr + fwd
  for (const f of [sqFile(to) - 1, sqFile(to) + 1]) {
    if (f < 0 || f > 7 || ar < 0 || ar > 7) continue
    const sq = mkSq(f, ar)
    const p = chess.get(sq)
    if (p && p.color === opp) return false    // kicks an enemy piece
    if (p && p.color === me && chess.attackers(sq, opp).length) return false
  }
  let myK = null, oppK = null
  for (const c of cells) if (c.type === "k") { if (c.color === me) myK = c.square; else oppK = c.square }
  if (oppK && myK && Math.abs(sqFile(oppK) - ff) <= 2 && Math.abs(sqFile(myK) - ff) >= 3) {
    return false                              // pawn storm at their king
  }
  return true
}

function moveFeaturesV9(uci, ctx, sh, bestSh, rank) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4)
  const me = ctx.me, opp = ctx.opp
  const x = new Array(V9_N).fill(0)

  const pre = chess.get(from)
  if (!pre) return null
  const mover = pre.type
  const moverVal = V9_PIECE_VAL[mover]
  const toAttacked = chess.isAttacked(to, opp)
  const fromAttacked = chess.isAttacked(from, opp)
  const danger = enPriseV9(from, me, opp)
  const enemyPawnsAttackTo = chess.attackers(to, opp).some(a => {
    const p = chess.get(a); return p && p.type === "p"
  })

  const played = tryMove({ from, to, promotion: uci.slice(4) || undefined })
  if (!played) return null
  const isCapture = !!played.captured
  const isCastle = played.flags.includes("k") || played.flags.includes("q")
  const givesCheck = chess.inCheck()
  const tf = sqFile(to), tr = sqRank(to), ff = sqFile(from), fr = sqRank(from)
  const fwd = me === "w" ? 1 : -1

  // post-move facts, gathered while the move is on the board
  const landedHanging = enPriseV9(to, me, opp)
  let count = landedHanging ? 1 : 0
  let others = false
  let ignoresFresh = false
  let bestAny = 0, bestLoose = 0
  const postCells = []
  for (const row of chess.board()) for (const c of row) if (c) postCells.push(c)
  for (const c of postCells) {
    if (c.color === me && "nbrq".includes(c.type) && c.square !== to) {
      if (enPriseV9(c.square, me, opp)) { others = true; count++ }
    }
    if (c.color === opp && c.type !== "k" && chess.attackers(c.square, me).includes(to)) {
      const v = V9_PIECE_VAL[c.type]
      if (v > bestAny) bestAny = v
      if ((!chess.isAttacked(c.square, opp) || v > moverVal) && v > bestLoose) bestLoose = v
    }
  }
  for (const sq of ctx.fresh) {
    if (sq === from) continue
    const p = chess.get(sq)
    if (p && p.color === me && enPriseV9(sq, me, opp)) { ignoresFresh = true; break }
  }
  let unsafeCheck = 0
  if (givesCheck && ctx.enemyKing) {
    if (chess.attackers(to, opp).includes(ctx.enemyKing) && !chess.isAttacked(to, me)) unsafeCheck = 1
  }
  let zonePressure = 0
  if (ctx.zone.length) {
    const seen = new Set()
    for (const sq of ctx.zone) for (const a of chess.attackers(sq, me)) seen.add(a)
    if (seen.size > ctx.zoneBefore) zonePressure = 1
  }
  let aimlessEdge = 0
  if (mover === "p" && !isCapture && !played.promotion && (ff === 0 || ff === 7)) {
    aimlessEdge = aimlessEdgePawnV9(me, opp, from, to) ? 1 : 0
  }
  let castlePressure = 0, castleNoShield = 0
  if (isCastle) {
    let kd = null
    for (const c of postCells) if (c.type === "k" && c.color === me) kd = c.square
    if (kd) {
      const seen = new Set()
      for (const sq of neighbours(kd)) for (const a of chess.attackers(sq, opp)) seen.add(a)
      castlePressure = Math.min(seen.size, 3) / 3
      const kf2 = sqFile(kd), kr2 = sqRank(kd)
      let shieldPawns = 0
      for (const f of [kf2 - 1, kf2, kf2 + 1]) {
        if (f < 0 || f > 7) continue
        for (let r = 0; r <= 7; r++) {
          if ((r - kr2) * fwd <= 0) continue
          const p = chess.get(mkSq(f, r))
          if (p && p.color === me && p.type === "p") { shieldPawns++; break }
        }
      }
      castleNoShield = (3 - shieldPawns) / 3
    }
  }
  chess.undo()

  // ---- 0-2: perception ----
  if (bestSh !== null && sh !== null) {
    const raw = Math.max(0, (bestSh - sh) / 100)
    x[0] = Math.log1p(Math.min(raw, 5))
    x[2] = 1 / (1 + Math.exp(-Math.max(-2000, Math.min(2000, sh)) / 150))
  }
  const quiet = (isCapture || givesCheck) ? 0 : 1
  x[1] = x[0] * quiet

  // ---- 3-13 ----
  x[3] = rank / 4
  const forcing = (isCapture || givesCheck || played.promotion) ? 1 : 0
  x[4] = forcing
  x[5] = isCapture ? 1 : 0
  x[6] = givesCheck ? 1 : 0
  x[7] = played.promotion ? 1 : 0
  x[8] = isCastle ? 1 : 0
  if (isCapture) x[9] = V9_PIECE_VAL[played.captured] / 9
  x[10] = (isCapture && toAttacked) ? 1 : 0
  x[11] = (isCapture && toAttacked && V9_PIECE_VAL[played.captured] < moverVal) ? 1 : 0
  x[12] = (isCapture && ctx.prevOppCapTo === to) ? 1 : 0
  x[13] = (x[12] && ctx.recapN > 1 && ctx.recapMin !== null && moverVal === ctx.recapMin) ? 1 : 0

  // ---- 14-22 ----
  x[14 + V9_ORDER.indexOf(mover)] = 1
  x[20] = (Math.abs(tf - 3.5) + Math.abs(tr - 3.5)) / 7
  const forward = (tr - fr) * fwd
  x[21] = forward > 0 ? 1 : 0
  x[22] = forward < 0 ? 1 : 0

  // ---- 23-32 ----
  x[23] = (mover === "n" || mover === "b") ? x[22] : 0
  x[24] = (mover === "n" && (tf === 0 || tf === 7 || tr === 0 || tr === 7)) ? 1 : 0
  x[25] = (mover === "p" && Math.abs(tr - fr) === 2) ? 1 : 0
  if (mover === "p" && !isCapture) {
    const back = tr - fwd
    if (back >= 0 && back <= 7) {
      for (const f of [tf - 1, tf + 1]) {
        if (f < 0 || f > 7) continue
        const p = chess.get(mkSq(f, back))
        if (p && p.color === me && p.type === "p") { x[26] = 1; break }
      }
    }
  }
  x[27] = (mover === "p" && !isCapture && (ff === 0 || ff === 7)) ? 1 : 0
  const ramp = Math.max(0, (12 - ctx.pieceCount) / 10)
  x[28] = ramp * ((mover === "p" && !isCapture) ? 1 : 0)
  x[29] = ((V9_HOME[me][mover] || []).includes(from)) ? 1 : 0
  x[30] = (mover === "b" && V9_FIANCHETTO[me].includes(to)) ? 1 : 0
  if (mover === "r") {
    for (const r of ctx.ownPawnRanks[tf]) { if ((r - tr) * fwd > 0) { x[31] = 1; break } }
  }
  if (!enemyPawnsAttackTo) {
    let reachable = false
    for (const f of [tf - 1, tf + 1]) {
      if (f < 0 || f > 7) continue
      for (const r of ctx.enemyPawnRanks[f]) { if ((r - tr) * fwd > 0) { reachable = true; break } }
      if (reachable) break
    }
    x[32] = reachable ? 0 : 1
  }

  // ---- 33-42 ----
  x[33] = fromAttacked ? 1 : 0
  x[34] = toAttacked ? 1 : 0
  x[35] = landedHanging ? 1 : 0
  x[36] = others ? 1 : 0
  x[37] = count
  x[38] = ("nbrq".includes(mover) && !isCapture && enemyPawnsAttackTo) ? 1 : 0
  x[39] = (danger && !landedHanging) ? moverVal / 9 : 0
  x[40] = ignoresFresh ? 1 : 0
  x[41] = (bestAny >= 3 && !landedHanging) ? 1 : 0
  x[42] = bestLoose / 9

  // ---- 43-44 ----
  if (ctx.lastMoveTo) x[43] = (6 - Math.min(cheb(to, ctx.lastMoveTo), 6)) / 6
  if (ctx.prevMyTo) x[44] = (4 - Math.min(cheb(to, ctx.prevMyTo), 4)) / 4

  // ---- 45-50 ----
  const kingMove = mover === "k" && !isCastle
  if (kingMove && !ctx.inCheck) {
    if (ctx.canCastle) x[45] = 1
    else { x[46] = ctx.npm / 62; x[47] = Math.min(ctx.kingExposure, 3) / 3 }
  }
  x[48] = (mover === "p" && !isCapture && ctx.shield.has(from) && ctx.castled) ? 1 : 0
  x[49] = unsafeCheck
  x[50] = zonePressure

  // ---- 51-56 ----
  x[51] = ctx.phase * forcing
  x[52] = ctx.imbalance * forcing
  x[53] = ctx.phase * (kingMove ? 1 : 0)
  x[54] = ctx.imbalance * x[0]
  x[55] = castlePressure
  x[56] = castleNoShield
  x[57] = aimlessEdge
  return x
}

// sample from the v9 policy. No salience guards: what they used to patch over
// is now carried by real features (ignores_fresh_threat, king_walk,
// quiet_edge_pawn, hangs_*), fitted at his own measured rates.
function stylePickV9(lines, shByUci) {
  if (!styleModel) return null
  const cands = Object.keys(lines).sort((a, b) => a - b).map(k => lines[k])
  if (cands.length < 2) return null
  const ctx = decisionContextV9()
  const { sh, bestSh } = horizonScores(cands.map(c => c.uci), shByUci)
  const w = styleModel.weights
  const scored = []
  for (let i = 0; i < cands.length; i++) {
    const x = moveFeaturesV9(cands[i].uci, ctx, sh[i], bestSh, i)
    if (!x) continue
    let z = 0
    for (let j = 0; j < w.length; j++) z += w[j] * x[j]
    scored.push({ uci: cands[i].uci, z })
  }
  if (scored.length < 2) return null
  const zmax = Math.max(...scored.map(c => c.z))
  let total = 0
  for (const c of scored) { c.p = Math.exp((c.z - zmax) / PLAY_TEMP); total += c.p }
  let r = Math.random() * total
  for (const c of scored) { r -= c.p; if (r <= 0) return c.uci }
  return scored[0].uci
}

// sample a move from the model's probabilities over the engine's candidates
function stylePick(lines) {
  if (!styleModel) return null
  const cands = Object.keys(lines).sort((a, b) => a - b).map(k => lines[k])
  if (cands.length < 2) return null
  const best = cands[0].cp
  const ctx = decisionContext()
  const w = styleModel.weights
  const act = styleModel.active
  const scored = []
  for (let i = 0; i < cands.length; i++) {
    const x = moveFeatures(cands[i].uci, (best - cands[i].cp) / 100, i, ctx)
    if (!x) continue
    let z = 0
    for (let j = 0; j < w.length; j++) z += w[j] * x[act[j]]
    // salience guard: the aggregate model can't know a threat is FRESH,
    // voluntary king-walks are a 1-in-175 event for me, purposeless
    // edge-pawn pushes a 1-in-300 one, and in a healthy position I play a
    // 2.5-pawn howler on only 2% of moves (when already lost I flail like
    // anyone, so the tail stays there) - candidates that do any of these
    // only survive as the engine's #1 (deep tactics earn respect)
    scored.push({ uci: cands[i].uci, z,
                  guarded: x.ignoresFresh || x.aimlessEdge || x[20] === 1 ||
                           (best > -200 && best - cands[i].cp >= 250),
                  // a queen hanging to a lesser piece is the one thing I see
                  // every time - a near-best grab of one (within half a pawn
                  // of the engine's #1) makes every non-grab candidate guarded
                  grab: x[2] === 1 && x[14] === 1 && x[28] < 1 &&
                        best - cands[i].cp <= 50 })
  }
  if (scored.length < 2) return null
  // rank-0 always survives, so the pool is never empty; a singleton pool
  // (forced spot: everything else guarded) just plays the engine's #1
  const hasGrab = scored.some(c => c.grab)
  const pool = scored.filter((c, i) => i === 0 || (hasGrab ? c.grab : !c.guarded))
  const zmax = Math.max(...pool.map(c => c.z))
  let total = 0
  for (const c of pool) { c.p = Math.exp((c.z - zmax) / PLAY_TEMP); total += c.p }
  let r = Math.random() * total
  for (const c of pool) {
    r -= c.p
    if (r <= 0) return c.uci
  }
  return pool[0].uci
}

// ---------- the "me" button: password sign-in for game capture ----------

// "ok" | "bad" | "unknown" - unknown means the network failed, not the key
async function keyStatus(key) {
  try {
    const r = await fetch(CAPTURE_URL.replace("/game", "/games") + "?key=" + encodeURIComponent(key))
    return r.ok ? "ok" : "bad"
  } catch (e) {
    return "unknown"
  }
}

function renderMe() {
  const btn = document.getElementById("bb-login")
  if (!btn) return
  let has = false
  try { has = !!localStorage.getItem("bb-key") } catch (e) {}
  btn.classList.toggle("on", has)
  btn.textContent = has ? "logout" : "I'm Burke"
  btn.title = has
    ? "signed in - finished games train the bot"
    : "Andrew's sign-in for game capture"
  renderPlayers()   // the challenger is named "Burke" once he signs in
}

function meClick() {
  const btn = document.getElementById("bb-login")
  let existing = null
  try { existing = localStorage.getItem("bb-key") } catch (e) {}
  if (existing) {
    try { localStorage.removeItem("bb-key") } catch (e) {}
    renderMe()
    return
  }
  const input = document.createElement("input")
  input.type = "password"
  input.className = "bb-me-input"
  input.placeholder = "password"
  input.autocomplete = "off"
  btn.replaceWith(input)
  input.focus()
  let busy = false
  const done = () => { input.replaceWith(btn); renderMe() }
  input.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") { done(); return }
    if (e.key !== "Enter" || busy) return
    const pw = input.value.trim()
    if (!pw) { done(); return }
    busy = true
    input.disabled = true
    const ok = (await keyStatus(pw)) === "ok"
    if (ok) {
      try { localStorage.setItem("bb-key", pw) } catch (err) {}
      done()
    } else {
      busy = false
      input.disabled = false
      input.value = ""
      input.placeholder = "you are not Burke"
      input.classList.add("bad")   // red ring to go with the placeholder
      input.focus()
    }
  })
  // typing again clears the rejection, so the red does not outlive the mistake
  input.addEventListener("input", () => input.classList.remove("bad"))
  input.addEventListener("blur", () => { if (!busy) done() })
}

// ---------- game capture (keyed - only games I flag as mine are stored) ----------

function captureGame(result) {
  let key = null
  try { key = localStorage.getItem("bb-key") } catch (e) {}
  const hist = chess.history({ verbose: true })
  if (!key || !stdStart || hist.length < 6) return
  const body = {
    key,
    id: (crypto.randomUUID && crypto.randomUUID()) || Date.now() + "-" + Math.random().toString(16).slice(2),
    color: botColor === "w" ? "b" : "w",
    result,
    moves: hist.map(m => m.from + m.to + (m.promotion || "")),
    ts: Date.now(),
  }
  window.bbCapture = body
  try {
    fetch(CAPTURE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => {})
  } catch (e) {}
}

// ---------- game flow ----------

async function botMove(board, id) {
  if (chess.isGameOver()) { finishAuto(); return }
  noteOpening()  // the user's move may have entered a named opening
  setStatus("thinking", "thinking", "…")
  updateEval(chess.fen())
  const started = Date.now()

  const fromBook = pickBookMove()
  let played, source
  if (fromBook) {
    await sleep(400 + Math.random() * 700)
    if (id !== gameId) return
    played = tryMove({ from: fromBook.uci.slice(0, 2), to: fromBook.uci.slice(2, 4), promotion: fromBook.uci.slice(4) || undefined })
    source = fromBook
  }
  if (!played) {
    await engine.ready
    const fen = chess.fen()
    const result = await engine.bestMove(fen)
    if (id !== gameId) return
    const elapsed = Date.now() - started
    if (elapsed < 500) await sleep(500 - elapsed)
    if (id !== gameId) return
    let uci = result.uci
    if (styleModel && styleModel.features === "v9") {
      // v9 needs a second, shallow look at the same position - his horizon.
      // If it fails the features collapse to constants and the bot would just
      // follow the engine's ordering, so say so rather than failing silently.
      let shBy = null
      try {
        const h = await engine.horizon(fen)
        if (id !== gameId) return
        shBy = {}
        for (const k of Object.keys(h.lines)) shBy[h.lines[k].uci] = h.lines[k].cp
      } catch (e) {
        console.warn("horizon search failed; falling back to the engine's move", e)
      }
      if (shBy) uci = stylePickV9(result.lines, shBy) || result.uci
    } else if (styleModel) {
      uci = stylePick(result.lines) || result.uci
    }
    window.bbLast = { pick: uci, best: result.uci, lines: result.lines }
    played = tryMove({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined })
    if (!played && uci !== result.uci) {
      played = tryMove({ from: result.uci.slice(0, 2), to: result.uci.slice(2, 4), promotion: result.uci.slice(4) || undefined })
    }
    source = null
  }
  if (!played) { finishAuto(); return }
  playMoveSound(played)

  await board.setPosition(chess.fen(), true)
  renderMoves()
  if (chess.isGameOver()) { finishAuto(); return }
  updateEval(chess.fen())
  noteOpening()  // ...and so may the bot's reply

  // the move itself is never named - it wears the highlight in the move list
  if (source) {
    const remark = takeOpeningRemark()
    if (remark) {
      setStatus("book", "from my games", remark + " Your move." + checkNote())
    } else {
      const pct = Math.round(100 * (source.wins + source.draws / 2) / source.n)
      const times = source.n === 1 ? "once before" : source.n + " of " + source.total + " times"
      setStatus("book", "from my games", "I've played this here " + times + " (" + pct + "% score with it). Your move." + checkNote())
    }
  } else if (inBook) {
    // the leaving-book line wins; a pending opening remark keeps for the next move
    inBook = false
    setStatus("engine", "on my own", "We've left my games, so I'm thinking for myself now. Your move." + checkNote())
  } else {
    const remark = takeOpeningRemark()
    setStatus("engine", "on my own", (remark ? remark + " " : "") + "Your move." + checkNote())
  }
  // idempotent: overlapping game-start paths can reach here with input
  // already on, and cm-chessboard throws on a double enable
  try { board.disableMoveInput() } catch (e) {}
  try { board.enableMoveInput(inputHandler, botColor === "w" ? COLOR.black : COLOR.white) } catch (e) {}
}

function autoResult() {
  if (chess.isCheckmate()) return chess.turn() === botColor ? "w" : "l"
  return "d"
}

function finishAuto() {
  finish(autoResult(), gameOverLine())
}

function finish(result, line) {
  gameActive = false
  if (chess.isCheckmate()) renderEvalBar(chess.turn() === "w" ? -10000 : 10000)
  else if (result === "d") renderEvalBar(0)
  sfx.end()
  setStatus("over", "game over", line)
  boardRef.disableMoveInput()
  setControls(false)
  showEnd(result, line)
  captureGame(result)
}

function showEnd(result, line) {
  document.getElementById("confirm").hidden = true
  document.getElementById("end-title").textContent =
    result === "w" ? "You win." : result === "l" ? "You lose." : "Draw."
  document.getElementById("end-line").textContent = line
  document.getElementById("pick").hidden = true
  document.getElementById("end").hidden = false
}

function setControls(on) {
  document.getElementById("resign-btn").disabled = !on
}

// ---------- pick a color / play again / resign ----------

function startGame(userColor) {
  document.getElementById("pick").hidden = true
  document.getElementById("confirm").hidden = true
  document.getElementById("end").hidden = true
  sfx.unlock()
  sfx.start()
  showEvalBar(true)
  gameActive = true
  newGame(userColor)
  setControls(true)
}

function playAgain() {
  gameId++
  gameActive = false
  chess = new Chess()
  inBook = true
  renderMoves()
  boardRef.disableMoveInput()
  setControls(false)
  boardRef.setOrientation(COLOR.white, false)
  boardRef.setPosition(chess.fen(), false)
  renderEvalBar(0)
  showEvalBar(false)
  setStatus("book", "new game", "Pick your color to start.")
  document.getElementById("confirm").hidden = true
  document.getElementById("end").hidden = true
  document.getElementById("pick").hidden = false
}

function resignClick() {
  if (!gameActive) return
  document.getElementById("confirm").hidden = false
}

function inputHandler(event) {
  if (event.type === INPUT_EVENT_TYPE.movingOverSquare) return
  if (event.type !== INPUT_EVENT_TYPE.moveInputFinished) {
    event.chessboard.removeLegalMovesMarkers()
  }
  if (event.type === INPUT_EVENT_TYPE.moveInputStarted) {
    const moves = chess.moves({ square: event.squareFrom, verbose: true })
    event.chessboard.addLegalMovesMarkers(moves)
    return moves.length > 0
  }
  if (event.type === INPUT_EVENT_TYPE.validateMoveInput) {
    const result = tryMove({ from: event.squareFrom, to: event.squareTo, promotion: event.promotion })
    if (result) {
      playMoveSound(result)
      const id = gameId
      event.chessboard.state.moveInputProcess.then(() => {
        event.chessboard.setPosition(chess.fen(), true).then(() => {
          renderMoves()
          botMove(event.chessboard, id)
        })
      })
      return true
    }
    // maybe a promotion
    const candidates = chess.moves({ square: event.squareFrom, verbose: true })
    for (const m of candidates) {
      if (m.promotion && m.to === event.squareTo) {
        const userColor = botColor === "w" ? COLOR.black : COLOR.white
        event.chessboard.showPromotionDialog(event.squareTo, userColor, (res) => {
          if (res.type === PROMOTION_DIALOG_RESULT_TYPE.pieceSelected) {
            playMoveSound(tryMove({ from: event.squareFrom, to: event.squareTo, promotion: res.piece.charAt(1) }))
            event.chessboard.setPosition(chess.fen(), true).then(() => {
              renderMoves()
              botMove(event.chessboard, gameId)
            })
          } else {
            event.chessboard.enableMoveInput(inputHandler, userColor)
            event.chessboard.setPosition(chess.fen(), true)
          }
        })
        return true
      }
    }
    return false
  }
  if (event.type === INPUT_EVENT_TYPE.moveInputFinished) {
    if (event.legalMove) event.chessboard.disableMoveInput()
  }
}

let boardRef

function newGame(userColor) {
  gameId++
  stdStart = true
  chess = new Chess()
  inBook = true
  resetOpening()
  statusByPly = {}
  botColor = userColor === "w" ? "b" : "w"
  renderMoves()
  boardRef.disableMoveInput()
  boardRef.setOrientation(userColor === "w" ? COLOR.white : COLOR.black, false)
  boardRef.setPosition(chess.fen(), false).then(() => {
    if (botColor === "w") {
      botMove(boardRef, gameId)
    } else {
      setStatus("book", "your move", "You have the white pieces — go ahead.")
      boardRef.enableMoveInput(inputHandler, COLOR.white)
      updateEval(chess.fen())
    }
  })
}

// ---------- stats panel ----------

function renderStats(stats) {
  const list = document.getElementById("stats-list")
  const rows = [
    ["rating", stats.bot_scale_strength ? String(stats.bot_scale_strength) : "—"],
    ["games learned from", String(stats.games)],
    ["book positions", stats.book_positions.toLocaleString("en-US")],
    ["favorite opening as White", stats.openings_white[0] ? stats.openings_white[0][0] : "—"],
    ["favorite opening as Black", stats.openings_black[0] ? stats.openings_black[0][0] : "—"],
  ]
  list.innerHTML = ""
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k
    const dd = document.createElement("dd"); dd.textContent = v
    if (k === "rating" || k.startsWith("favorite")) dd.className = "g"
    // The opening labels run nearly the full width of the card. Marked so a
    // phone can drop their value onto its own line instead of squeezing it
    // into what little room is left beside the label.
    if (k.startsWith("favorite")) { dt.classList.add("wide"); dd.classList.add("wide") }
    list.appendChild(dt); list.appendChild(dd)
  }
}

// ---------- boot ----------

async function boot() {
  const [bookRes, statsRes, styleRes, openingsRes] = await Promise.all([
    fetch("book.json"), fetch("stats.json"), fetch("style-v9.json").catch(() => null),
    fetch("openings.json").catch(() => null),
  ])
  book = await bookRes.json()
  const stats = await statsRes.json()
  openingStats = stats.opening_stats || {}
  // "my favorite" comes from the panel's once-per-game counts (a game's most
  // specific family) - the remark stats count pass-through families, where
  // generic waypoints like the King's Pawn Game would win unfairly
  const perGame = {}
  for (const [name, n] of [...(stats.openings_white || []), ...(stats.openings_black || [])]) {
    perGame[name] = (perGame[name] || 0) + n
    if (!favoriteFamily || perGame[name] > perGame[favoriteFamily]) favoriteFamily = name
  }
  try {
    if (openingsRes && openingsRes.ok) openingsMap = await openingsRes.json()
  } catch (e) { openingsMap = null }
  try {
    if (styleRes && styleRes.ok) styleModel = await styleRes.json()
  } catch (e) { styleModel = null }
  const okModel = styleModel &&
    (styleModel.features === "v9" || styleModel.features === "v8") &&
    Array.isArray(styleModel.weights) && Array.isArray(styleModel.active) &&
    styleModel.active.length === styleModel.weights.length &&
    styleModel.active.every(i => Number.isInteger(i) && i >= 0 &&
      i < (styleModel.features === "v9" ? 58 : 30))
  if (okModel) {
    engine.ready.then(() => engine.useMultipv())
  } else {
    styleModel = null
  }
  // copy the move list: people reach for this to paste a game into an
  // analysis board, and highlighting a two-column scrolling list by hand is
  // miserable. Writes standard PGN movetext, not what is on screen.
  const copyBtn = document.getElementById("copy-moves")
  let copyTimer = null
  if (copyBtn) {
    const COPY_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
    const DONE_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4 12 9 17 20 6" pathLength="100"/></svg>'
    copyBtn.innerHTML = COPY_ICON
    copyBtn.addEventListener("click", async () => {
      const hist = chess.history()
      if (!hist.length) return
      let text = ""
      for (let i = 0; i < hist.length; i += 2) {
        text += (i / 2 + 1) + ". " + hist[i] + (hist[i + 1] ? " " + hist[i + 1] : "") + " "
      }
      text = text.trim()
      let ok = true
      try {
        await navigator.clipboard.writeText(text)
      } catch (e) {
        // The async API refuses for reasons that have nothing to do with the
        // page being wrong - no permission, the document not visible or not
        // focused - so fall back rather than give up. iOS Safari ignores a
        // plain .select() on a hidden textarea, hence the contentEditable and
        // explicit Range dance, which is the combination it does honour.
        try {
          const ta = document.createElement("textarea")
          ta.value = text
          ta.setAttribute("readonly", "")
          ta.style.cssText = "position:fixed;top:0;left:0;width:1px;height:1px;padding:0;border:none;opacity:0"
          document.body.appendChild(ta)
          ta.contentEditable = "true"
          ta.readOnly = false
          const range = document.createRange()
          range.selectNodeContents(ta)
          const sel = window.getSelection()
          sel.removeAllRanges()
          sel.addRange(range)
          ta.setSelectionRange(0, text.length)
          ok = document.execCommand("copy")
          sel.removeAllRanges()
          ta.remove()
        } catch (e2) { ok = false }
      }
      // Two things were making this look broken. The revert timer was never
      // cleared, so a second copy within 1400ms left the FIRST timer running
      // and it wiped the tick almost immediately - press it twice and it
      // barely flashes. And a failed copy showed NOTHING: no tick, no error,
      // identical to the button not working, which is what "sometimes it does
      // not animate at all" actually is. Failure now flashes red instead.
      clearTimeout(copyTimer)
      copyBtn.classList.toggle("done", ok)
      copyBtn.classList.toggle("fail", !ok)
      copyBtn.innerHTML = ok ? DONE_ICON : COPY_ICON
      copyBtn.title = ok ? "Copied" : "Could not copy - your browser blocked it"
      copyTimer = setTimeout(() => {
        copyBtn.innerHTML = COPY_ICON
        copyBtn.classList.remove("done", "fail")
        copyBtn.title = "Copy the move list"
      }, 1400)
    })
  }

  // heartbeat: the nightly job stamps this on EVERY successful run, including
  // the no-op ones. A cron that stops firing produces no failure and so no
  // email - this is the only thing on the page that would ever say so.
  fetch("last-run.json?t=" + Date.now()).then(r => r.ok ? r.json() : null).then(hb => {
    const el = document.getElementById("heartbeat")
    if (!el || !hb || !hb.utc) return
    const ageH = (Date.now() - Date.parse(hb.utc)) / 36e5
    if (!isFinite(ageH)) return
    // Silent when healthy - he did not want a stamp sitting in the footer on
    // a normal day. It speaks ONLY when the nightly job has gone quiet, which
    // is the case no email can cover: a cron that never fires raises no
    // failure, so silence and success otherwise look identical.
    if (ageH <= 36) return
    el.textContent = ` \u00b7 nightly update last ran ${Math.floor(ageH / 24)}d ago`
    el.classList.add("stale")
    el.title = "The nightly update has not run in over 36 hours - check the Actions tab"
    el.hidden = false
  }).catch(() => {})

  // boot() never reaches renderMoves(), so the rows are named here - they are
  // on screen from page load, before a colour has been picked
  renderPlayers()

  const verEl = document.getElementById("model-version")
  if (verEl && styleModel && styleModel.version) {
    verEl.textContent = "v" + styleModel.version
    verEl.hidden = false
    // The version is the feature contract, not the weights - a nightly retrain
    // does not move it. The trained-at stamp is what actually answers "have I
    // got this morning's model, or a copy my browser kept?"
    let tip = `style model v${styleModel.version} - ${styleModel.n_features} features`
    if (styleModel.trained_utc) {
      const t = Date.parse(styleModel.trained_utc)
      if (isFinite(t)) {
        const ageH = (Date.now() - t) / 36e5
        const when = ageH < 1 ? "under an hour ago"
          : ageH < 48 ? `${Math.round(ageH)}h ago`
          : `${Math.round(ageH / 24)} days ago`
        tip += `\ntrained ${when} (${styleModel.trained_utc})`
      }
    }
    verEl.title = tip
  }
  renderStats(stats)

  const meBtn = document.getElementById("bb-login")
  if (meBtn) meBtn.addEventListener("click", meClick)
  renderMe()
  try {
    const stored = localStorage.getItem("bb-key")
    if (stored) keyStatus(stored).then(st => {
      // only a definitive rejection signs out - a flaky network does not
      if (st === "bad") { try { localStorage.removeItem("bb-key") } catch (e) {} }
      renderMe()
    })
  } catch (e) {}

  boardRef = new Chessboard(document.getElementById("board"), {
    position: chess.fen(),
    assetsUrl: "vendor/cm-chessboard/assets/",
    style: { borderType: BORDER_TYPE.none, pieces: { file: "pieces/staunty.svg" }, animationDuration: 250 },
    orientation: COLOR.white,
    extensions: [
      { class: Markers, props: { autoMarkers: MARKER_TYPE.square } },
      { class: PromotionDialog },
      { class: Accessibility, props: { visuallyHidden: true } },
    ],
  })

  document.getElementById("pick-white").addEventListener("click", () => startGame("w"))
  document.getElementById("pick-black").addEventListener("click", () => startGame("b"))
  document.getElementById("again").addEventListener("click", playAgain)
  document.getElementById("resign-btn").addEventListener("click", resignClick)
  document.getElementById("resign-no").addEventListener("click", () => {
    document.getElementById("confirm").hidden = true
  })
  document.getElementById("resign-yes").addEventListener("click", () => {
    document.getElementById("confirm").hidden = true
    if (gameActive) finish("l", "You resigned — I'll take it.")
  })
  setStatus("book", "new game", "Pick your color to start.")
  document.addEventListener("keydown", browseKey)
  placeControls()
  fitMoveList()

  // sound on/off for the whole site, remembered like the theme. Speaker with
  // waves when on; speaker with a line struck through it when off.
  const soundBtn = document.getElementById("sound-toggle")
  if (soundBtn) {
    const SPK = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>'
    const ON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      SPK + '<path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>'
    // a small x takes the WAVES' place rather than cutting across the speaker:
    // a full diagonal fragments the speaker into unreadable pieces at 18px,
    // while this leaves it whole and in the same spot in both states, so
    // toggling only swaps waves for x
    const OFF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      SPK + '<line x1="16.2" y1="9.4" x2="21.3" y2="14.6"/><line x1="21.3" y1="9.4" x2="16.2" y2="14.6"/></svg>'
    const renderSound = () => {
      const off = sfx.isMuted()
      soundBtn.innerHTML = off ? OFF : ON
      soundBtn.setAttribute("aria-label", off ? "Turn sounds on" : "Turn sounds off")
      soundBtn.setAttribute("aria-pressed", String(off))
      soundBtn.title = off ? "Sounds off" : "Sounds on"
    }
    soundBtn.addEventListener("click", () => {
      const turningOn = sfx.isMuted()
      sfx.setMuted(!turningOn)
      renderSound()
      if (turningOn) { sfx.unlock(); sfx.soundOn() }
    })
    renderSound()
  }
  // unlock audio on the FIRST touch anywhere, not just on Play - otherwise a
  // phone that taps the board or the theme toggle first stays silent all game
  const unlockOnce = () => {
    sfx.unlock()
    document.removeEventListener("pointerdown", unlockOnce)
    document.removeEventListener("touchend", unlockOnce)
  }
  document.addEventListener("pointerdown", unlockOnce)
  document.addEventListener("touchend", unlockOnce)
  // coming back from the background leaves the context suspended on iOS
  document.addEventListener("visibilitychange", () => { if (!document.hidden) sfx.unlock() })
  let relayout
  window.addEventListener("resize", () => {
    clearTimeout(relayout)
    relayout = setTimeout(() => { placeControls(); fitMoveList(); revealCurrentMove() }, 120)
  })

  // lift the boot veil (index.html adds .booting before first paint): the
  // board and stats are built by now - wait for the initial position to
  // render and the webfonts to land, capped so a slow font CDN can't hold
  // the page hostage; index.html's own 3s timeout covers a boot() failure
  try { await boardRef.setPosition(chess.fen(), false) } catch (e) {}
  try {
    if (document.fonts && document.fonts.ready) {
      await Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 1200))])
    }
  } catch (e) {}
  // removed directly, not in requestAnimationFrame: rAF callbacks do not run
  // in a hidden tab, which would hold the veil for anyone who opens the site
  // in a background tab until the inline fallback fires
  document.documentElement.classList.remove("booting")

  // dev hooks (console-only): load a FEN, drive moves, inspect state
  window.bb = {
    newGame,
    turn: () => chess.turn(),
    over: () => chess.isGameOver(),
    legal: () => chess.moves({ verbose: true }).map(m => m.from + m.to + (m.promotion || "")),
    move(uci) {
      const m = tryMove({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined })
      if (!m) return false
      try { boardRef.disableMoveInput() } catch (e) {}
      boardRef.setPosition(chess.fen(), false)
      renderMoves()
      botMove(boardRef, gameId)
      return m.san
    },
    load(fen, userColor = "w") {
      gameId++
      stdStart = false
      chess = new Chess(fen)
      inBook = false
      resetOpening()
      statusByPly = {}
      botColor = userColor === "w" ? "b" : "w"
      document.getElementById("pick").hidden = true
      document.getElementById("end").hidden = true
      document.getElementById("confirm").hidden = true
      showEvalBar(true)
      gameActive = true
      setControls(true)
      renderMoves()
      boardRef.disableMoveInput()
      boardRef.setOrientation(userColor === "w" ? COLOR.white : COLOR.black, false)
      boardRef.setPosition(chess.fen(), false).then(() => {
        if (chess.turn() === botColor) {
          botMove(boardRef, gameId)
        } else {
          setStatus("book", "your move", "Custom position — your move.")
          boardRef.enableMoveInput(inputHandler, chess.turn() === "w" ? COLOR.white : COLOR.black)
          updateEval(chess.fen())
        }
      })
    },
  }
}

boot()
