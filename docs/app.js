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
// sampling temperature: 1 plays the learned distribution exactly; below 1
// leans toward my most likely choices and trims the blunder tail.
// keep in sync with PLAY_TEMP in pipeline/audit.py
const PLAY_TEMP = 0.8

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
      worker.postMessage("setoption name MultiPV value 10")
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
}

function setStatus(kind, label, line) {
  setStatusRaw(kind, label, line)
  lastLiveStatus = { kind, label, line }
  if (kind !== "thinking") statusByPly[chess.history().length] = { kind, label, line }
}

function renderMoves() {
  const hist = chess.history()
  movelistEl.innerHTML = ""
  for (let i = 0; i < hist.length; i += 2) {
    const li = document.createElement("li")
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
  movelistEl.scrollTop = movelistEl.scrollHeight
  // the freshest move wears the highlight (the chat no longer names moves)
  const last = movelistEl.querySelector('[data-ply="' + (hist.length - 1) + '"]')
  if (last) last.classList.add("cur")
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
  let ctx = null
  const ac = () => ctx || (ctx = new (window.AudioContext || window.webkitAudioContext)())
  // a piece landing, modeled on measurements of chess.com's real samples:
  // their move sound is PITCHED, not noise - a couple of damped tones near
  // 460+900Hz, ~1ms attack, dead within ~20ms, nothing below 300Hz or
  // above 2400Hz. So: a few exponentially damped sines, instantly muted.
  function tock(freqs, weights, gain, when = 0, tau = 0.004, dur = 0.05) {
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
      const s = c.createBufferSource(); s.buffer = buf
      const g = c.createGain(); g.gain.value = gain
      s.connect(g); g.connect(c.destination)
      s.start(t)
    } catch (e) {}
  }
  function blip(freq, dur, gain, when = 0, slide = 0) {
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
    unlock() { try { ac().resume() } catch (e) {} },
    // frequencies/weights tuned so each synth's spectrum lands on the
    // measured band profile of the matching chess.com sample
    // his picks from the tasting panel: A move, B capture, A castle, D check
    move() { tock([460, 900, 1280], [0.75, 1, 0.4], 0.5) },
    capture() { tock([473, 938, 1103], [0.95, 1, 0.7], 0.63, 0, 0.0065, 0.06) },
    castle() { tock([200, 420, 750], [1, 0.75, 0.6], 0.5, 0, 0.005, 0.06); tock([230, 460, 780], [1, 0.75, 0.6], 0.5, 0.09, 0.005, 0.06) },
    check() { tock([882, 1260], [1, 0.6], 0.63, 0, 0.0042, 0.07) },
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
    if (span) { span.classList.add("cur"); span.scrollIntoView({ block: "nearest" }) }
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
  await engine.ready
  const result = await engine.bestMove(fen)
  if (my !== evalToken) return  // a newer position took over
  const line = result.lines[1]
  if (!line) return
  const cpWhite = fen.split(" ")[1] === "w" ? line.cp : -line.cp
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
  let ek = null
  for (const row of chess.board()) {
    for (const sq of row) {
      if (sq && sq.type === "k" && sq.color === userColor) ek = sq.square
    }
  }
  const mineLast = myLast && myLast.color === botColor ? myLast : null
  return {
    prevMyTo: mineLast ? mineLast.to : null,
    prevMyFrom: mineLast ? mineLast.from : null,
    prevOppCapTo: oppLast && oppLast.captured ? oppLast.to : null,
    ek,
    tension: Math.min(1, chess.moves({ verbose: true }).filter(m => m.captured).length / 10),
    userColor,
  }
}

function moveFeatures(uci, cpGapPawns, rank, ctx) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4)
  // attack facts are judged in the pre-move position, so read them first
  const fromAtt = chess.isAttacked(from, ctx.userColor) ? 1 : 0
  const toAtt = chess.isAttacked(to, ctx.userColor) ? 1 : 0
  const played = tryMove({ from, to, promotion: uci.slice(4) || undefined })
  if (!played) return null
  // post-move facts, judged with the move on the board
  const defended = chess.isAttacked(to, played.color) ? 1 : 0
  const hangs = !defended && chess.isAttacked(to, ctx.userColor) ? 1 : 0
  let leaves = 0
  for (const row of chess.board()) {
    for (const sq of row) {
      if (!sq || sq.color !== played.color || sq.square === to) continue
      if (sq.type === "p" || sq.type === "k") continue
      if (chess.isAttacked(sq.square, ctx.userColor) && !chess.isAttacked(sq.square, played.color)) {
        leaves = 1
        break
      }
    }
    if (leaves) break
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
  if (ctx.ek) {
    const ef = ctx.ek.charCodeAt(0) - 97, er = ctx.ek.charCodeAt(1) - 49
    x[20] = CHEB(tf, tr, ef, er) / 7
  }
  x[21] = played.captured && toAtt ? 1 : 0
  x[22] = defended
  x[23] = (played.color === "w" ? fr === 0 : fr === 7) ? 1 : 0
  x[24] = ctx.prevMyTo === from && ctx.prevMyFrom === to ? 1 : 0
  x[25] = hangs
  x[26] = leaves
  x[27] = played.captured && toAtt && PIECE_VAL[played.captured] < PIECE_VAL[played.piece] ? 1 : 0
  x[28] = PIECE_VAL[played.piece] / 9
  x[29] = x[0] * ctx.tension
  return x
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
    scored.push({ uci: cands[i].uci, z })
  }
  if (scored.length < 2) return null
  const zmax = Math.max(...scored.map(c => c.z))
  let total = 0
  for (const c of scored) { c.p = Math.exp((c.z - zmax) / PLAY_TEMP); total += c.p }
  let r = Math.random() * total
  for (const c of scored) {
    r -= c.p
    if (r <= 0) return c.uci
  }
  return scored[0].uci
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
  btn.textContent = has ? "logout" : "It's me"
  btn.title = has
    ? "signed in - finished games train the bot"
    : "Andrew's sign-in for game capture"
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
      input.placeholder = "nope"
      input.focus()
    }
  })
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
    const result = await engine.bestMove(chess.fen())
    if (id !== gameId) return
    const elapsed = Date.now() - started
    if (elapsed < 500) await sleep(500 - elapsed)
    if (id !== gameId) return
    const uci = (styleModel && stylePick(result.lines)) || result.uci
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
  board.enableMoveInput(inputHandler, botColor === "w" ? COLOR.black : COLOR.white)
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
  document.getElementById("draw-btn").disabled = !on
  document.getElementById("resign-btn").disabled = !on
}

// ---------- pick a color / play again / resign / offer a draw ----------

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

async function drawClick() {
  if (!gameActive) return
  const human = botColor === "w" ? "b" : "w"
  if (chess.turn() !== human) return
  const btn = document.getElementById("draw-btn")
  btn.disabled = true
  setStatus("thinking", "draw offer", "Hmm, let me look at the position…")
  await engine.ready
  const myGame = gameId
  const result = await engine.bestMove(chess.fen())
  if (myGame !== gameId || !gameActive) return
  const line1 = result.lines[1]
  const botCp = line1 ? -line1.cp : 0
  const early = chess.moveNumber() <= 15
  // early on, only a clearly losing bot takes the escape hatch; later,
  // any roughly equal (or worse) position is a fair handshake
  if (early ? botCp <= -150 : botCp <= 60) {
    finish("d", "I'll take the draw.")
  } else {
    btn.disabled = false
    setStatus("engine", "draw declined",
      early ? "A draw already? No — let's play on." : "No — I like my position. Your move.")
  }
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
    list.appendChild(dt); list.appendChild(dd)
  }
}

// ---------- boot ----------

async function boot() {
  const [bookRes, statsRes, styleRes, openingsRes] = await Promise.all([
    fetch("book.json"), fetch("stats.json"), fetch("style.json?v=5").catch(() => null),
    fetch("openings.json").catch(() => null),
  ])
  book = await bookRes.json()
  const stats = await statsRes.json()
  openingStats = stats.opening_stats || {}
  favoriteFamily = Object.keys(openingStats).reduce((best, f) =>
    !best || openingStats[f].n > openingStats[best].n ? f : best, null)
  try {
    if (openingsRes && openingsRes.ok) openingsMap = await openingsRes.json()
  } catch (e) { openingsMap = null }
  try {
    if (styleRes && styleRes.ok) styleModel = await styleRes.json()
  } catch (e) { styleModel = null }
  const okModel = styleModel && styleModel.features === "v5" &&
    Array.isArray(styleModel.weights) && Array.isArray(styleModel.active) &&
    styleModel.active.length === styleModel.weights.length &&
    styleModel.active.every(i => Number.isInteger(i) && i >= 0 && i < 30)
  if (okModel) {
    engine.ready.then(() => engine.useMultipv())
  } else {
    styleModel = null
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
  document.getElementById("draw-btn").addEventListener("click", drawClick)
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
