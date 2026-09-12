// Burkeley Bot — plays from Andrew's real games while the position is known,
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
const CAPTURE_URL = "https://prompt-yourself-bot.andrewburke225.workers.dev/chess/game"

// ---------- engine (single-threaded Stockfish 10, tuned to ~1200) ----------

const engine = (() => {
  const worker = new Worker("vendor/stockfish/stockfish.js")
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
    } else if (line.startsWith("bestmove") && onBest) {
      const uci = line.split(" ")[1]
      const resolve = onBest
      onBest = null
      resolve({ uci, lines })
    }
  }
  worker.postMessage("uci")
  return {
    ready,
    // with the style model in charge of humanness, search runs clean:
    // full strength, five candidate lines for the model to choose among
    useMultipv() {
      worker.postMessage("setoption name Skill Level value 20")
      worker.postMessage("setoption name MultiPV value 5")
    },
    bestMove(fen) {
      return new Promise(res => {
        lines = {}
        onBest = res
        worker.postMessage("position fen " + fen)
        worker.postMessage("go depth 8")
      })
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

function setStatus(kind, label, line) {
  statusDot.className = "dot " + kind
  statusLabel.textContent = label
  statusLine.textContent = line
}

function renderMoves() {
  const hist = chess.history()
  movelistEl.innerHTML = ""
  for (let i = 0; i < hist.length; i += 2) {
    const li = document.createElement("li")
    const w = document.createElement("span"); w.className = "w"; w.textContent = hist[i]
    li.appendChild(w)
    if (hist[i + 1]) {
      const b = document.createElement("span"); b.className = "b"; b.textContent = hist[i + 1]
      li.appendChild(b)
    }
    movelistEl.appendChild(li)
  }
  movelistEl.scrollTop = movelistEl.scrollHeight
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

// ---------- the style model (behavioral cloning over engine candidates) ----------

// feature order is a contract with pipeline/train_style.py - change both or neither
function moveFeatures(uci, cpGapPawns, rank) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4)
  const played = tryMove({ from, to, promotion: uci.slice(4) || undefined })
  if (!played) return null
  chess.undo()
  const x = new Array(15).fill(0)
  x[0] = 1
  x[1] = Math.min(Math.max(cpGapPawns, 0), 5)
  x[2] = rank / 4
  x[3] = played.captured ? 1 : 0
  x[4] = played.san.includes("+") || played.san.includes("#") ? 1 : 0
  x[5] = played.promotion ? 1 : 0
  x[6] = played.flags.includes("k") || played.flags.includes("q") ? 1 : 0
  x[7 + "pnbrqk".indexOf(played.piece)] = 1
  const tf = to.charCodeAt(0) - 97, tr = to.charCodeAt(1) - 49
  x[13] = (Math.abs(tf - 3.5) + Math.abs(tr - 3.5)) / 7
  const fr = from.charCodeAt(1) - 49
  x[14] = (played.color === "w" ? tr > fr : tr < fr) ? 1 : 0
  return x
}

// sample a move from the model's probabilities over the engine's candidates
function stylePick(lines) {
  if (!styleModel) return null
  const cands = Object.keys(lines).sort((a, b) => a - b).map(k => lines[k])
  if (cands.length < 2) return null
  const best = cands[0].cp
  const scored = []
  for (let i = 0; i < cands.length; i++) {
    const x = moveFeatures(cands[i].uci, (best - cands[i].cp) / 100, i)
    if (!x) continue
    scored.push({ uci: cands[i].uci, z: x.reduce((sum, v, j) => sum + v * styleModel.weights[j], 0) })
  }
  if (scored.length < 2) return null
  const zmax = Math.max(...scored.map(c => c.z))
  let total = 0
  for (const c of scored) { c.p = Math.exp(c.z - zmax); total += c.p }
  let r = Math.random() * total
  for (const c of scored) {
    r -= c.p
    if (r <= 0) return c.uci
  }
  return scored[0].uci
}

// ---------- game capture (keyed - only games I flag as mine are stored) ----------

function captureGame() {
  let key = null
  try { key = localStorage.getItem("bb-key") } catch (e) {}
  const hist = chess.history({ verbose: true })
  if (!key || !stdStart || hist.length < 6 || !chess.isGameOver()) return
  let result = "d"
  if (chess.isCheckmate()) result = chess.turn() === botColor ? "w" : "l"
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
  if (chess.isGameOver()) { finish() ; return }
  setStatus("thinking", "thinking", "…")
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
  if (!played) { finish(); return }

  await board.setPosition(chess.fen(), true)
  renderMoves()
  if (chess.isGameOver()) { finish(); return }

  if (source) {
    const pct = Math.round(100 * (source.wins + source.draws / 2) / source.n)
    const times = source.n === 1 ? "once before" : source.n + " of " + source.total + " times"
    setStatus("book", "book move", played.san + " — I've played this here " + times + " (" + pct + "% score with it). Your move." + checkNote())
  } else {
    if (inBook) {
      inBook = false
      setStatus("engine", "on my own", played.san + " — we've left my games, so I'm thinking for myself now. Your move." + checkNote())
    } else {
      setStatus("engine", "on my own", played.san + ". Your move." + checkNote())
    }
  }
  board.enableMoveInput(inputHandler, botColor === "w" ? COLOR.black : COLOR.white)
}

function finish() {
  setStatus("over", "game over", gameOverLine())
  boardRef.disableMoveInput()
  captureGame()
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
            tryMove({ from: event.squareFrom, to: event.squareTo, promotion: res.piece.charAt(1) })
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
  botColor = userColor === "w" ? "b" : "w"
  document.getElementById("new-white").classList.toggle("active", userColor === "w")
  document.getElementById("new-black").classList.toggle("active", userColor === "b")
  renderMoves()
  boardRef.disableMoveInput()
  boardRef.setOrientation(userColor === "w" ? COLOR.white : COLOR.black, false)
  boardRef.setPosition(chess.fen(), false).then(() => {
    if (botColor === "w") {
      botMove(boardRef, gameId)
    } else {
      setStatus("book", "your move", "You have the white pieces — go ahead.")
      boardRef.enableMoveInput(inputHandler, COLOR.white)
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
  try {
    const params = new URLSearchParams(location.search)
    if (params.get("key")) {
      localStorage.setItem("bb-key", params.get("key"))
      history.replaceState(null, "", location.pathname)
    }
  } catch (e) {}
  const [bookRes, statsRes, styleRes] = await Promise.all([
    fetch("book.json"), fetch("stats.json"), fetch("style.json?v=1").catch(() => null),
  ])
  book = await bookRes.json()
  const stats = await statsRes.json()
  try {
    if (styleRes && styleRes.ok) styleModel = await styleRes.json()
  } catch (e) { styleModel = null }
  if (styleModel && Array.isArray(styleModel.weights)) {
    engine.ready.then(() => engine.useMultipv())
  } else {
    styleModel = null
  }
  renderStats(stats)

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

  document.getElementById("new-white").addEventListener("click", () => newGame("w"))
  document.getElementById("new-black").addEventListener("click", () => newGame("b"))
  newGame("w")

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
      botColor = userColor === "w" ? "b" : "w"
      renderMoves()
      boardRef.disableMoveInput()
      boardRef.setOrientation(userColor === "w" ? COLOR.white : COLOR.black, false)
      boardRef.setPosition(chess.fen(), false).then(() => {
        if (chess.turn() === botColor) {
          botMove(boardRef, gameId)
        } else {
          setStatus("book", "your move", "Custom position — your move.")
          boardRef.enableMoveInput(inputHandler, chess.turn() === "w" ? COLOR.white : COLOR.black)
        }
      })
    },
  }
}

boot()
