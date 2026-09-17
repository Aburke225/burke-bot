// Play a game against a built bot, on a real board.
//
// This is Burke Bot's play loop (docs/app.js) lifted out of the page it grew
// in. app.js owns a single global game: module-scoped `chess`, `botColor`,
// `boardRef`, element lookups by hard-coded id. None of that survives a page
// that builds a bot and then plays it, so everything here lives inside one
// createGame() closure and every element arrives as an argument.
//
// What is deliberately IDENTICAL to app.js, because it is the product and not
// the plumbing:
//   - the depth/width contract with pipeline/analyse.py (5/20 pool, 2/20
//     horizon, temperature 1.0)
//   - the synthesized sounds, note for note, and which one each move type gets
//   - the eval bar: its OWN engine at depth 18, the logistic mapping, the clamp
//   - captured pieces drawn from the board's own sprite, and the +N from the
//     placement field
//   - arrow-key browsing, including replaying a step's sound
//
// THE ONE THING MOST LIKELY TO GO SILENTLY WRONG is the v9 context. v9.js's
// pick() builds a no-history context, which is right for a one-off position and
// wrong for move 30 of a game: features 12, 13, 40, 43 and 44 all read the
// previous move and all collapse to constants without it, and NOTHING throws.
// So this module never calls pick(); it builds the context from
// chess.history() and calls pickWithContext(). See buildContext() below.

import { INPUT_EVENT_TYPE, COLOR, Chessboard, BORDER_TYPE } from "../vendor/cm-chessboard/src/Chessboard.js"
import { MARKER_TYPE, Markers } from "../vendor/cm-chessboard/src/extensions/markers/Markers.js"
import { PROMOTION_DIALOG_RESULT_TYPE, PromotionDialog } from "../vendor/cm-chessboard/src/extensions/promotion-dialog/PromotionDialog.js"
import { Accessibility } from "../vendor/cm-chessboard/src/extensions/accessibility/Accessibility.js"
import { Chess } from "../vendor/chess.js"

import { makeContext, pickWithContext } from "./v9.js"
import { createEngine } from "./engine.js"
import { bookKey, trieToMoveMap } from "./share.js"

// ---------------------------------------------------------------------------
// contracts with the pipeline (app.js's DEPTH / MULTIPV / HORIZON_DEPTH /
// EVAL_DEPTH / PLAY_TEMP - see the comments there for why each is what it is)

const POOL_DEPTH = 5
const MULTIPV = 20
const HORIZON_DEPTH = 2
const EVAL_DEPTH = 18
const PLAY_TEMP = 1.0

// the bot never answers instantly: a move that lands the moment you release
// yours reads as a machine, not an opponent
const MIN_THINK_MS = 500
const BOOK_THINK_MS = [400, 1100]

// assets resolve against THIS module, not the page: a harness in another
// directory would otherwise ask for its own ../vendor
const BOARD_ASSETS = new URL("../vendor/cm-chessboard/assets/", import.meta.url).href

// ---------------------------------------------------------------------------
// the player rows

// NOT the v9 feature extractor's piece values - this one has no king, which is
// what makes the FEN-placement material count below work.
const CAP_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9 }

const PIECE_VIEWBOX = "6 3 28 34"   // crops the sprite tile's padding
const ICON_VIEWBOX = "-1.04 -1.04 26.09 26.09"
const PAWN_VIEWBOX = "-2.2 -2.25 28.4 28.4"

// The challenger wears a person, drawn in the same 2px round-cap language as
// the rest of the site's icons.
const HUMAN_AVATAR = [
  '<svg viewBox="' + PAWN_VIEWBOX + '" aria-hidden="true">',
  '<g fill="#e9ecd6">',
  '<circle cx="12" cy="6.4" r="4.2"/>',
  '<path d="M8.5 11.5h7l-.8 2H9.3z"/>',
  '<path d="M9.71 13.3C9.71 16.3 8.48 18.1 7.07 19.3H16.93C15.52 18.1 14.29 16.3 14.29 13.3Z"/>',
  '<path d="M6.32 19.1h11.36a1.1 1.1 0 0 1 1.1 1.1v1.5H5.22v-1.5a1.1 1.1 0 0 1 1.1-1.1z"/>',
  '</g>',
  '<g fill="#527a4b"><circle cx="10.15" cy="5.3" r=".85"/><circle cx="13.85" cy="5.3" r=".85"/></g>',
  '<path d="M10.4 7.8Q12 9.1 13.6 7.8" fill="none" stroke="#527a4b" ' +
    'stroke-width="1.25" stroke-linecap="round"/>',
  '</svg>',
].join("")

// The bot wears the site's own mark: rook battlements over a visor of eight
// board squares, on the flared collar every chess piece stands on, drawn in
// the board's two square colours.
const BOT_AVATAR = [
  '<svg viewBox="' + ICON_VIEWBOX + '" aria-hidden="true">',
  '<defs><clipPath id="bb-visor-play">',
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
  '<g fill="#8fc57f" clip-path="url(#bb-visor-play)">',
  '<rect x="6.6" y="8.4" width="2.7" height="2.7"/>',
  '<rect x="12" y="8.4" width="2.7" height="2.7"/>',
  '<rect x="9.3" y="11.1" width="2.7" height="2.7"/>',
  '<rect x="14.7" y="11.1" width="2.7" height="2.7"/>',
  '</g>',
  '<rect x="8.6" y="14.9" width="6.8" height="1.4" rx=".7" fill="#2f4a2a"/>',
  '<circle cx="12" cy="1.4" r="1.4" fill="#8fc57f"/>',
  '</svg>',
].join("")

// the captured pieces are the board's OWN artwork: cm-chessboard caches the
// sprite into #cm-chessboard-sprite on document.body, so <use href="#wp"> is
// literally the piece standing on the board
function pieceGlyph(colour, type) {
  return '<svg viewBox="' + PIECE_VIEWBOX + '" aria-hidden="true">' +
         '<use href="#' + colour + type + '"/></svg>'
}

// ---------------------------------------------------------------------------
// sounds (synthesized with WebAudio - no files, no licenses)
//
// Ported wholesale from app.js's sfx IIFE, with two changes: it is a factory
// rather than a singleton (two games on one page must not share a mute flag),
// and muting is caller state rather than localStorage - the host owns that
// preference and passes it in.

function makeSfx(startMuted) {
  let ctx = null
  const ac = () => ctx || (ctx = new (window.AudioContext || window.webkitAudioContext)())
  let muted = !!startMuted

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
    setMuted(on) { muted = !!on },
    // frequencies/weights tuned so each synth's spectrum lands on the
    // measured band profile of the matching chess.com sample
    move() { tock([460, 900, 1280], [0.75, 1, 0.4], 0.5) },
    capture() { tock([473, 938, 1103], [0.95, 1, 0.7], 0.63, 0, 0.0065, 0.06) },
    castle() { tock([200, 420, 750], [1, 0.75, 0.6], 0.5, 0, 0.005, 0.06); tock([230, 460, 780], [1, 0.75, 0.6], 0.5, 0.09, 0.005, 0.06) },
    check() { tock([882, 1260], [1, 0.6], 0.63, 0, 0.0042, 0.07) },
    soundOn() { tock([540, 1030], [1, 0.5], 0.42, 0, 0.005, 0.05)
                tock([760, 1440], [1, 0.5], 0.42, 0.07, 0.005, 0.05) },
    promote() { blip(440, 0.09, 0.1); blip(660, 0.13, 0.1, 0.09) },
    start() { blip(392, 0.1, 0.1); blip(523, 0.15, 0.1, 0.1) },
    end() { blip(523, 0.1, 0.1); blip(392, 0.18, 0.1, 0.1) },
    close() { try { if (ctx) ctx.close() } catch (e) {} ctx = null },
  }
}

// one sound per move, chess.com priorities: check > promote > capture > castle > plain
function playMoveSound(sfx, m) {
  if (!m) return
  if (m.san.includes("+") || m.san.includes("#")) sfx.check()
  else if (m.promotion) sfx.promote()
  else if (m.captured) sfx.capture()
  else if (m.flags.includes("k") || m.flags.includes("q")) sfx.castle()
  else sfx.move()
}

// ---------------------------------------------------------------------------
// small helpers

const sleep = (ms) => new Promise(res => setTimeout(res, ms))

function uciArgs(uci) {
  return { from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined }
}

/** first match among several selectors, searched inside `root` only */
function findIn(root, selectors) {
  if (!root) return null
  for (const sel of selectors) {
    const el = root.querySelector(sel)
    if (el) return el
  }
  return null
}

/**
 * The book may arrive three ways, and only one of them is documented, so take
 * all three rather than fail on a caller's reasonable guess:
 *   - a move map { bookKey: { uci: {san, n, w, d} } }   (what pickBookMove eats)
 *   - a hydrated trie node from share.js's hydrateTrie() (has .kids)
 *   - null / undefined / {}                             (no book)
 * A hydrated trie is converted by share.js's own trieToMoveMap, owner edges
 * only - the bridge edges in a trie are the OPPONENT's replies and were never
 * the bot's choices.
 */
function normaliseBook(book) {
  if (!book) return {}
  if (Array.isArray(book.kids)) {
    // A RAW trie (straight out of decode()) has the same shape but its edges
    // carry only from/to/promo indices - no uci, no san. Feeding that to
    // trieToMoveMap builds a map keyed "undefined", and every book lookup then
    // misses in silence. Say so instead.
    if (book.kids.length && !book.kids[0].uci) {
      throw new Error("createGame: book looks like a raw trie - run it through share.js hydrateTrie(trie, {Chess}) first")
    }
    return trieToMoveMap(null, { hydrated: book })
  }
  return book
}

// ---------------------------------------------------------------------------

/**
 * Build a playable game.
 *
 * @param {object} opts
 * @param {Element} opts.boardEl      container the board is built into
 * @param {Element} [opts.evalBarEl]  eval bar track (gets .flip when flipped)
 * @param {Element} [opts.evalFillEl] the filled part; its height is set
 * @param {Element} [opts.evalNumEl]  the number beside the bar
 * @param {object}  [opts.topRow]     {avatarEl, nameEl, capsEl} - the BOT
 * @param {object}  [opts.bottomRow]  {avatarEl, nameEl, capsEl} - the PLAYER
 * @param {Element} [opts.pickOverlay]    holds [data-pick="w"] / [data-pick="b"]
 * @param {Element} [opts.confirmOverlay] holds [data-resign="yes"] / [data-resign="no"]
 * @param {Element} [opts.endOverlay]     holds [data-end-title], [data-end-line], [data-again]
 *        (app.js's own ids - #pick-white, #resign-yes, #end-title, #again … -
 *        are accepted as fallbacks, so index.html's markup works unchanged)
 * @param {Element} [opts.resignBtn]  enabled while a game is live; opens confirm
 * @param {object}  opts.bot          { weights: Float64Array(58), book, meta }
 *        meta: { username, rating, temp, poolDepth, horizonDepth, multipv }
 * @param {object}  opts.engine       an engine.js instance, already created
 * @param {boolean} [opts.muted]
 * @param {function}[opts.onStatus]   called with a state object on every change -
 *        each move, each browse step, and each time the eval bar lands a new
 *        score (the whole state goes every time, so a host can just re-render)
 * @param {boolean} [opts.ownsEngine] destroy() also quits opts.engine (default
 *        false: the caller created it, the caller keeps it)
 * @param {object}  [opts.evalEngineOpts] passed to createEngine for the eval bar
 * @returns {Promise<{start, resign, setMuted, destroy}>}
 */
export async function createGame(opts = {}) {
  const {
    boardEl,
    evalBarEl = null, evalFillEl = null, evalNumEl = null,
    topRow = {}, bottomRow = {},
    pickOverlay = null, confirmOverlay = null, endOverlay = null, resignBtn = null,
    bot = {}, engine = null,
    muted = false,
    onStatus = null,
    ownsEngine = false,
    evalEngineOpts = {},
  } = opts

  if (!boardEl) throw new Error("createGame: boardEl is required")
  if (!engine) throw new Error("createGame: an engine.js instance is required")

  const meta = bot.meta || {}
  const weights = bot.weights || null
  const book = normaliseBook(bot.book)
  const temp = typeof meta.temp === "number" ? meta.temp : PLAY_TEMP
  const poolDepth = meta.poolDepth || POOL_DEPTH
  const horizonDepth = meta.horizonDepth || HORIZON_DEPTH
  const multipv = meta.multipv || MULTIPV
  // "Burkeley Bot", not "Burkeley" - the row names the OPPONENT, and the
  // opponent is a bot built from that person's games, not the person. Burke Bot
  // labels its own the same way, and these two boards sit one click apart.
  const botName = meta.name
    ? String(meta.name) + " Bot"
    : (meta.username ? String(meta.username) + " Bot" : "Bot")

  const sfx = makeSfx(muted)

  // ----- game state (all of app.js's module globals, scoped to this game) ---
  let chess = new Chess()
  let botColor = "b"
  let gameId = 0
  let gameActive = false
  let inBook = true
  let destroyed = false
  let viewPly = -1          // -1 = the live position; otherwise "after ply N"
  let endHiddenForBrowse = false
  let lastEvalCp = null
  let lastMove = null
  let lastContext = null   // the previous-move facts the last v9 context carried

  // ----- overlay controls --------------------------------------------------
  const pickWhiteBtn = findIn(pickOverlay, ['[data-pick="w"]', '[data-pick="white"]', "#pick-white"])
  const pickBlackBtn = findIn(pickOverlay, ['[data-pick="b"]', '[data-pick="black"]', "#pick-black"])
  const resignYesBtn = findIn(confirmOverlay, ['[data-resign="yes"]', "#resign-yes"])
  const resignNoBtn = findIn(confirmOverlay, ['[data-resign="no"]', "#resign-no"])
  const againBtn = findIn(endOverlay, ["[data-again]", "#again"])
  const endTitleEl = findIn(endOverlay, ["[data-end-title]", "#end-title", ".ov-title"])
  const endLineEl = findIn(endOverlay, ["[data-end-line]", "#end-line", ".ov-line"])

  function show(el, on) { if (el) el.hidden = !on }

  // ----- status ------------------------------------------------------------
  let state = {
    kind: "idle", label: "new game", line: "Pick your colour to start.",
    gameActive: false, inBook: true,
    turn: chess.turn(), ply: 0, viewPly: -1,
    botColor, userColor: "w",
    fen: chess.fen(), lastMove: null, source: null, result: null, evalCp: null,
    context: null,
  }
  function emit(patch) {
    state = Object.assign({}, state, {
      gameActive, inBook,
      turn: chess.turn(), ply: chess.history().length, viewPly,
      botColor, userColor: botColor === "w" ? "b" : "w",
      fen: chess.fen(), lastMove, evalCp: lastEvalCp, context: lastContext,
    }, patch || {})
    if (onStatus) { try { onStatus(state) } catch (e) { console.warn("onStatus threw", e) } }
  }
  function setStatus(kind, label, line, extra) {
    emit(Object.assign({ kind, label, line }, extra || {}))
  }

  // ----- board -------------------------------------------------------------
  const board = new Chessboard(boardEl, {
    position: chess.fen(),
    assetsUrl: opts.assetsUrl || BOARD_ASSETS,
    style: { borderType: BORDER_TYPE.none, pieces: { file: "pieces/staunty.svg" }, animationDuration: 250 },
    orientation: COLOR.white,
    extensions: [
      { class: Markers, props: { autoMarkers: MARKER_TYPE.square } },
      { class: PromotionDialog },
      { class: Accessibility, props: { visuallyHidden: true } },
    ],
  })

  function tryMove(args) {
    try { return chess.move(args) } catch { return null }
  }

  // ----- the eval bar's own engine -----------------------------------------
  // It searches far deeper than the bot plays and must never queue behind a
  // bot move, so it gets its own worker. Lazily, because a visitor who never
  // starts a game should not pay 7 MB and 128 MiB for it - and on a phone the
  // second worker is exactly the allocation that fails, which is why failing
  // here only hides the bar.
  let evalEngine = null, evalBoot = null, evalDead = false, evalToken = 0

  function bootEval() {
    if (evalDead) return Promise.resolve(null)
    if (!evalBoot) {
      evalBoot = createEngine(evalEngineOpts).then(e => {
        if (destroyed) { e.quit(); return null }
        evalEngine = e
        return e
      }).catch(err => {
        evalDead = true
        showEvalBar(false)
        console.warn("eval engine unavailable; the bar stays hidden", err)
        return null
      })
    }
    return evalBoot
  }

  function renderEvalBar(cpWhite) {
    if (!evalBarEl || !evalFillEl || !evalNumEl) return
    const mate = Math.abs(cpWhite) >= 9000
    let p = mate ? (cpWhite > 0 ? 1 : 0) : 1 / (1 + Math.pow(10, -cpWhite / 400))
    if (!mate) p = Math.min(0.95, Math.max(0.05, p))
    evalFillEl.style.height = Math.round(p * 100) + "%"
    evalNumEl.textContent = mate ? "#" : (cpWhite >= 0 ? "+" : "−") + Math.abs(cpWhite / 100).toFixed(1)
    let flipped = false
    try { flipped = board.getOrientation() === COLOR.black } catch (e) {}
    evalBarEl.classList.toggle("flip", flipped)
    lastEvalCp = cpWhite
  }

  function showEvalBar(on) {
    const live = on && !evalDead
    if (evalBarEl) evalBarEl.hidden = !live
    if (evalNumEl) evalNumEl.hidden = !live
  }

  async function updateEval(fen) {
    if (!evalBarEl || evalDead || destroyed) return
    const my = ++evalToken
    const e = await bootEval()
    if (!e || my !== evalToken || destroyed) return
    let lines
    try { lines = await e.analyse(fen, EVAL_DEPTH, 1) } catch (err) { return }
    if (my !== evalToken || destroyed || !lines || !lines.length) return
    const cp = lines[0].cp
    if (cp === null || cp === undefined) return
    // the engine scores from the side to move; the bar is always White's
    renderEvalBar(fen.split(" ")[1] === "w" ? cp : -cp)
    emit({})
  }

  // ----- the player rows ---------------------------------------------------
  function renderPlayers() {
    if (topRow.avatarEl && topRow.avatarEl.dataset.set !== "bot") {
      topRow.avatarEl.className = "avatar bot"
      topRow.avatarEl.innerHTML = BOT_AVATAR
      topRow.avatarEl.dataset.set = "bot"
    }
    if (bottomRow.avatarEl && bottomRow.avatarEl.dataset.set !== "human") {
      bottomRow.avatarEl.className = "avatar human"
      bottomRow.avatarEl.innerHTML = HUMAN_AVATAR
      bottomRow.avatarEl.dataset.set = "human"
    }
    if (topRow.nameEl) {
      topRow.nameEl.textContent = meta.rating ? botName + " (" + meta.rating + ")" : botName
    }
    if (bottomRow.nameEl) bottomRow.nameEl.textContent = "You"
  }

  function fenAtPly(k) {
    const hist = chess.history({ verbose: true })
    if (!hist.length) return chess.fen()
    return k <= 0 ? hist[0].before : hist[Math.min(k, hist.length) - 1].after
  }

  function renderCaptures() {
    if (!topRow.capsEl && !bottomRow.capsEl) return
    // Counted from the MOVE LIST, not by differencing the board: a promotion
    // removes a pawn and adds a queen with nobody capturing anything, and
    // board-differencing would report a phantom captured pawn for the rest of
    // the game. Only up to the ply being viewed, so stepping back un-takes
    // the pieces instead of showing the final tally at move 3.
    const hist = chess.history({ verbose: true })
    const upto = viewPly === -1 ? hist.length : viewPly
    const taken = { w: {}, b: {} }
    for (let i = 0; i < upto; i++) {
      const m = hist[i]
      if (!m.captured) continue
      const victim = m.color === "w" ? "b" : "w"
      taken[victim][m.captured] = (taken[victim][m.captured] || 0) + 1
    }

    // the +N comes from the pieces actually ON the board at that ply, which IS
    // what a player means by being up material - and unlike the capture list it
    // counts a promotion, worth eight points with no capture. CAP_VAL has no
    // king, so kings, digits and slashes all fall through.
    const placement = (viewPly === -1 ? chess.fen() : fenAtPly(viewPly)).split(" ")[0]
    let mat = 0
    for (const ch of placement) {
      const type = ch.toLowerCase()
      if (!CAP_VAL[type]) continue
      mat += (ch === type ? -1 : 1) * CAP_VAL[type]
    }

    const userColor = botColor === "w" ? "b" : "w"
    // a player's row shows what THEY captured, so it carries the enemy's colour
    paintCaptures(topRow.capsEl, taken[userColor], userColor, botColor === "w" ? mat : -mat)
    paintCaptures(bottomRow.capsEl, taken[botColor], botColor, userColor === "w" ? mat : -mat)
  }

  function paintCaptures(el, counts, colour, advantage) {
    if (!el) return
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

  function afterMove() {
    // any real move snaps history browsing back to the live position
    viewPly = -1
    endHiddenForBrowse = false
    try { board.removeMarkers(MARKER_TYPE.square) } catch (e) {}
    renderPlayers()
    renderCaptures()
  }

  // ----- the book ----------------------------------------------------------
  function pickBookMove() {
    const entry = book[bookKey(chess.fen())]
    if (!entry) return null
    const legal = new Set(chess.moves({ verbose: true }).map(m => m.from + m.to + (m.promotion || "")))
    const candidates = Object.entries(entry).filter(([uci]) => legal.has(uci))
    if (!candidates.length) return null
    const total = candidates.reduce((s, [, v]) => s + (v.n || 1), 0)
    let r = Math.random() * total
    for (const [uci, v] of candidates) {
      r -= (v.n || 1)
      if (r <= 0) return { uci, san: v.san, n: v.n, total }
    }
    return null
  }

  // ----- the v9 context: the part that must not be got wrong ---------------
  // Everything v9 knows about the PREVIOUS move comes from here. Hand
  // makeContext nulls when the history is non-empty and features 12, 13, 40,
  // 43 and 44 quietly become constants - no error, just a different bot.
  function buildContext() {
    const hist = chess.history({ verbose: true })
    const oppLast = hist.length ? hist[hist.length - 1] : null
    const myLast = hist.length > 1 ? hist[hist.length - 2] : null
    // myLast is only "mine" if it was played by the side now to move
    const mine = (myLast && myLast.color === chess.turn()) ? myLast : null
    return makeContext(
      chess,
      mine ? mine.to : null,
      mine ? mine.from : null,
      (oppLast && oppLast.captured) ? oppLast.to : null,
      oppLast ? oppLast.to : null,
    )
  }

  // ----- the bot's move ----------------------------------------------------
  async function botMove(id) {
    if (chess.isGameOver()) { finishAuto(); return }
    setStatus("thinking", "thinking", "…")
    updateEval(chess.fen())
    const started = Date.now()

    let played = null, source = null

    const fromBook = pickBookMove()
    if (fromBook) {
      await sleep(BOOK_THINK_MS[0] + Math.random() * (BOOK_THINK_MS[1] - BOOK_THINK_MS[0]))
      if (id !== gameId || destroyed) return
      played = tryMove(uciArgs(fromBook.uci))
      if (played) source = { kind: "book", uci: fromBook.uci, san: fromBook.san }
    }

    if (!played) {
      const fen = chess.fen()
      let cands = null, uci = null, kind = "engine"
      try {
        // the pool the model chooses among, and the same position seen from
        // the player's own horizon - exactly the pair pipeline/analyse.py
        // stores per decision (POOL_DEPTH/MULTIPV then HORIZON_DEPTH/MULTIPV)
        cands = await engine.analyse(fen, poolDepth, multipv)
        if (id !== gameId || destroyed) return
        if (!cands.length) { finishAuto(); return }   // "bestmove (none)": mate or stalemate

        let shByUci = null
        try {
          const hz = await engine.analyse(fen, horizonDepth, multipv)
          if (id !== gameId || destroyed) return
          shByUci = {}
          for (const l of hz) shByUci[l.uci] = l.cp
        } catch (e) {
          // v9's three strongest features are built from the horizon score. If
          // it is missing they collapse to constants and the bot just follows
          // the engine's ordering, so say so rather than failing silently.
          console.warn("horizon search failed; falling back to the engine's move", e)
        }

        if (weights && weights.length && shByUci) {
          const ctx = buildContext()
          // the previous-move facts, published so a host can SEE that the
          // context carried history. Nulls here on anything but the first
          // move mean the context was built wrong and five features have
          // quietly gone constant.
          lastContext = {
            prevMyTo: ctx.prevMyTo, prevMyFrom: ctx.prevMyFrom,
            prevOppCapTo: ctx.prevOppCapTo, lastMoveTo: ctx.lastMoveTo,
          }
          const chosen = pickWithContext(chess, cands, shByUci, weights, temp, ctx)
          if (chosen) { uci = chosen; kind = "model" }
        }
        if (!uci) uci = cands[0].uci
      } catch (err) {
        // The engine died mid-game. A dead board is worse than an honest
        // random move, and the status says which one this was.
        console.warn("analysis failed; playing a legal move at random", err)
        const legal = chess.moves({ verbose: true })
        if (!legal.length) { finishAuto(); return }
        uci = legal[Math.floor(Math.random() * legal.length)].lan
        kind = "fallback"
      }

      const elapsed = Date.now() - started
      if (elapsed < MIN_THINK_MS) await sleep(MIN_THINK_MS - elapsed)
      if (id !== gameId || destroyed) return

      played = tryMove(uciArgs(uci))
      if (!played && cands && cands.length && uci !== cands[0].uci) {
        played = tryMove(uciArgs(cands[0].uci))
      }
      if (played) {
        source = { kind, uci: played.lan, san: played.san, best: cands ? cands[0].uci : null, pool: cands ? cands.length : 0 }
      }
    }

    if (!played) { finishAuto(); return }
    lastMove = { san: played.san, uci: played.lan, from: played.from, to: played.to, color: played.color }
    playMoveSound(sfx, played)

    await board.setPosition(chess.fen(), true)
    if (id !== gameId || destroyed) return
    afterMove()
    if (chess.isGameOver()) { finishAuto(); return }
    updateEval(chess.fen())

    if (source && source.kind === "book") {
      // No percentages. A shared book carries rank order, not the owner's real
      // win counts (share.js zeroes w and d), so any "x% score" line here would
      // be a number that is not in the data.
      setStatus("book", "from my games", "I have played this position before. Your move." + checkNote(), { source: source.kind })
    } else if (source && source.kind === "fallback") {
      setStatus("engine", "engine trouble", "My engine stopped answering, so that was a random legal move. Your move." + checkNote(), { source: source.kind })
    } else if (inBook) {
      inBook = false
      setStatus("engine", "on my own", "We have left my games, so I am thinking for myself now. Your move." + checkNote(), { source: source ? source.kind : null })
    } else {
      setStatus("engine", "on my own", "Your move." + checkNote(), { source: source ? source.kind : null })
    }

    // idempotent: overlapping start paths can reach here with input already on,
    // and cm-chessboard throws on a double enable
    try { board.disableMoveInput() } catch (e) {}
    try { board.enableMoveInput(inputHandler, botColor === "w" ? COLOR.black : COLOR.white) } catch (e) {}
  }

  function checkNote() {
    return chess.inCheck() ? " Check." : ""
  }

  // ----- the player's move -------------------------------------------------
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
        lastMove = { san: result.san, uci: result.lan, from: result.from, to: result.to, color: result.color }
        playMoveSound(sfx, result)
        const id = gameId
        event.chessboard.state.moveInputProcess.then(() => {
          event.chessboard.setPosition(chess.fen(), true).then(() => {
            if (id !== gameId || destroyed) return
            afterMove()
            if (chess.isGameOver()) { finishAuto(); return }
            botMove(id)
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
              const done = tryMove({ from: event.squareFrom, to: event.squareTo, promotion: res.piece.charAt(1) })
              if (done) lastMove = { san: done.san, uci: done.lan, from: done.from, to: done.to, color: done.color }
              playMoveSound(sfx, done)
              const id = gameId
              event.chessboard.setPosition(chess.fen(), true).then(() => {
                if (id !== gameId || destroyed) return
                afterMove()
                if (chess.isGameOver()) { finishAuto(); return }
                botMove(id)
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

  // ----- history browsing (arrow keys, like chess.com) ---------------------
  function browseTo(k) {
    const n = chess.history().length
    if (!n || destroyed) return
    k = Math.max(0, Math.min(n, k))
    const prev = viewPly === -1 ? n : viewPly
    viewPly = k === n ? -1 : k
    const live = viewPly === -1
    // a single step replays that move's sound; a jump gets a plain tap
    if (Math.abs(k - prev) === 1) {
      playMoveSound(sfx, chess.history({ verbose: true })[Math.max(k, prev) - 1])
    } else if (k !== prev) {
      sfx.move()
    }
    const fen = live ? chess.fen() : fenAtPly(k)
    board.setPosition(fen, true)
    renderCaptures()   // the rows describe the position on the board
    // pieces only move at the live position, on the player's turn
    try { board.disableMoveInput() } catch (e) {}
    if (live && gameActive && chess.turn() !== botColor) {
      try { board.enableMoveInput(inputHandler, botColor === "w" ? COLOR.black : COLOR.white) } catch (e) {}
    }
    // the end screen steps aside while reviewing and returns at the end
    if (endOverlay) {
      if (!live && !endOverlay.hidden) { endOverlay.hidden = true; endHiddenForBrowse = true }
      if (live && endHiddenForBrowse) { endOverlay.hidden = false; endHiddenForBrowse = false }
    }
    // highlight the viewed move's from/to squares
    const mark = live ? n - 1 : k - 1
    try {
      board.removeMarkers(MARKER_TYPE.square)
      if (mark >= 0) {
        const m = chess.history({ verbose: true })[mark]
        board.addMarker(MARKER_TYPE.square, m.from)
        board.addMarker(MARKER_TYPE.square, m.to)
      }
    } catch (e) {}
    updateEval(fen)
    emit({ kind: live ? state.kind : "browse" })
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

  // ----- finishing ---------------------------------------------------------
  function gameOverLine() {
    if (chess.isCheckmate()) {
      return chess.turn() === botColor
        ? "Checkmate — you got me. Rematch?"
        : "Checkmate — that one is mine."
    }
    if (chess.isStalemate()) return "Stalemate — nobody wins that one."
    if (chess.isThreefoldRepetition()) return "Draw by repetition."
    if (chess.isInsufficientMaterial()) return "Draw — not enough pieces left to win."
    if (chess.isDraw()) return "Draw."
    return "Game over."
  }

  function autoResult() {
    if (chess.isCheckmate()) return chess.turn() === botColor ? "w" : "l"
    return "d"
  }

  function finishAuto() { finish(autoResult(), gameOverLine()) }

  function finish(result, line) {
    if (!gameActive) return
    gameActive = false
    // a depth-18 search started before the last move is still running, and its
    // answer is about to be wrong: cancel it before painting the final score
    evalToken++
    if (chess.isCheckmate()) renderEvalBar(chess.turn() === "w" ? -10000 : 10000)
    else if (result === "d") renderEvalBar(0)
    sfx.end()
    try { board.disableMoveInput() } catch (e) {}
    setControls(false)
    showEnd(result, line)
    setStatus("over", "game over", line, { result })
  }

  function showEnd(result, line) {
    show(confirmOverlay, false)
    show(pickOverlay, false)
    if (endTitleEl) endTitleEl.textContent = result === "w" ? "You win." : result === "l" ? "You lose." : "Draw."
    if (endLineEl) endLineEl.textContent = line
    show(endOverlay, true)
  }

  function setControls(on) {
    if (resignBtn) resignBtn.disabled = !on
  }

  // ----- starting ----------------------------------------------------------
  function newGame(userColor) {
    gameId++
    chess = new Chess()
    inBook = true
    viewPly = -1
    lastMove = null
    lastContext = null
    botColor = userColor === "w" ? "b" : "w"
    const id = gameId
    renderPlayers()
    renderCaptures()
    try { board.disableMoveInput() } catch (e) {}
    board.setOrientation(userColor === "w" ? COLOR.white : COLOR.black, false)
    board.setPosition(chess.fen(), false).then(() => {
      if (id !== gameId || destroyed) return
      try { board.removeMarkers(MARKER_TYPE.square) } catch (e) {}
      if (botColor === "w") {
        botMove(id)
      } else {
        setStatus("book", "your move", "You have the white pieces — go ahead.")
        try { board.enableMoveInput(inputHandler, COLOR.white) } catch (e) {}
        updateEval(chess.fen())
      }
    })
  }

  function start(userColor) {
    if (destroyed) return
    const c = userColor === "b" || userColor === COLOR.black ? "b" : "w"
    show(pickOverlay, false)
    show(confirmOverlay, false)
    show(endOverlay, false)
    endHiddenForBrowse = false
    sfx.unlock()
    sfx.start()
    showEvalBar(true)
    gameActive = true
    newGame(c)
    setControls(true)
  }

  function playAgain() {
    if (destroyed) return
    gameId++
    gameActive = false
    chess = new Chess()
    inBook = true
    viewPly = -1
    lastMove = null
    lastContext = null
    try { board.disableMoveInput() } catch (e) {}
    setControls(false)
    board.setOrientation(COLOR.white, false)
    board.setPosition(chess.fen(), false)
    try { board.removeMarkers(MARKER_TYPE.square) } catch (e) {}
    renderCaptures()
    renderEvalBar(0)
    showEvalBar(false)
    show(confirmOverlay, false)
    show(endOverlay, false)
    show(pickOverlay, true)
    setStatus("idle", "new game", "Pick your colour to start.", { result: null, source: null })
  }

  // ----- listeners ---------------------------------------------------------
  const listeners = []
  function on(el, type, fn) {
    if (!el) return
    el.addEventListener(type, fn)
    listeners.push([el, type, fn])
  }

  on(pickWhiteBtn, "click", () => start("w"))
  on(pickBlackBtn, "click", () => start("b"))
  on(againBtn, "click", playAgain)
  on(resignBtn, "click", () => { if (gameActive) show(confirmOverlay, true) })
  on(resignNoBtn, "click", () => show(confirmOverlay, false))
  on(resignYesBtn, "click", () => {
    show(confirmOverlay, false)
    if (gameActive) finish("l", "You resigned — I will take it.")
  })
  on(document, "keydown", browseKey)

  // unlock audio on the FIRST gesture anywhere, not just on Play - otherwise a
  // phone that taps the board first stays silent all game
  const unlockOnce = () => {
    sfx.unlock()
    document.removeEventListener("pointerdown", unlockOnce)
    document.removeEventListener("touchend", unlockOnce)
  }
  document.addEventListener("pointerdown", unlockOnce)
  document.addEventListener("touchend", unlockOnce)
  // coming back from the background leaves the context suspended on iOS
  const onVisible = () => { if (!document.hidden) sfx.unlock() }
  on(document, "visibilitychange", onVisible)

  // ----- first paint -------------------------------------------------------
  renderPlayers()
  renderCaptures()
  showEvalBar(false)
  show(confirmOverlay, false)
  show(endOverlay, false)
  show(pickOverlay, true)
  setControls(false)
  try { await board.setPosition(chess.fen(), false) } catch (e) {}
  emit({})

  // ----- the handle --------------------------------------------------------
  return {
    /** "w" or "b" - the colour the PLAYER takes */
    start,
    /** resign right now, with no confirm step (the button asks first) */
    resign() { if (gameActive) finish("l", "You resigned — I will take it.") },
    /**
     * Make the PLAYER's move from code, the same way dragging a piece would:
     * a uci string, the same sound, the same board animation, and the bot
     * replies. Returns the SAN played, or null if it was not legal or not the
     * player's turn. For a host that wants keyboard entry, a hint button or a
     * scripted game - and it is how this module is tested without a mouse.
     */
    async playerMove(uci) {
      if (!gameActive || destroyed) return null
      if (viewPly !== -1) browseTo(chess.history().length)
      if (chess.turn() === botColor) return null
      const m = tryMove(uciArgs(String(uci)))
      if (!m) return null
      lastMove = { san: m.san, uci: m.lan, from: m.from, to: m.to, color: m.color }
      playMoveSound(sfx, m)
      const id = gameId
      try { board.disableMoveInput() } catch (e) {}
      await board.setPosition(chess.fen(), true)
      if (id !== gameId || destroyed) return m.san
      afterMove()
      if (chess.isGameOver()) { finishAuto(); return m.san }
      botMove(id)
      return m.san
    },
    setMuted(b) { sfx.setMuted(b) },
    /** the live state object, same shape onStatus receives */
    getState() { return state },
    /** for a host that wants its own "play again" control */
    playAgain,
    destroy() {
      if (destroyed) return
      destroyed = true
      gameId++              // abandons any bot move still in flight
      gameActive = false
      evalToken++
      for (const [el, type, fn] of listeners) {
        try { el.removeEventListener(type, fn) } catch (e) {}
      }
      listeners.length = 0
      document.removeEventListener("pointerdown", unlockOnce)
      document.removeEventListener("touchend", unlockOnce)
      try { board.disableMoveInput() } catch (e) {}
      try { board.destroy() } catch (e) {}
      // cm-chessboard's destroy() fires its extensions' destroy points (so the
      // Accessibility extension's own document keydown listener does go), but
      // view.destroy() only removes the view's container - the accessibility
      // block and the live region were appended to OUR element and stay.
      // Measured: boardEl still held a .cm-chessboard-accessibility form after
      // destroy, which would stack up on a second createGame.
      try {
        boardEl.querySelectorAll('[class*="cm-chessboard-"]').forEach(el => el.remove())
      } catch (e) {}
      // the eval worker is ours, so it always goes; the move engine belongs to
      // the caller unless they said otherwise
      if (evalEngine) { try { evalEngine.quit() } catch (e) {} }
      else if (evalBoot) { evalBoot.then(e => { if (e) { try { e.quit() } catch (err) {} } }) }
      evalEngine = null
      if (ownsEngine && engine) { try { engine.quit() } catch (e) {} }
      sfx.close()
    },
  }
}
