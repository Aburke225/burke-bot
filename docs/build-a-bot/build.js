// Build-a-Bot: username -> bot, entirely in the browser.
//
// This file is ORCHESTRATION ONLY. Every piece of real arithmetic already
// exists and is already verified:
//
//   games.js   downloads and normalises the games
//   engine.js  runs Stockfish in a worker
//   v9.js      the 58-feature contract (features, makeContext, horizonScores)
//   train.js   the conditional-logit fit, L2 by out-of-fold log-likelihood
//   share.js   the book trie and the URL packing
//
// Nothing here re-implements any of them. Where this file does arithmetic of
// its own it is one of exactly three things, each called out where it happens:
//
//   1. the horizon-column patch (patchHorizon), which is a 4-line duplication
//      of v9.features' horizon block and is VERIFIED against the real thing at
//      run time rather than trusted - see verifyPatch();
//   2. the lever measurement (out-of-pool rate by width, pool depth 4 vs 5);
//   3. the rating estimate, which is a documented single-anchor calibration
//      and is honest about being one.
//
// THE PIPELINE
//
//   download -> probe -> score -> fit -> book -> stats
//
// The probe is what makes this work on a stranger. The Python pipeline hard-
// codes MultiPV 20 and pool depth 5 because those were measured on ONE player
// (out-of-pool 5.1% at width 20). A different player has a different pool-miss
// curve, so the width is measured here instead of assumed.
//
// WHAT RUNS WHERE
//
// Engine searches happen in the Stockfish worker, so the main thread is free
// while they run. The only main-thread work is feature extraction and the fit;
// both yield between units of work through scheduler.yield(), falling back to a
// MessageChannel. Not setTimeout, and not requestAnimationFrame: a hidden tab
// throttles the first to about one tick a second and stops the second
// completely, and a build left in a background tab must still finish. See
// makeYielder() for the measurements.

import { Chess } from "../vendor/chess.js"
import { createEngine } from "./engine.js"
import { fetchGames, gameKey } from "./games.js"
import {
  N_FEATURES, FEATURE_NAMES, makeContext, horizonScores, features,
} from "./v9.js"
import { fitAsync } from "./train.js"
import { buildTrie, bookKey, trieStats } from "./share.js"

// ---------------------------------------------------------------------------
// contract constants (each one is a contract with a file in pipeline/)

// pipeline/train_style.py MIN_PLY - the book covers the opening, so the style
// model is only fitted on moves after it.
const MIN_PLY = 8
// pipeline/build.py MAX_BOOK_PLY - the first 15 full moves.
const MAX_BOOK_PLY = 30
// pipeline/analyse.py POOL_DEPTH. The probe may lower this, never raise it.
const DEFAULT_POOL_DEPTH = 5
const MIN_POOL_DEPTH = 4
const MAX_POOL_DEPTH = 8
// The three horizons computed in one pass. pipeline/analyse.py fixed this at 2;
// here all three are measured and the fit chooses. See runFits().
const HORIZONS = [1, 2, 3]
// Widths the probe prices. Width is very nearly free - measured on this
// engine at 11.5 / 11.1 / 10.9 ms for width 10 / 20 / 32 at depth 5 - so the
// only reason not to take 32 is the design matrix it produces.
const WIDTHS = [14, 16, 20, 24, 28, 32]
const PROBE_WIDTH = 32
const PROBE_POSITIONS = 150
// Positions re-searched at the chosen width to confirm the curve did not lie.
const CONFIRM_POSITIONS = 60
const OUT_OF_POOL_TARGET = 0.05
// A single blunder must not own the mean. Same clamp lichess uses on ACPL.
const LOSS_CLAMP_CP = 1000
// Refuse to allocate a design matrix bigger than this rather than dying with
// an out-of-memory the caller cannot interpret.
const MAX_DESIGN_BYTES = 256 * 1024 * 1024

// Feature indices are looked up by NAME, never hard-coded, so a renumbering of
// the v9 contract fails loudly here instead of silently mis-patching a column.
const IX = {
  horizonLoss: FEATURE_NAMES.indexOf("horizon_loss_log"),
  horizonLossQuiet: FEATURE_NAMES.indexOf("horizon_loss_log_quiet"),
  horizonWinprob: FEATURE_NAMES.indexOf("horizon_winprob"),
  imbalanceXHorizon: FEATURE_NAMES.indexOf("imbalance_x_horizon_loss"),
  capture: FEATURE_NAMES.indexOf("capture"),
  givesCheck: FEATURE_NAMES.indexOf("gives_check"),
}
for (const [k, v] of Object.entries(IX)) {
  if (v < 0) throw new Error(`build.js: v9 has no feature named for ${k}`)
}
// The four columns the horizon moves, in the order they are stored in hcols.
const HCOLS = [IX.horizonLoss, IX.horizonLossQuiet, IX.horizonWinprob, IX.imbalanceXHorizon]

export class BuildError extends Error {
  constructor(message, code) {
    super(message)
    this.name = "BuildError"
    this.code = code
  }
}

// ---------------------------------------------------------------------------
// plumbing

function abortError() {
  const e = new Error("Cancelled")
  e.name = "AbortError"
  return e
}

function throwIfAborted(signal) {
  if (signal && signal.aborted) throw abortError()
}

/**
 * A yield to the browser that is neither clamped nor throttled.
 *
 * MEASURED in Chrome, in a BACKGROUND tab, 200 yields each:
 *   scheduler.yield()   0.014 ms
 *   MessageChannel      0.127 ms
 *   setTimeout(..., 0)  did not finish - a hidden tab throttles timers to
 *                       roughly one a second, so a build would take hours the
 *                       moment the user switched away from it
 * requestAnimationFrame is worse still: in a hidden tab it does not fire at all.
 *
 * So: scheduler.yield() where it exists, MessageChannel everywhere else, and
 * setTimeout only as a last resort that should never be reached in a browser
 * new enough to run this page.
 */
function makeYielder() {
  if (typeof scheduler === "object" && scheduler && typeof scheduler.yield === "function") {
    return () => scheduler.yield()
  }
  if (typeof MessageChannel === "function") {
    const ch = new MessageChannel()
    const waiting = []
    ch.port1.onmessage = () => { const r = waiting.shift(); if (r) r() }
    ch.port1.start?.()
    return () => new Promise((res) => { waiting.push(res); ch.port2.postMessage(0) })
  }
  return () => new Promise((res) => setTimeout(res, 0))
}

function reporter(onProgress) {
  return (phase, done, total, label) => {
    if (typeof onProgress !== "function") return
    try {
      onProgress({ phase, done, total, label })
    } catch {
      /* a page that throws inside its own progress handler is not our problem */
    }
  }
}

function safeMove(chess, uci) {
  try {
    return chess.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci.slice(4) || undefined,
    })
  } catch {
    return null
  }
}

/** n indices spread evenly over [0, total). Deterministic on purpose. */
function uniformSample(total, n) {
  if (n >= total) return Array.from({ length: total }, (_, i) => i)
  const out = []
  for (let i = 0; i < n; i++) out.push(Math.floor(((i + 0.5) * total) / n))
  return out
}

const RESULT_CODE = { win: "w", loss: "l", draw: "d" }

// ---------------------------------------------------------------------------
// 0. decision points
//
// One replay per game, recording exactly what pipeline/analyse.py records: the
// position, the move played, and the four previous-move facts v9's context
// needs. They are TRACKED, not reconstructed from history, which is the same
// thing analyse.py does and sidesteps the trap in the play path entirely - a
// context built with nulls where history exists silently changes features
// 12, 13, 40, 43 and 44 and raises no error.

function planDecisions(games) {
  const points = []
  const perGame = new Int32Array(games.length)
  let truncated = 0
  for (let gi = 0; gi < games.length; gi++) {
    const g = games[gi]
    const chess = new Chess()
    let prevMyTo = null, prevMyFrom = null, prevOppCapTo = null, lastTo = null
    for (let ply = 0; ply < g.moves.length; ply++) {
      const mine = chess.turn() === g.color
      const want = ply >= MIN_PLY && mine && !chess.isGameOver()
      const fenBefore = want ? chess.fen() : null
      const mv = safeMove(chess, g.moves[ply])
      // An unplayable move means the movetext and the replay have diverged;
      // everything after it is meaningless, so the game stops here.
      if (!mv) { truncated++; break }
      if (want) {
        points.push({
          gi, ply, fen: fenBefore, played: g.moves[ply],
          prevMyTo, prevMyFrom, prevOppCapTo, lastTo,
        })
        perGame[gi]++
      }
      if (mine) { prevMyTo = mv.to; prevMyFrom = mv.from }
      else { prevOppCapTo = mv.captured ? mv.to : null }
      lastTo = mv.to
    }
  }
  return { points, perGame, truncated }
}

// ---------------------------------------------------------------------------
// 1. the probe
//
// ~150 positions, two searches each (depth 5 and depth 4, both at width 32),
// which on this engine is about 3.4 seconds. From those two sorted line lists
// every lever falls out with no extra engine work: the rank of the played move
// in each list gives the whole out-of-pool curve, the top-1 rate and the mean
// log rank at once.

function rankOf(lines, played) {
  for (let i = 0; i < lines.length; i++) if (lines[i].uci === played) return i
  return -1
}

function outOfPoolCurve(ranks) {
  const curve = {}
  for (const w of WIDTHS) {
    let miss = 0
    for (const r of ranks) if (r < 0 || r >= w) miss++
    curve[w] = ranks.length ? miss / ranks.length : 1
  }
  return curve
}

function chooseWidth(curve) {
  for (const w of WIDTHS) if (curve[w] <= OUT_OF_POOL_TARGET) return { width: w, capped: false }
  return { width: WIDTHS[WIDTHS.length - 1], capped: true }
}

async function runProbe(engine, points, opts, report, breathe) {
  // PINNING. The weights this build ships are fitted against a pool of a
  // particular depth and width: engine_rank is literally the candidate's index
  // in it, and the softmax is normalised over exactly those candidates. A
  // player that searches at a different depth or width is running the model on
  // a candidate set it was never fitted against, and nothing will complain.
  //
  // share.js carries weights, book and meta - not levers - so a bot sent as a
  // URL arrives with no record of the pool it belongs to. Two ways out, and the
  // caller has to pick one: carry levers.poolDepth / levers.multipv alongside
  // the link, or pin them here to whatever the player is hard-coded to run.
  if (opts.pinLevers) {
    const poolDepth = Math.min(MAX_POOL_DEPTH, Math.max(1, opts.pinLevers.poolDepth ?? DEFAULT_POOL_DEPTH))
    const multipv = Math.max(2, opts.pinLevers.multipv ?? 20)
    return {
      levers: {
        multipv, poolDepth, horizon: null, horizonsTried: HORIZONS.slice(),
        pinned: true,
        outOfPool: { byWidth: null, chosen: null, capped: null, target: OUT_OF_POOL_TARGET },
      },
      probe: { skipped: true, reason: "levers pinned by the caller", positions: 0, ms: 0 },
    }
  }

  const idx = uniformSample(points.length, Math.min(opts.probePositions, points.length))
  const ranks5 = []
  const ranks4 = []
  const t0 = nowMs()
  for (let i = 0; i < idx.length; i++) {
    throwIfAborted(opts.signal)
    report("probe", i, idx.length, "Measuring how far you see...")
    const fen = points[idx[i]].fen
    const played = points[idx[i]].played
    // Sequential on purpose: the engine serialises analyses internally anyway,
    // so firing both at once would only hide the ordering.
    const l5 = await engine.analyse(fen, DEFAULT_POOL_DEPTH, PROBE_WIDTH)
    const l4 = await engine.analyse(fen, MIN_POOL_DEPTH, PROBE_WIDTH)
    ranks5.push(rankOf(l5, played))
    ranks4.push(rankOf(l4, played))
    await breathe()
  }
  report("probe", idx.length, idx.length, "Measuring how far you see...")

  const curve5 = outOfPoolCurve(ranks5)
  const curve4 = outOfPoolCurve(ranks4)
  const pick5 = chooseWidth(curve5)
  const pick4 = chooseWidth(curve4)
  // The probe may LOWER the pool depth, never raise it: depth 4 is cheaper, so
  // if it also loses fewer of the player's moves there is no argument for 5.
  //
  // The comparison is (width it needs, rate it gets there) and NOT the best
  // rate each depth can reach. Both curves hit zero at width 28 on a real
  // account, so "is the minimum lower" compares 0 with 0 and can never pick
  // anything - which is exactly what it did on the first run here, keeping
  // depth 5 while depth 4 was strictly better at every single width.
  const useDepth4 =
    pick4.width < pick5.width ||
    (pick4.width === pick5.width && curve4[pick4.width] < curve5[pick5.width])
  const poolDepth = Math.min(MAX_POOL_DEPTH, useDepth4 ? MIN_POOL_DEPTH : DEFAULT_POOL_DEPTH)
  const curve = useDepth4 ? curve4 : curve5
  const ranks = useDepth4 ? ranks4 : ranks5
  const { width, capped } = useDepth4 ? pick4 : pick5

  // ---- CONFIRM THE WIDTH ----
  //
  // The curve above is read out of ONE width-32 search, which is what makes it
  // free. It is also, measured, WRONG BY A FACTOR OF TWO: MultiPV changes what
  // the search prunes, so a move sitting at rank 12 of a width-32 list is not
  // guaranteed to appear in a width-14 list at all. On this account the curve
  // promised 4.0% at width 14 and the real width-14 searches delivered 8.6% -
  // 4.7 standard errors out, so not sampling noise.
  //
  // Nothing detects that later. The decisions just quietly vanish from the fit.
  // So the chosen width is confirmed with real searches at that exact width,
  // and widened until it actually holds, at a cost of about a second per try.
  //
  // End to end on the same account, after the fix below: confirm said 8.3% at
  // width 14, widened to 20, confirmed 3.3%, and the full scoring run then
  // realized 3.4%. Before it, the lever shipped at width 14 and quietly lost
  // 8.6% of every account's decisions against a stated target of 5%.
  //
  // CONFIRM ON POSITIONS THE PROBE NEVER TOUCHED. This is not fussiness. The
  // engine keeps a 16 MiB transposition table and only ever sees `ucinewgame`
  // once, at boot, so everything it learns about a position survives into the
  // next search of that position. Confirming on the probe's own positions
  // re-searches them with a table already warmed by the width-32 pass, and the
  // narrow search then reproduces the wide search's move list - which is
  // exactly the thing being tested for. Measured: confirming on probe
  // positions reported 1.7% where the untouched positions scored later
  // delivered 8.6%.
  const probeSet = new Set(idx)
  const fresh = []
  for (let i = 0; i < points.length; i++) if (!probeSet.has(i)) fresh.push(i)
  const reusedProbePositions = fresh.length < 20
  const pool = reusedProbePositions ? idx : fresh
  const confirmIdx = uniformSample(pool.length, Math.min(opts.confirmPositions, pool.length))
    .map((i) => pool[i])
  const confirm = async (w) => {
    let miss = 0
    for (const j of confirmIdx) {
      throwIfAborted(opts.signal)
      report("probe", idx.length, idx.length, "Measuring how far you see...")
      const lines = await engine.analyse(points[j].fen, poolDepth, w)
      if (rankOf(lines, points[j].played) < 0) miss++
      await breathe()
    }
    return confirmIdx.length ? miss / confirmIdx.length : 0
  }

  let finalWidth = width
  let confirmed = await confirm(finalWidth)
  const trail = [{ width: finalWidth, predicted: curve[finalWidth], confirmed }]
  for (let attempt = 0; attempt < 2 && confirmed > OUT_OF_POOL_TARGET; attempt++) {
    // Scale the whole predicted curve by how wrong it was here, then take the
    // narrowest width that still clears the target once scaled.
    const ratio = curve[finalWidth] > 0 ? Math.max(1, confirmed / curve[finalWidth]) : 2
    const next =
      WIDTHS.find((w) => w > finalWidth && curve[w] * ratio <= OUT_OF_POOL_TARGET) ??
      WIDTHS.find((w) => w > finalWidth)
    if (next === undefined) break
    finalWidth = next
    confirmed = await confirm(finalWidth)
    trail.push({ width: finalWidth, predicted: curve[finalWidth], confirmed })
  }
  const missedTarget = confirmed > OUT_OF_POOL_TARGET

  const found = ranks.filter((r) => r >= 0)
  const top1 = ranks.length ? ranks.filter((r) => r === 0).length / ranks.length : 0
  const meanLogRank = found.length
    ? found.reduce((s, r) => s + Math.log(r + 1), 0) / found.length
    : null

  return {
    levers: {
      multipv: finalWidth,
      poolDepth,
      // Not chosen here on purpose: all three are computed in the scoring pass
      // and the fit picks one. See runFits().
      horizon: null,
      horizonsTried: HORIZONS.slice(),
      outOfPool: {
        byWidth: curve,
        // what the free width-32 curve promised at this width...
        predicted: curve[finalWidth],
        // ...and what real searches at this width actually delivered
        chosen: confirmed,
        capped: capped || missedTarget,
        target: OUT_OF_POOL_TARGET,
        confirmTrail: trail,
      },
    },
    probe: {
      positions: idx.length,
      width: PROBE_WIDTH,
      ms: Math.round(nowMs() - t0),
      top1,
      meanLogRank,
      outOfPoolByWidthDepth5: curve5,
      outOfPoolByWidthDepth4: curve4,
      widthAtDepth5: pick5.width,
      widthAtDepth4: pick4.width,
      widthFromCurve: width,
      widthAfterConfirm: finalWidth,
      confirmPositions: confirmIdx.length,
      confirmOnFreshPositions: !reusedProbePositions,
      confirmTrail: trail,
      poolDepthChosen: poolDepth,
      poolDepthReason: useDepth4
        ? `depth 4 reaches the ${OUT_OF_POOL_TARGET * 100}% target at width ${pick4.width} ` +
          `where depth 5 needs ${pick5.width}, and depth 4 is cheaper`
        : "depth 5 is the default and depth 4 did not beat it",
    },
  }
}

// ---------------------------------------------------------------------------
// 2. the horizon patch
//
// Only four of the 58 features move when the horizon depth changes, and they
// all fall out of (sh, bestSh) with no board access. Recomputing the whole
// vector three times would triple the board scanning - the one genuinely
// expensive part of the main thread - to recompute 54 identical numbers.
//
// This IS the duplication v9.js's header warns about, so it is not trusted:
// verifyPatch() below recomputes a real decision point through v9.features and
// demands bit-identical output before the run is allowed to continue.

function patchHorizon(out, at, x0base, ctxImbalance, sh, bestSh) {
  let h0 = 0
  let h2 = 0
  if (bestSh !== null && bestSh !== undefined && sh !== null && sh !== undefined) {
    const raw = Math.max(0, (bestSh - sh) / 100)
    h0 = Math.log1p(Math.min(raw, 5))
    h2 = 1 / (1 + Math.exp(-Math.max(-2000, Math.min(2000, sh)) / 150))
  }
  // quiet = not a capture and not a check, read off the two exact indicators
  // the base vector already carries rather than recomputed from the board.
  const quiet = (x0base[IX.capture] || x0base[IX.givesCheck]) ? 0 : 1
  out[at] = h0
  out[at + 1] = h0 * quiet
  out[at + 2] = h2
  out[at + 3] = ctxImbalance * h0
}

function verifyPatch(chess, ucis, ctx, shList, bestSh, bases) {
  for (let i = 0; i < ucis.length; i++) {
    if (!bases[i]) continue
    const truth = features(chess, ucis[i], ctx, shList[i], bestSh, i)
    if (!truth) continue
    const patched = Float64Array.from(bases[i])
    const four = new Float64Array(4)
    patchHorizon(four, 0, bases[i], ctx.imbalance, shList[i], bestSh)
    for (let c = 0; c < 4; c++) patched[HCOLS[c]] = four[c]
    for (let d = 0; d < N_FEATURES; d++) {
      if (patched[d] !== truth[d]) {
        throw new BuildError(
          `horizon patch diverged from v9.features at feature ${d} ` +
          `(${FEATURE_NAMES[d]}): patched ${patched[d]}, v9 ${truth[d]}. ` +
          `The v9 contract changed under build.js - fix patchHorizon().`,
          "PATCH_DIVERGED",
        )
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 3. scoring
//
// Per decision point: one pool search at the chosen depth and width, plus one
// shallow search per horizon. Progress is reported per GAME, because a bar that
// steps once per position is noise at 28 positions a game.

function nowMs() {
  return (typeof performance === "object" && performance && typeof performance.now === "function")
    ? performance.now()
    : Date.now()
}

async function scoreAll(engine, points, games, levers, opts, report, breathe) {
  const K = levers.multipv
  const D = N_FEATURES
  const cap = points.length
  const bytes = cap * K * D * 4
  if (bytes > MAX_DESIGN_BYTES) {
    throw new BuildError(
      `this many games needs a ${Math.round(bytes / 1e6)} MB design matrix; ` +
      `lower maxGames (about ${Math.floor((MAX_DESIGN_BYTES / (28 * K * D * 4)))} games fits)`,
      "TOO_BIG",
    )
  }

  const X = new Float32Array(cap * K * D)
  const M = new Uint8Array(cap * K)
  const y = new Int32Array(cap)
  const g = new Int32Array(cap)
  // Only the four horizon-dependent columns are stored per horizon; the other
  // 54 are shared. Three full copies of X would be three times the memory for
  // 93% identical numbers.
  const hcols = HORIZONS.map(() => new Float32Array(cap * K * 4))
  // Centipawn loss of each candidate against the best in its own pool, for the
  // rating estimate. Clamped per move so one mate-score does not own the mean.
  const slotLoss = new Float32Array(cap * K)
  const playedLoss = new Float64Array(cap)

  let n = 0
  let outOfPool = 0
  let thinPool = 0
  let deadCandidates = 0
  let chosenUnplayable = 0
  let verified = false
  let lastGame = -1
  const t0 = nowMs()

  for (let pi = 0; pi < points.length; pi++) {
    throwIfAborted(opts.signal)
    const p = points[pi]
    if (p.gi !== lastGame) {
      lastGame = p.gi
      report("score", p.gi, games.length,
        `Scoring your moves - game ${p.gi + 1} of ${games.length}`)
    }

    const lines = await engine.analyse(p.fen, levers.poolDepth, K)
    const cands = lines.filter((l) => l && l.uci).slice(0, K)
    // pipeline/analyse.py's rule: a position with one candidate teaches a
    // conditional logit nothing, because there was no choice to model.
    if (cands.length < 2) { thinPool++; continue }
    const ucis = cands.map((c) => c.uci)
    const chosen = ucis.indexOf(p.played)
    if (chosen < 0) { outOfPool++; continue }

    const shMaps = []
    for (const h of HORIZONS) {
      throwIfAborted(opts.signal)
      const hl = await engine.analyse(p.fen, h, K)
      const m = new Map()
      for (const l of hl) if (l && l.uci) m.set(l.uci, l.cp)
      shMaps.push(m)
    }

    const chess = new Chess(p.fen)
    const ctx = makeContext(chess, p.prevMyTo, p.prevMyFrom, p.prevOppCapTo, p.lastTo)

    // The base vector: everything except the four horizon columns, which are
    // left at zero exactly as v9.features leaves them when sh/bestSh are null.
    const bases = new Array(cands.length)
    let chosenOk = true
    for (let i = 0; i < cands.length; i++) {
      const x = features(chess, ucis[i], ctx, null, null, i)
      bases[i] = x
      if (!x) {
        deadCandidates++
        if (i === chosen) chosenOk = false
      }
    }
    // NOT an out-of-pool miss: the move is in the pool, but v9 would not build
    // a vector for it. Engine moves are legal by construction, so this should
    // be zero - it is counted separately precisely so that if it ever is not,
    // it does not hide inside the out-of-pool rate.
    if (!chosenOk) { chosenUnplayable++; continue }

    const horizons = HORIZONS.map((_, hi) => horizonScores(ucis, shMaps[hi]))
    if (!verified) {
      // One real decision point, all three horizons, full recompute against the
      // patch. If v9 ever changes shape this is where the run stops.
      for (let hi = 0; hi < HORIZONS.length; hi++) {
        verifyPatch(chess, ucis, ctx, horizons[hi].sh, horizons[hi].bestSh, bases)
      }
      verified = true
    }

    let bestCp = -Infinity
    for (const c of cands) if (c.cp > bestCp) bestCp = c.cp

    for (let i = 0; i < cands.length; i++) {
      const slot = n * K + i
      if (!bases[i]) continue
      M[slot] = 1
      const base = slot * D
      const x = bases[i]
      for (let d = 0; d < D; d++) X[base + d] = x[d]
      slotLoss[slot] = Math.min(Math.max(bestCp - cands[i].cp, 0), LOSS_CLAMP_CP)
      for (let hi = 0; hi < HORIZONS.length; hi++) {
        patchHorizon(hcols[hi], slot * 4, x, ctx.imbalance, horizons[hi].sh[i], horizons[hi].bestSh)
      }
    }
    y[n] = chosen
    g[n] = p.gi
    playedLoss[n] = slotLoss[n * K + chosen]
    n++
    await breathe()
  }
  report("score", games.length, games.length,
    `Scoring your moves - game ${games.length} of ${games.length}`)

  if (n === 0) {
    throw new BuildError(
      "no usable decision points: none of the moves played were in the engine's pool",
      "NO_EXAMPLES",
    )
  }

  // Trim to what was actually used. A view costs nothing; a copy is only worth
  // it when a lot was dropped and the buffer would otherwise stay resident.
  const used = n / cap
  const cut = (arr, per) =>
    used < 0.8 ? arr.slice(0, n * per) : arr.subarray(0, n * per)

  const gamesWithExamples = new Set()
  for (let i = 0; i < n; i++) gamesWithExamples.add(g[i])

  return {
    X: cut(X, K * D), M: cut(M, K), y: cut(y, 1), g: cut(g, 1),
    hcols: hcols.map((h) => cut(h, K * 4)),
    slotLoss: cut(slotLoss, K),
    playedLoss: cut(playedLoss, 1),
    N: n, K, D,
    outOfPool, thinPool, deadCandidates, chosenUnplayable,
    decisionPoints: points.length,
    gamesWithExamples: gamesWithExamples.size,
    ms: Math.round(nowMs() - t0),
  }
}

// ---------------------------------------------------------------------------
// 4. the fit
//
// Three fits, one per horizon, selected on OUT-OF-FOLD LOG-LIKELIHOOD - the
// same proper scoring rule train.js is already required to use for L2, and for
// the same reason: it is the only score that does not reward a distribution for
// sharpening itself into a deterministic bot.
//
// A NOTE ON THE BRIEF. The brief asked for horizons 1, 2 and 3 to be "extra
// columns, let the regulariser select". That cannot ship. share.js writes
// exactly N_FEATURES = 58 int16 weights and v9.pick() dots exactly 58 features,
// so a 66-column fit produces a weight vector that no player can use - the
// silent invalidation v9.js's header is about. Selecting among three 58-column
// fits on the same criterion keeps the contract and keeps the choice measured.

function applyHorizon(X, hcol, N, K, D) {
  const slots = N * K
  for (let s = 0; s < slots; s++) {
    const base = s * D
    const hb = s * 4
    X[base + HCOLS[0]] = hcol[hb]
    X[base + HCOLS[1]] = hcol[hb + 1]
    X[base + HCOLS[2]] = hcol[hb + 2]
    X[base + HCOLS[3]] = hcol[hb + 3]
  }
}

/**
 * How many L-BFGS iterations to run between yields.
 *
 * Yielding is cheap - measured in this browser, in a BACKGROUND tab, at 0.014ms
 * per scheduler.yield() and 0.127ms per MessageChannel round trip - so the
 * lever is responsiveness, not overhead. train.js's default of 5 is right on a
 * small account, where an iteration is half a millisecond.
 *
 * It stops being right on a large one. An iteration costs O(nonzeros), so at
 * ten thousand decision points it is ~13ms and five of them is a 65ms block -
 * a visible stutter. This keeps a turn near 25ms by yielding more often as the
 * problem grows, and never less often than train.js would.
 */
function yieldEveryFor(examples, pool) {
  const perIterMs = Math.max(0.05, (examples * pool * 11) / 1.3e6)
  return Math.min(5, Math.max(1, Math.round(25 / perIterMs)))
}

async function runFits(scored, opts, report, breathe) {
  const { X, M, y, g, N, K, D } = scored
  const fits = []
  const total = HORIZONS.length
  const yieldEvery = yieldEveryFor(N, K)
  for (let hi = 0; hi < HORIZONS.length; hi++) {
    throwIfAborted(opts.signal)
    report("fit", hi, total, "Fitting your style...")
    applyHorizon(X, scored.hcols[hi], N, K, D)
    const tFit = nowMs()
    let res
    try {
      res = await fitAsync(X, M, y, g, {
        nFeatures: D,
        pool: K,
        featureNames: FEATURE_NAMES,
        yieldEvery,
        ...opts.fit,
        yield: async () => { throwIfAborted(opts.signal); await breathe() },
      })
    } catch (err) {
      if (err && err.name === "AbortError") throw err
      throw new BuildError(`the fit failed at horizon ${HORIZONS[hi]}: ${err.message}`, "FIT_FAILED")
    }
    fits.push({ horizon: HORIZONS[hi], ms: Math.round(nowMs() - tFit), ...res })
  }
  report("fit", total, total, "Fitting your style...")

  // Out-of-fold where there were enough games to split folds; in-sample is the
  // documented fallback and is FLAGGED, because it is not a fair comparison -
  // it just has to break a tie between three fits of identical shape.
  const scoreOf = (f) => (f.outOfFoldLogLik === null ? f.inSampleLogLik : f.outOfFoldLogLik)
  const outOfFold = fits.every((f) => f.outOfFoldLogLik !== null)
  let best = fits[0]
  for (const f of fits) if (scoreOf(f) > scoreOf(best)) best = f

  // HOW MUCH THIS SWEEP IS WORTH, measured rather than assumed.
  //
  // On a 12-game account the three horizons scored -1.6992, -1.7029 and
  // -1.6959: a spread of 0.007 nats, well under half a percent, and two builds
  // of the SAME account picked different winners (1, then 3). So on a small
  // account the horizon sweep costs two extra shallow searches per decision and
  // two extra fits to make a choice that is inside its own noise.
  //
  // The spread is reported so a caller can see when that is happening. The
  // 0.01-nat line is a heuristic, not a test - a real test needs the per-fold
  // log-likelihoods, which train.js averages away before returning.
  const lls = fits.map(scoreOf)
  const spread = Math.max(...lls) - Math.min(...lls)
  // leave X holding the winner's columns, so anything downstream that reads it
  // is reading the matrix the shipped weights were fitted against
  applyHorizon(X, scored.hcols[fits.indexOf(best)], N, K, D)
  return {
    best,
    outOfFold,
    spread,
    decisive: spread >= 0.01,
    curve: fits.map((f) => ({
      horizon: f.horizon, logLik: scoreOf(f), l2: f.l2, top1: f.top1,
      behaviourMismatch: f.behaviourMismatch, dropped: f.droppedFeatures.length,
      ms: f.ms, iterations: f.iterations, nonzeros: f.nonzeros,
    })),
  }
}

// ---------------------------------------------------------------------------
// 5. the book
//
// pipeline/build.py's construction, move for move: the first 30 ply, the
// owner's own moves only, keyed on the first three FEN fields (the en-passant
// field is dropped because chess.js and python-chess disagree about when to
// write it, and a book that misses its own positions is worse than no book).

function buildBook(games, openings, opts) {
  const map = Object.create(null)
  const white = new Map()
  const black = new Map()
  let noResult = 0
  let asWhite = 0
  let asBlack = 0

  for (const game of games) {
    const chess = new Chess()
    const res = RESULT_CODE[game.result] || null
    if (!res) noResult++
    if (game.color === "w") asWhite++
    else asBlack++
    let opening = null

    for (let ply = 0; ply < game.moves.length && ply < MAX_BOOK_PLY; ply++) {
      const mine = chess.turn() === game.color
      const keyBefore = mine ? bookKey(chess.fen()) : null
      const mv = safeMove(chess, game.moves[ply])
      if (!mv) break
      if (mine) {
        const uci = game.moves[ply]
        let entry = map[keyBefore] && map[keyBefore][uci]
        if (!entry) {
          entry = { san: mv.san, n: 0, w: 0, d: 0, l: 0 }
          if (!map[keyBefore]) map[keyBefore] = Object.create(null)
          map[keyBefore][uci] = entry
        }
        entry.n++
        // A game whose result the API did not give still tells the book what
        // was PLAYED, which is all the trie reads. build.py drops such games
        // outright because its PGN source always carries a Result header.
        if (res) entry[res]++
      }
      if (openings) {
        const nm = openings[bookKey(chess.fen())]
        // the LAST name seen wins: the deepest match is the most specific one
        if (nm) opening = nm
      }
    }
    if (opening) {
      const fam = opening.split(":")[0].trim()
      const counter = game.color === "w" ? white : black
      counter.set(fam, (counter.get(fam) || 0) + 1)
    }
  }

  // minCount 2 is share.js's default and is right for a player with hundreds of
  // games. On a small account it deletes the whole book, so a thin account gets
  // 1 - every line they played once. Stated in levers.book.minCount.
  const minCount = opts.bookMinCount ?? (games.length >= 40 ? 2 : 1)
  let trie = null
  try {
    trie = buildTrie(map, { Chess, minCount, maxDepth: MAX_BOOK_PLY })
  } catch {
    trie = null
  }

  return {
    map, trie, minCount, noResult, asWhite, asBlack,
    families: { w: white, b: black },
    positions: Object.keys(map).length,
    stats: trie ? trieStats(trie) : null,
  }
}

function favouriteOpening(book) {
  // "whichever colour has more games" - the same rule build.py's panel uses,
  // so a player who is mostly Black is described by their Black repertoire.
  const colour = book.asWhite >= book.asBlack ? "w" : "b"
  const counter = book.families[colour]
  if (!counter.size) {
    const other = book.families[colour === "w" ? "b" : "w"]
    if (!other.size) return null
    return topFamily(other, colour === "w" ? "b" : "w", colour === "w" ? book.asBlack : book.asWhite)
  }
  return topFamily(counter, colour, colour === "w" ? book.asWhite : book.asBlack)
}

function topFamily(counter, colour, colourGames) {
  let name = null
  let n = 0
  for (const [k, v] of counter) if (v > n || (v === n && k < name)) { name = k; n = v }
  return {
    name, colour, games: n,
    ofGames: colourGames,
    share: colourGames ? n / colourGames : null,
  }
}

// ---------------------------------------------------------------------------
// 6. the rating estimate
//
// READ THIS BEFORE QUOTING THE NUMBER.
//
// Two measured inputs, both from the scoring pass, both at the POOL DEPTH the
// levers chose - which is 5, not the depth 20 every published ACPL table is
// computed at. That alone makes this a different quantity from "ACPL" as
// normally used, so published tables cannot be applied directly.
//
//   acpl  the fitted policy's EXPECTED centipawn loss: sum over candidates of
//         p(move) * loss(move), averaged over decisions. Not the player's own
//         loss - the bot samples, so it is slightly worse than the human it
//         copies, and that gap is measured rather than assumed. Both numbers
//         are returned.
//   top1  the fitted policy's probability of playing the engine's first choice.
//
// THE CALIBRATION IS ONE PLAYER. Andrew Burke: 9,431 decision points over 359
// rapid games, measured depth-5 ACPL 66.8 and depth-5 engine agreement 34.9%,
// against a chess.com rapid rating of 1002. That fixes the curve's height. Its
// SLOPE is not measured - it is set from the second, assumed anchor below.
//
// So: the number is a scale-anchored guess with one real point on it. It is
// quoted on the chess.com rapid scale (which runs a few hundred points above
// Lichess and FIDE), rounded to 25, clamped to [400, 2600], and returned with a
// +-250 band that is the honest width for a single-anchor fit. Do not present
// it without the band, and do not present more than three significant figures.

export const RATING_CALIBRATION = {
  scale: "chess.com rapid",
  anchor: {
    who: "burkeley",
    rating: 1002,
    acpl: 66.8,
    top1: 0.3488,
    basis: "9,431 decisions / 359 rapid games at pool depth 5, MultiPV 20",
    source: "pipeline/raw-cache.json and api.chess.com /pub/player/burkeley/stats",
  },
  // ASSUMED, not measured: a player with depth-5 ACPL 20 and depth-5 agreement
  // 55% is taken to be about 2200 chess.com rapid. Everything about the slope
  // rests on this one guess, which is why the band is +-250.
  assumed: { acpl: 20, top1: 0.55, rating: 2200 },
  acplWeight: 0.65,
  band: 250,
  min: 400,
  max: 2600,
}

const logit = (p) => Math.log(p / (1 - p))

/**
 * Map measured move quality to a rating. Exported so it can be argued with.
 * @returns {{rating, low, high, fromAcpl, fromTop1, scale, confidence, note}}
 */
export function estimateRating({ acpl, top1 }) {
  const C = RATING_CALIBRATION
  const a0 = C.anchor
  const a1 = C.assumed
  const acplSlope = (a1.rating - a0.rating) / Math.log(a0.acpl / a1.acpl)
  const top1Slope = (a1.rating - a0.rating) / (logit(a1.top1) - logit(a0.top1))

  const safeAcpl = Math.max(1, Number.isFinite(acpl) ? acpl : a0.acpl)
  const safeTop1 = Math.min(0.95, Math.max(0.02, Number.isFinite(top1) ? top1 : a0.top1))

  const fromAcpl = a0.rating + acplSlope * Math.log(a0.acpl / safeAcpl)
  const fromTop1 = a0.rating + top1Slope * (logit(safeTop1) - logit(a0.top1))
  const blended = C.acplWeight * fromAcpl + (1 - C.acplWeight) * fromTop1
  const clamp = (r) => Math.min(C.max, Math.max(C.min, r))
  const round25 = (r) => Math.round(r / 25) * 25

  const rating = round25(clamp(blended))
  return {
    rating,
    low: round25(clamp(blended - C.band)),
    high: round25(clamp(blended + C.band)),
    fromAcpl: round25(clamp(fromAcpl)),
    fromTop1: round25(clamp(fromTop1)),
    scale: C.scale,
    confidence: "low",
    note:
      `+-${C.band} at best. One real calibration point (${a0.who} at ${a0.rating}); ` +
      `the slope is assumed, not measured. Centipawn loss here is measured at ` +
      `pool depth, not at analysis depth, so published ACPL tables do not apply.`,
  }
}

/**
 * The fitted policy's own move quality: what the BOT is expected to lose per
 * move, and how often it picks the engine's first choice. This is the fit
 * playing, not the human - it is the thing being rated.
 */
function policyQuality(scored, weights, temp) {
  const { X, M, slotLoss, playedLoss, N, K, D } = scored
  const z = new Float64Array(K)
  const t = temp > 0 ? temp : 1
  let botLoss = 0
  let botTop1 = 0
  let humanLoss = 0
  let humanTop1 = 0
  for (let n = 0; n < N; n++) {
    let zmax = -Infinity
    for (let k = 0; k < K; k++) {
      if (!M[n * K + k]) { z[k] = -Infinity; continue }
      const base = (n * K + k) * D
      let acc = 0
      for (let d = 0; d < D; d++) acc += weights[d] * X[base + d]
      z[k] = acc
      if (acc > zmax) zmax = acc
    }
    let sum = 0
    for (let k = 0; k < K; k++) {
      if (!M[n * K + k]) continue
      const e = Math.exp((z[k] - zmax) / t)
      z[k] = e
      sum += e
    }
    for (let k = 0; k < K; k++) {
      if (!M[n * K + k]) continue
      const p = z[k] / sum
      botLoss += p * slotLoss[n * K + k]
      if (k === 0) botTop1 += p
    }
    humanLoss += playedLoss[n]
    if (M[n * K] && scored.y[n] === 0) humanTop1++
  }
  return {
    botAcpl: botLoss / N,
    botTop1: botTop1 / N,
    humanAcpl: humanLoss / N,
    humanTop1: humanTop1 / N,
  }
}

// ---------------------------------------------------------------------------
// openings.json

async function loadOpenings(url, signal) {
  if (url === null) return null
  const href = url || new URL("../openings.json", import.meta.url).href
  try {
    const res = await fetch(href, { signal })
    if (!res.ok) return null
    return await res.json()
  } catch (err) {
    if (err && err.name === "AbortError") throw err
    // A missing ECO table costs the favourite-opening line and nothing else.
    return null
  }
}

// ---------------------------------------------------------------------------
// the public API

function normaliseOpts(opts) {
  const o = opts || {}
  const accounts = (o.accounts || []).filter((a) => a && a.site && a.username)
  if (!accounts.length) throw new BuildError("no accounts given", "NO_ACCOUNTS")
  return {
    accounts,
    extraGames: Array.isArray(o.extraGames) ? o.extraGames : [],
    speeds: o.speeds && o.speeds.length ? o.speeds : ["rapid", "blitz"],
    ratedOnly: o.ratedOnly !== false,
    includeBots: !!o.includeBots,
    maxGames: Math.max(1, Math.floor(o.maxGames ?? 400)),
    probePositions: Math.max(10, Math.floor(o.probePositions ?? PROBE_POSITIONS)),
    confirmPositions: Math.max(10, Math.floor(o.confirmPositions ?? CONFIRM_POSITIONS)),
    signal: o.signal,
    engine: o.engine || null,
    engineOpts: o.engineOpts || {},
    openingsUrl: o.openingsUrl,
    bookMinCount: o.bookMinCount,
    // {poolDepth, multipv} to skip the probe and fit against a fixed pool -
    // see the note at the top of runProbe()
    pinLevers: o.pinLevers || null,
    temp: o.temp ?? 1.0,
    fit: o.fit || {},
  }
}

/**
 * Turn a username into a bot.
 *
 * @param {object} opts
 *   accounts     [{site:'chesscom'|'lichess', username}]
 *   speeds       ['rapid','blitz',...]  default ['rapid','blitz']
 *   ratedOnly    default true
 *   includeBots  default false
 *   maxGames     default 400
 *   signal       AbortSignal - honoured between every engine call, inside the
 *                download and inside the fit
 *   engine       an already-booted engine.js engine; one is created and quit
 *                here if absent
 * @param {function} onProgress ({phase, done, total, label}) where phase is
 *   'download' | 'probe' | 'score' | 'fit' and label is the whole user-facing
 *   line.
 * @returns {Promise<{weights, book, levers, stats}>} plus bookMap, fit and
 *   timing as extras. `book` is a share.js trie, ready for share.encode().
 */
export async function buildBot(opts, onProgress) {
  const o = normaliseOpts(opts)
  const report = reporter(onProgress)
  const breathe = makeYielder()
  const t0 = nowMs()

  // ---- 1. download ----
  report("download", 0, o.maxGames, `Downloading your games - 0 of ${o.maxGames}`)
  let base = 0
  const games = await fetchGames(
    {
      accounts: o.accounts,
      speeds: o.speeds,
      ratedOnly: o.ratedOnly,
      includeBots: o.includeBots,
      max: o.maxGames,
      signal: o.signal,
    },
    (p) => {
      if (p.phase !== "download") return
      // fetchGames reports twice: from inside an account (carries `scanned`)
      // and at each account boundary (does not). Only the boundary count is
      // cumulative, so the inner one is offset by it.
      const done = "scanned" in p ? base + p.done : (base = p.done)
      report("download", Math.min(done, o.maxGames), o.maxGames,
        `Downloading your games - ${Math.min(done, o.maxGames)} of ${o.maxGames}`)
    },
  )
  // Games the player uploaded as PGN. They are merged in rather than fetched
  // because no API will ever hand them over: chess.com publishes no games
  // against its own bots. Deduped by move list and side, so re-dropping a file
  // that overlaps the archive cannot double-weight those games in the fit.
  if (o.extraGames && o.extraGames.length) {
    const seen = new Set(games.map(gameKey))
    let added = 0
    for (const g of o.extraGames) {
      const k = gameKey(g)
      if (seen.has(k)) continue
      seen.add(k)
      games.push(g)
      added++
    }
    // newest-first is the order everything downstream assumes, and the uploads
    // were appended, not merged in date order
    if (added) games.sort((a, b) => (b.endTime || 0) - (a.endTime || 0))
    if (games.length > o.maxGames) games.length = o.maxGames
    report("download", Math.min(games.length, o.maxGames), o.maxGames,
      `Downloading your games - ${games.length} of ${o.maxGames}`)
  }

  const tDownload = nowMs() - t0
  if (!games.length) {
    throw new BuildError(
      "no games came back - check the username, or widen the speeds",
      "NO_GAMES",
    )
  }

  const { points, truncated } = planDecisions(games)
  if (!points.length) {
    throw new BuildError(
      `${games.length} games, but none of them had a move after ply ${MIN_PLY} to learn from`,
      "NO_DECISIONS",
    )
  }

  const engine = o.engine || (await createEngine(o.engineOpts))
  const ownEngine = !o.engine

  try {
    // ---- 2. probe ----
    const tProbe0 = nowMs()
    const { levers, probe } = await runProbe(engine, points, o, report, breathe)
    const tProbe = nowMs() - tProbe0

    // ---- 3. score ----
    const scored = await scoreAll(engine, points, games, levers, o, report, breathe)
    // The probe predicts the out-of-pool rate by reading rank >= W out of ONE
    // width-32 search. Scoring then runs a real width-W search, and MultiPV
    // changes what the search prunes, so the two are not the same measurement.
    // Measured on a 12-game account: 4.7% predicted, 7.4% realized. Both are
    // recorded here so the gap is visible rather than inferred.
    levers.outOfPool.realized = scored.outOfPool / scored.decisionPoints
    levers.outOfPool.realizedNote = levers.pinned ? "levers pinned; no probe to compare against" :
      "predicted from a width-32 probe search; realized by the width-" +
      `${levers.multipv} searches scoring actually ran`

    // ---- 4. fit ----
    const tFit0 = nowMs()
    const { best, outOfFold, curve, spread, decisive } = await runFits(scored, o, report, breathe)
    const tFit = nowMs() - tFit0
    levers.horizon = best.horizon
    levers.horizonChosenBy = outOfFold ? "out-of-fold log-likelihood" : "in-sample log-likelihood (too few games to fold)"
    levers.horizonCurve = curve
    levers.horizonSpread = spread
    levers.horizonDecisive = decisive
    if (!decisive) {
      levers.horizonNote =
        `the three horizons scored within ${spread.toFixed(4)} nats of each other - ` +
        `this choice is inside its own noise, and horizon ${best.horizon} is only the ` +
        `winner of a coin toss`
    }
    levers.l2 = best.l2
    levers.temp = o.temp

    // ---- 5. book ----
    // Book and stats used to run silently. They are quick, but loadOpenings
    // goes to the network, so a page that shows its steps would sit on a
    // finished "Fitting your style" with nothing moving. One phase covers
    // both: it is the last thing between here and a playable bot.
    report("finish", 1, 3, "Building your opening book...")
    const openings = await loadOpenings(o.openingsUrl, o.signal)
    const book = buildBook(games, openings, o)
    levers.book = { minCount: book.minCount, maxPly: MAX_BOOK_PLY }

    // ---- 6 & 7. stats ----
    report("finish", 2, 3, "Working out how you play...")
    const quality = policyQuality(scored, best.weights, o.temp)
    const rating = estimateRating({ acpl: quality.botAcpl, top1: quality.botTop1 })

    const stats = {
      gamesUsed: best.games,
      decisions: best.examples,
      top1: best.top1 === null ? best.inSampleTop1 : best.top1,
      botRating: rating.rating,
      favouriteOpening: favouriteOpening(book),
      // everything below is extra, and is what makes the four above readable
      top1IsOutOfFold: best.top1 !== null,
      rating,
      quality,
      gamesDownloaded: games.length,
      gamesTruncated: truncated,
      decisionPoints: scored.decisionPoints,
      outOfPool: scored.outOfPool,
      outOfPoolRate: scored.outOfPool / scored.decisionPoints,
      thinPool: scored.thinPool,
      chosenUnplayable: scored.chosenUnplayable,
      deadCandidates: scored.deadCandidates,
      bookPositions: book.positions,
      bookTrie: book.stats,
      bookGamesWithoutResult: book.noResult,
      asWhite: book.asWhite,
      asBlack: book.asBlack,
      outOfFoldLogLik: best.outOfFoldLogLik,
      behaviourMismatch: best.behaviourMismatch,
      droppedFeatures: best.droppedFeatures.map((d) => FEATURE_NAMES[d]),
      warnings: games.warnings || [],
    }

    return {
      weights: best.weights,
      book: book.trie,
      levers,
      stats,
      // extras
      bookMap: book.map,
      fit: best,
      timing: {
        downloadMs: Math.round(tDownload),
        probeMs: Math.round(tProbe),
        scoreMs: scored.ms,
        fitMs: Math.round(tFit),
        totalMs: Math.round(nowMs() - t0),
      },
      probe,
    }
  } finally {
    if (ownEngine) {
      try { await engine.quit() } catch { /* nothing left to save */ }
    }
  }
}

// ---------------------------------------------------------------------------
// estimateTime
//
// Every constant below is measured off a real 12-game build of a real account
// (82.7 seconds end to end, 408 decision points), not estimated. The engine
// half is re-measured per machine through calibrate(); the CPU half is not, and
// a slow phone will beat these numbers in the wrong direction.
//
// THE HEADLINE, because it will decide what maxGames is allowed to be: THE FIT
// DOMINATES, and it grows with the number of decisions. 400 games is roughly
// half an hour on this laptop, of which about 24 minutes is three L-BFGS runs
// over an L2 grid. If that is too slow the lever is the horizon sweep (three
// fits instead of one) long before it is anything the engine does.

// pipeline/games-cache.json: 10,104 decision points over 359 games.
const DECISIONS_PER_GAME = 28.1
// Both CPU terms are per CANDIDATE SLOT, not per decision, because both scale
// with the pool width as well as the decision count. Measured across two real
// builds of the same account at two different widths:
//   width 14, 372 decisions: fit 47.8s -> 9.19 ms/slot
//   width 20, 393 decisions: fit 69.7s -> 8.87 ms/slot
// Two independent measurements 4% apart, so the shape is right.
const REFERENCE_WIDTH = 20
// Feature extraction, one chess.js per decision, and the design-matrix writes.
const CPU_MS_PER_SLOT = 0.55
// All THREE fits over the full L2 grid. An L-BFGS iteration is O(nonzeros), and
// the iteration COUNT creeps up with sample size too, so this is a floor.
const FIT_MS_PER_SLOT = 8.9
// chess.com pays one HTTP round trip per month of archive whatever the game
// count, so the cost is mostly fixed. Measured 1.2-1.9s for 12 games.
const DOWNLOAD_BASE_MS = 1500
const DOWNLOAD_MS_PER_GAME = 120

/**
 * Seconds a build of `gameCount` games is expected to take.
 * @param {object} engine a booted engine.js engine, or null for defaults
 * @param {number} gameCount
 * @param {number} [width] pool width, if a probe has already chosen one
 * @returns {Promise<number>} seconds
 */
export async function estimateTime(engine, gameCount, width) {
  const detail = await estimateTimeDetail(engine, gameCount, width)
  return detail.seconds
}

/** The same estimate with its parts showing. */
export async function estimateTimeDetail(engine, gameCount, width = REFERENCE_WIDTH) {
  const n = Math.max(0, Math.floor(gameCount || 0))
  const decisions = n * DECISIONS_PER_GAME
  let ms = { 1: 3, 2: 4, 3: 6, 4: 9, 5: 11 }
  let measured = false
  if (engine && typeof engine.calibrate === "function") {
    const out = {}
    for (const d of [1, 2, 3, 4, 5]) {
      const c = await engine.calibrate({ depth: d, multipv: 20 })
      out[d] = c.msPerPosition
    }
    ms = out
    measured = true
  }
  // 150 positions x 2 depths for the curve, plus up to 3 x 60 confirming the
  // width it picked.
  const probeMs = PROBE_POSITIONS * (ms[5] + ms[4]) + 2 * CONFIRM_POSITIONS * ms[5]
  const searchMs = decisions * (ms[5] + ms[1] + ms[2] + ms[3])
  const slots = decisions * width
  const cpuMs = slots * CPU_MS_PER_SLOT
  const fitMs = slots * FIT_MS_PER_SLOT
  const downloadMs = n ? DOWNLOAD_BASE_MS + n * DOWNLOAD_MS_PER_GAME : 0
  const totalMs = probeMs + searchMs + cpuMs + fitMs + downloadMs
  return {
    seconds: Math.round(totalMs / 1000),
    measured,
    msPerPosition: ms,
    decisions: Math.round(decisions),
    parts: {
      downloadSec: Math.round(downloadMs / 1000),
      probeSec: Math.round(probeMs / 1000),
      scoreSec: Math.round((searchMs + cpuMs) / 1000),
      fitSec: Math.round(fitMs / 1000),
    },
    // The fit is the part that decides whether a game count is usable at all.
    dominatedBy: fitMs > searchMs + cpuMs ? "fit" : "engine",
    assumes: {
      decisionsPerGame: DECISIONS_PER_GAME,
      poolWidth: width === REFERENCE_WIDTH
        ? `${width} (the reference width; the probe may pick another)`
        : `${width} (as chosen by the probe; the reference is ${REFERENCE_WIDTH})`,
      cpuMsPerCandidateSlot: CPU_MS_PER_SLOT,
      fitMsPerCandidateSlot: `${FIT_MS_PER_SLOT} (all three horizon fits)`,
      downloadMs: `${DOWNLOAD_BASE_MS} + ${DOWNLOAD_MS_PER_GAME}/game`,
      searchesPerDecision: "1 pool + 3 horizons",
      calibratedOn: "a 2026 laptop; the CPU terms do not scale to a phone",
    },
  }
}

export default { buildBot, estimateTime, estimateTimeDetail, estimateRating, BuildError }
