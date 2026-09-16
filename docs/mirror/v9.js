// The v9 feature contract, as a standalone ES module.
//
// This is the THIRD copy of one piece of arithmetic. The other two are
// pipeline/features_v9.py (which trains the model) and the v9 block in
// docs/app.js (which Burke Bot plays with); pipeline/test_parity.py already
// proves those two agree to 1e-6 by brace-extracting the functions out of
// app.js. This file must agree with both, for the same reason: a changed
// feature silently invalidates whatever weights were fitted against it and
// NOTHING in the training metrics notices - the loss is computed on the python
// side and looks perfect either way. The bot just plays a different game.
//
// The only thing deliberately different here is the plumbing. app.js's v9
// functions read a MODULE-SCOPED `chess` (and dig the previous move out of
// chess.history()), so a second page cannot drive them. Every function below
// takes the chess.js instance as its first argument, and makeContext() takes
// the previous-move facts as arguments instead of reconstructing them - which
// is also what features_v9.py's Context does.
//
// Feature order IS the contract. Never renumber; append only.
//
// No imports on purpose: the caller owns the chess.js instance, so this module
// has no dependencies at all.

export const N_FEATURES = 58

// P N B R Q K - the king is 0 as a VICTIM (it is never captured)...
const PIECE_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }
// ...but 100 as an ATTACKER. v8 used 0 here, which made every defended piece
// standing next to the enemy king look hanging (the king was the "cheapest
// attacker" at value 0). Indices 35-42 all depend on this.
const ATTACKER_VAL = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 100 }
const ORDER = ["p", "n", "b", "r", "q", "k"]
const HOME = {
  w: { n: ["b1", "g1"], b: ["c1", "f1"], r: ["a1", "h1"] },
  b: { n: ["b8", "g8"], b: ["c8", "f8"], r: ["a8", "h8"] },
}
const FIANCHETTO = { w: ["b2", "g2"], b: ["b7", "g7"] }

const sqFile = (s) => s.charCodeAt(0) - 97
const sqRank = (s) => s.charCodeAt(1) - 49
const mkSq = (f, r) => String.fromCharCode(97 + f) + (r + 1)
const cheb = (a, b) =>
  Math.max(Math.abs(sqFile(a) - sqFile(b)), Math.abs(sqRank(a) - sqRank(b)))

function tryMove(chess, args) {
  try { return chess.move(args) } catch { return null }
}

// attacked, and not adequately defended (a king attacker counts 100, so a
// defended piece beside the enemy king is NOT hanging)
function enPrise(chess, sq, owner, opp) {
  const atk = chess.attackers(sq, opp)
  if (!atk.length) return false
  if (!chess.isAttacked(sq, owner)) return true
  const pc = chess.get(sq)
  if (!pc) return false
  let cheapest = 999
  for (const a of atk) {
    const ap = chess.get(a)
    if (ap) cheapest = Math.min(cheapest, ATTACKER_VAL[ap.type])
  }
  return cheapest < PIECE_VAL[pc.type]
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

/**
 * Everything true of the POSITION rather than of one candidate, computed once
 * per decision point. The per-candidate loop then does almost no board
 * scanning.
 *
 * The previous-move facts are arguments, not history lookups, so a page that
 * feeds positions in by FEN can pass null for all four and get the same
 * "no history" context features_v9.py builds from Context(board, None, None,
 * None). Where app.js derives them from chess.history():
 *   prevMyTo     - the square I moved to last turn        (myLast.to)
 *   prevMyFrom   - the square I moved from last turn      (myLast.from)
 *   prevOppCapTo - where the opponent just CAPTURED, else null
 *   lastTo       - the square the opponent just moved to  (oppLast.to)
 * prevMyFrom is carried for callers but no v9 feature reads it.
 */
export function makeContext(chess, prevMyTo, prevMyFrom, prevOppCapTo, lastTo) {
  const me = chess.turn()
  const opp = me === "w" ? "b" : "w"
  const castling = chess.fen().split(" ")[2]

  const cells = []
  for (const row of chess.board()) for (const c of row) if (c) cells.push(c)

  let pieceCount = 0, npm = 0, mine = 0, theirs = 0
  let myKing = null, enemyKing = null
  const ownPawnRanks = [[], [], [], [], [], [], [], []]
  const enemyPawnRanks = [[], [], [], [], [], [], [], []]
  for (const c of cells) {
    pieceCount++
    const v = PIECE_VAL[c.type]
    if (c.type !== "p" && c.type !== "k") npm += v
    if (c.color === me) mine += v; else theirs += v
    if (c.type === "k") { if (c.color === me) myKing = c.square; else enemyKing = c.square }
    if (c.type === "p") {
      (c.color === me ? ownPawnRanks : enemyPawnRanks)[sqFile(c.square)].push(sqRank(c.square))
    }
  }

  // index 47: distinct enemy pieces bearing on my king's neighbourhood
  // index 48: the six squares in front of a castled king
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

  // index 50: the enemy king's zone, and who already bears on it
  let zone = [], zoneBefore = 0
  if (enemyKing) {
    zone = neighbours(enemyKing)
    const seen = new Set()
    for (const sq of zone) for (const a of chess.attackers(sq, me)) seen.add(a)
    zoneBefore = seen.size
  }

  // index 40: pieces of mine the opponent's LAST move put en prise
  const fresh = []
  if (lastTo !== null && lastTo !== undefined) {
    for (const c of cells) {
      if (c.color !== me || !"nbrq".includes(c.type)) continue
      if (chess.attackers(c.square, opp).includes(lastTo) && enPrise(chess, c.square, me, opp)) {
        fresh.push(c.square)
      }
    }
  }

  // index 13: is a cheapest-recapture available, and is it forced?
  const capTo = (prevOppCapTo === undefined) ? null : prevOppCapTo
  let recapN = 0, recapMin = null
  if (capTo !== null) {
    const vals = chess.moves({ verbose: true })
                      .filter(m => m.to === capTo && m.captured)
                      .map(m => PIECE_VAL[m.piece])
    recapN = vals.length
    if (vals.length) recapMin = Math.min(...vals)
  }

  return {
    me, opp,
    prevMyTo: (prevMyTo === undefined) ? null : prevMyTo,
    prevMyFrom: (prevMyFrom === undefined) ? null : prevMyFrom,
    prevOppCapTo: capTo,
    lastMoveTo: (lastTo === undefined) ? null : lastTo,
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

/**
 * Impute missing depth-2 scores, then take the best - IN THAT ORDER.
 *
 * A candidate the depth-2 search did not report is treated as looking at least
 * as bad as the worst move it did report. Imputing after taking bestSh
 * diverges on the 3.3% of candidates that need it.
 *
 * shByUci may be a plain object or a Map; a missing or null entry is imputed.
 */
export function horizonScores(ucis, shByUci) {
  const look = (shByUci instanceof Map) ? (u) => shByUci.get(u) : (u) => shByUci[u]
  const present = ucis.map(look).filter(v => v !== undefined && v !== null)
  const floor = present.length ? Math.min(...present) : null
  const out = ucis.map(u => {
    const v = look(u)
    return (v === undefined || v === null) ? floor : v
  })
  const usable = out.filter(v => v !== null)
  return { sh: out, bestSh: usable.length ? Math.max(...usable) : null }
}

// a quiet edge-pawn push with no job. Called with the move ALREADY on the
// board; `from`/`to` are its squares. v8 carried this as a hard play-time
// guard; v9 carries it as a feature so the model fits its own measured rate
// (Andrew kicks a bishop already on the square 17.1% of the time and pushes
// with no purpose at all 0.7%) instead of a hand-set rule.
function aimlessEdgePawn(chess, me, opp, from, to) {
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

/**
 * The 58-vector for one candidate. `chess` is at the PRE-move position; the
 * move is pushed and popped internally, so the instance is left exactly as it
 * was found.
 *
 * Returns a Float64Array(58), or null if `uci` is not legal here.
 */
export function features(chess, uci, ctx, sh, bestSh, rank) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4)
  const me = ctx.me, opp = ctx.opp
  const x = new Float64Array(N_FEATURES)

  const pre = chess.get(from)
  if (!pre) return null
  const mover = pre.type
  const moverVal = PIECE_VAL[mover]
  const toAttacked = chess.isAttacked(to, opp)
  const fromAttacked = chess.isAttacked(from, opp)
  const danger = enPrise(chess, from, me, opp)
  const enemyPawnsAttackTo = chess.attackers(to, opp).some(a => {
    const p = chess.get(a); return p && p.type === "p"
  })

  const played = tryMove(chess, { from, to, promotion: uci.slice(4) || undefined })
  if (!played) return null
  const isCapture = !!played.captured
  const isCastle = played.flags.includes("k") || played.flags.includes("q")
  const givesCheck = chess.inCheck()
  const tf = sqFile(to), tr = sqRank(to), ff = sqFile(from), fr = sqRank(from)
  const fwd = me === "w" ? 1 : -1

  // post-move facts, gathered while the move is on the board
  const landedHanging = enPrise(chess, to, me, opp)
  let count = landedHanging ? 1 : 0
  let others = false
  let ignoresFresh = false
  let bestAny = 0, bestLoose = 0
  const postCells = []
  for (const row of chess.board()) for (const c of row) if (c) postCells.push(c)
  for (const c of postCells) {
    if (c.color === me && "nbrq".includes(c.type) && c.square !== to) {
      if (enPrise(chess, c.square, me, opp)) { others = true; count++ }
    }
    if (c.color === opp && c.type !== "k" && chess.attackers(c.square, me).includes(to)) {
      const v = PIECE_VAL[c.type]
      if (v > bestAny) bestAny = v
      if ((!chess.isAttacked(c.square, opp) || v > moverVal) && v > bestLoose) bestLoose = v
    }
  }
  for (const sq of ctx.fresh) {
    if (sq === from) continue
    const p = chess.get(sq)
    if (p && p.color === me && enPrise(chess, sq, me, opp)) { ignoresFresh = true; break }
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
    aimlessEdge = aimlessEdgePawn(chess, me, opp, from, to) ? 1 : 0
  }
  // 55/56: castling is not automatically safe. The model fits a big positive
  // weight to `castle` because people castle a lot, and with no notion of what
  // it is castling INTO it will happily walk the king next to an enemy rook.
  // These two ask what the king's new home looks like.
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

  // ---- 0-2: perception. The player's own horizon, never the deep score. ----
  if (bestSh !== null && bestSh !== undefined && sh !== null && sh !== undefined) {
    const raw = Math.max(0, (bestSh - sh) / 100)
    x[0] = Math.log1p(Math.min(raw, 5))       // clamp BEFORE the log
    x[2] = 1 / (1 + Math.exp(-Math.max(-2000, Math.min(2000, sh)) / 150))
  }
  const quiet = (isCapture || givesCheck) ? 0 : 1
  x[1] = x[0] * quiet

  // ---- 3-13: the engine's ordering, and the capture family ----
  x[3] = rank / 4
  const forcing = (isCapture || givesCheck || played.promotion) ? 1 : 0
  x[4] = forcing
  x[5] = isCapture ? 1 : 0
  x[6] = givesCheck ? 1 : 0
  x[7] = played.promotion ? 1 : 0
  x[8] = isCastle ? 1 : 0
  if (isCapture) x[9] = PIECE_VAL[played.captured] / 9
  x[10] = (isCapture && toAttacked) ? 1 : 0
  x[11] = (isCapture && toAttacked && PIECE_VAL[played.captured] < moverVal) ? 1 : 0
  x[12] = (isCapture && ctx.prevOppCapTo === to) ? 1 : 0
  x[13] = (x[12] && ctx.recapN > 1 && ctx.recapMin !== null && moverVal === ctx.recapMin) ? 1 : 0

  // ---- 14-22: who moved, and which way ----
  x[14 + ORDER.indexOf(mover)] = 1
  x[20] = (Math.abs(tf - 3.5) + Math.abs(tr - 3.5)) / 7
  const forward = (tr - fr) * fwd
  x[21] = forward > 0 ? 1 : 0
  x[22] = forward < 0 ? 1 : 0

  // ---- 23-32: shape of the square and the structure ----
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
  x[29] = ((HOME[me][mover] || []).includes(from)) ? 1 : 0
  x[30] = (mover === "b" && FIANCHETTO[me].includes(to)) ? 1 : 0
  if (mover === "r") {
    // a rook stepping onto a file its OWN pawn still blocks ahead of it
    for (const r of ctx.ownPawnRanks[tf]) { if ((r - tr) * fwd > 0) { x[31] = 1; break } }
  }
  // a square no enemy pawn attacks now or could ever attack
  if (!enemyPawnsAttackTo) {
    let reachable = false
    for (const f of [tf - 1, tf + 1]) {
      if (f < 0 || f > 7) continue
      for (const r of ctx.enemyPawnRanks[f]) { if ((r - tr) * fwd > 0) { reachable = true; break } }
      if (reachable) break
    }
    x[32] = reachable ? 0 : 1
  }

  // ---- 33-42: danger, given and taken ----
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

  // ---- 43-44: where the action was ----
  if (ctx.lastMoveTo) x[43] = (6 - Math.min(cheb(to, ctx.lastMoveTo), 6)) / 6
  if (ctx.prevMyTo) x[44] = (4 - Math.min(cheb(to, ctx.prevMyTo), 4)) / 4

  // ---- 45-50: kings ----
  const kingMove = mover === "k" && !isCastle
  if (kingMove && !ctx.inCheck) {
    if (ctx.canCastle) x[45] = 1
    else { x[46] = ctx.npm / 62; x[47] = Math.min(ctx.kingExposure, 3) / 3 }
  }
  x[48] = (mover === "p" && !isCapture && ctx.shield.has(from) && ctx.castled) ? 1 : 0
  x[49] = unsafeCheck
  x[50] = zonePressure

  // ---- 51-57: the interactions that survive the softmax ----
  x[51] = ctx.phase * forcing
  x[52] = ctx.imbalance * forcing
  x[53] = ctx.phase * (kingMove ? 1 : 0)
  x[54] = ctx.imbalance * x[0]
  x[55] = castlePressure
  x[56] = castleNoShield
  x[57] = aimlessEdge
  return x
}

/**
 * Sample from the v9 policy - the move the bot plays.
 *
 * No salience guards: what they used to patch over is now carried by real
 * features (ignores_fresh_threat, king_walk, quiet_edge_pawn, hangs_*), fitted
 * at the player's own measured rate.
 *
 *   chess    - at the position to move in; left untouched
 *   lines    - the engine's candidates, best first. Either an array of
 *              { uci } objects or app.js's multipv map { 1: {uci}, 2: {uci} },
 *              which is sorted by numeric key. Index in that order IS the
 *              `rank` feature, so do not reorder it.
 *   shByUci  - depth-2 centipawn scores by uci; missing entries are imputed
 *   weights  - the fitted weight vector, length N_FEATURES
 *   temp     - softmax temperature (PLAY_TEMP; 1.0 in app.js)
 *
 * Returns a uci string, or null if there is nothing to choose from.
 *
 * One deliberate difference from app.js's stylePickV9, which is plumbing and
 * not policy: that function returns null when fewer than two candidates
 * survive, because its caller then falls back to the engine's move. With no
 * such caller here, a lone candidate is returned rather than dropped. The
 * distribution over two or more candidates is identical.
 */
export function pick(chess, lines, shByUci, weights, temp = 1.0) {
  if (!weights || !weights.length) return null
  const cands = Array.isArray(lines)
    ? lines
    : Object.keys(lines).sort((a, b) => a - b).map(k => lines[k])
  if (!cands.length) return null
  const ctx = makeContext(chess, null, null, null, null)
  return pickWithContext(chess, cands, shByUci, weights, temp, ctx)
}

// the same sampler, for a caller that already built a context with real
// previous-move facts (makeContext's four arguments). pick() above is the
// no-history convenience wrapper.
export function pickWithContext(chess, cands, shByUci, weights, temp, ctx) {
  if (!weights || !weights.length || !cands.length) return null
  const { sh, bestSh } = horizonScores(cands.map(c => c.uci), shByUci)
  const scored = []
  for (let i = 0; i < cands.length; i++) {
    const x = features(chess, cands[i].uci, ctx, sh[i], bestSh, i)
    if (!x) continue
    let z = 0
    for (let j = 0; j < weights.length; j++) z += weights[j] * x[j]
    scored.push({ uci: cands[i].uci, z })
  }
  if (!scored.length) return null
  if (scored.length === 1) return scored[0].uci
  const zmax = Math.max(...scored.map(c => c.z))
  let total = 0
  const t = temp > 0 ? temp : 1.0
  for (const c of scored) { c.p = Math.exp((c.z - zmax) / t); total += c.p }
  let r = Math.random() * total
  for (const c of scored) { r -= c.p; if (r <= 0) return c.uci }
  return scored[0].uci
}

export const FEATURE_NAMES = [
  "horizon_loss_log", "horizon_loss_log_quiet", "horizon_winprob", "engine_rank",
  "forcing", "capture", "gives_check", "promotion", "castle", "captured_value",
  "capture_of_defended", "loses_exchange", "recapture", "recapture_with_cheapest",
  "is_pawn", "is_knight", "is_bishop", "is_rook", "is_queen", "is_king",
  "center_distance", "forward", "retreat", "minor_piece_retreat", "knight_to_edge",
  "pawn_double_step", "pawn_supported_push", "quiet_edge_pawn",
  "endgame_ramp_x_pawn_push", "home_on_start_square", "bishop_to_fianchetto",
  "rook_blocked_file", "square_no_pawn_can_reach", "from_attacked", "to_attacked",
  "hangs_mover", "leaves_another_hanging", "hangs_count", "hangs_to_pawn",
  "escapes_with_value", "ignores_fresh_threat", "attacks_minor_safely",
  "attacks_undefended_value", "prox_to_last_move", "prox_to_my_last_move",
  "king_walk", "king_walk_free_x_phase", "king_walk_free_x_king_exposure",
  "shield_push_castled", "unsafe_contact_check", "enemy_king_zone_pressure",
  "phase_x_forcing", "imbalance_x_forcing", "phase_x_king_move",
  "imbalance_x_horizon_loss",
  // appended after v9 shipped its first fit - see the castling note above
  "castle_into_pressure", "castle_no_shield",
  // v8 carried this as a hard guard; as a feature the model sets its own rate
  "aimless_edge_pawn",
]

if (FEATURE_NAMES.length !== N_FEATURES) {
  throw new Error(`FEATURE_NAMES is ${FEATURE_NAMES.length}, expected ${N_FEATURES}`)
}
