// Mirror Bot: the conditional-logit fit, in the browser.
//
// This is a port of pipeline/train_v9.py - the same loss, the same pure
// L-BFGS with Armijo backtracking, the same scale-for-the-optimiser /
// ship-raw-weights contract. What differs is everything that was tuned for
// ONE user with 9,431 examples and is wrong for a stranger with 25 games:
//
//  1. FOLDS SPLIT BY GAME. Same as the Python. Two positions from one game
//     are not independent, so a position-wise split leaks and flatters every
//     lambda equally, which is worse than useless when you are choosing one.
//
//  2. L2 SELECTED ON OUT-OF-FOLD LOG-LIKELIHOOD, never on mean probability
//     of the played move. The log score is proper. Mean probability is the
//     LINEAR score: it rises monotonically as a distribution sharpens, so
//     selecting on it walks you to a deterministic bot every time.
//
//  3. THE GRID RUNS TO 1.0. The Python stops at 1e-2, which is a real
//     optimum at 9,000+ examples and badly wrong below that. At a few
//     hundred examples cross-validation lands around 1e-1, ten times
//     stronger. Shipping 1e-2 to a 25-game user is a measurable accuracy
//     loss, and the fix costs four more points on a grid.
//
//  4. FEATURES ARE PRUNED BY THE USER'S OWN FIRING RATE. A feature that
//     fires on under ~0.5% of their candidate rows is fitted on a handful of
//     rows and is noise. It is zeroed and reported. The exception is
//     castling: the three castle features COLLAPSE into one indicator rather
//     than castling vanishing from the model, because how often someone
//     castles is too behaviourally visible to lose.
//
//  5. THE ZERO-VARIANCE GUARD IS 0.02, NOT 1e-8. This is the one that bites.
//     Features are divided by their std for the optimiser and the weights
//     are divided back out before they ship. A feature firing ten times in
//     9,500 rows has std ~0.01; it sails through a 1e-8 guard, gets
//     preconditioned by 100x, and the 100x lands in the RAW weight. That is
//     how a small-sample fit produces +11 and -13 next to a shipped maximum
//     of about 3. Anything below 0.02 std is simply not scaled.
//
// Shipped weights apply to RAW features, exactly as in the Python, so the
// player never has to know scaling happened.
//
// THREADING: fit() is synchronous and touches no DOM, no globals and no
// closures over page state, so it runs in a Worker as-is - that is how the
// page uses it. fitAsync() is the same computation driven through a
// generator that pauses between folds and every few L-BFGS iterations, for
// callers that want it on the main thread anyway. Both return the same
// object; fit() is the one to use.

const NEG = -1e9 // logit for a candidate slot that is not a real move

// ---------------------------------------------------------------- defaults

export const DEFAULTS = {
  // reaches 1.0; the Python stops at 1e-2
  l2Grid: [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0],
  folds: 5,
  seed: 11,
  // a feature firing on fewer than this fraction of candidate rows is zeroed
  pruneRate: 0.005,
  // optional absolute floor on firing rows; 0 disables it
  minFiringRows: 0,
  // below this std a feature is NOT divided by its std - see note 5 above
  varianceFloor: 0.02,
  maxIter: 400,
  memory: 10,
  tol: 1e-7,
  // L-BFGS starts from zero for every fold, like the Python. Warm-starting
  // down the lambda grid is faster and lands in the same place (the problem
  // is strictly convex for lam > 0), but it is off by default so a fit is
  // reproducible from its inputs alone.
  warmStart: false,
  // fallback when there are too few games to cross-validate at all
  fallbackL2: 0.1,
  // fitAsync pauses roughly this often inside L-BFGS
  yieldEvery: 5,
}

// ------------------------------------------------------------------- utils

// seeded PRNG: folds have to be reproducible, and Math.random is not
function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// numpy's array_split: k parts, the first (n % k) of them one longer
function arraySplit(arr, k) {
  const out = []
  const n = arr.length
  const base = Math.floor(n / k)
  const extra = n % k
  let at = 0
  for (let i = 0; i < k; i++) {
    const len = base + (i < extra ? 1 : 0)
    out.push(arr.slice(at, at + len))
    at += len
  }
  return out
}

function nextTick() {
  return new Promise((resolve) => setTimeout(resolve, 0))
}

// ------------------------------------------------------- design matrix prep

// One pass over the dense X to get, per feature, the firing rate and the std
// over REAL candidate rows only (padded slots would drag both toward zero).
function featureStats(X, M, N, K, D) {
  const count = new Float64Array(D)
  const sum = new Float64Array(D)
  const sumsq = new Float64Array(D)
  let rows = 0
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < K; k++) {
      if (!M[n * K + k]) continue
      rows++
      const base = (n * K + k) * D
      for (let d = 0; d < D; d++) {
        const v = X[base + d]
        if (v === 0) continue
        count[d]++
        sum[d] += v
        sumsq[d] += v * v
      }
    }
  }
  const rate = new Float64Array(D)
  const std = new Float64Array(D)
  for (let d = 0; d < D; d++) {
    rate[d] = rows ? count[d] / rows : 0
    const mean = rows ? sum[d] / rows : 0
    const varc = rows ? Math.max(0, sumsq[d] / rows - mean * mean) : 0
    std[d] = Math.sqrt(varc)
  }
  return { rate, std, rows, count }
}

// Which castle features are we allowed to collapse into one? Names win; an
// explicit index list is honoured; the v9 contract is the last resort.
function castleGroup(opts, D) {
  if (Array.isArray(opts.castleFeatures)) return opts.castleFeatures.filter((i) => i >= 0 && i < D)
  const names = opts.featureNames
  if (Array.isArray(names) && names.length === D) {
    // anchored on purpose: shield_push_castled is a pawn move made WHILE
    // castled, not a castling move, and folding it in here would be wrong
    const hit = []
    for (let d = 0; d < D; d++) if (/^castle/i.test(String(names[d]))) hit.push(d)
    return hit
  }
  if (D === 58) return [8, 55, 56] // castle, castle_into_pressure, castle_no_shield
  return []
}

// Decide the active feature set, the scaling, and what castling collapses to.
function planFeatures(stats, opts, D) {
  const pruneRate = opts.pruneRate ?? DEFAULTS.pruneRate
  const minRows = opts.minFiringRows ?? DEFAULTS.minFiringRows
  const floor = opts.varianceFloor ?? DEFAULTS.varianceFloor

  const keep = new Uint8Array(D)
  for (let d = 0; d < D; d++) {
    keep[d] = stats.rate[d] >= pruneRate && stats.count[d] >= minRows && stats.std[d] > 0 ? 1 : 0
  }

  // Castling: if ANY of the castle features is too rare to fit, we do not
  // drop castling - we keep the single best-populated castle column and fold
  // the rest into it. The finer castle features only fire on castling moves
  // and are bounded by the plain castle indicator, so the survivor carries
  // the union of the group and the weight stays a RAW-unit weight on a real
  // feature. `lossy` says so when that containment does not hold.
  const group = castleGroup(opts, D).filter((d) => stats.std[d] > 0)
  let collapse = null
  if (group.length > 1) {
    const anyThin = group.some((d) => stats.rate[d] < pruneRate || stats.count[d] < minRows)
    if (anyThin) {
      let base = group[0]
      for (const d of group) if (stats.rate[d] > stats.rate[base]) base = d
      const folded = group.filter((d) => d !== base)
      collapse = { base, folded, lossy: false }
      keep[base] = 1
      for (const d of folded) keep[d] = 0
    }
  }

  const scale = new Float64Array(D)
  const guarded = []
  for (let d = 0; d < D; d++) {
    if (stats.std[d] < floor) {
      scale[d] = 1
      if (keep[d]) guarded.push(d)
    } else {
      scale[d] = stats.std[d]
    }
  }

  const active = []
  const dropped = []
  for (let d = 0; d < D; d++) (keep[d] ? active : dropped).push(d)
  return { keep, scale, active, dropped, guarded, collapse }
}

// Build a compact sparse design over the kept positions. ~79% of a v9 row is
// zero, so candidate-major CSR turns both the score pass and the gradient
// pass into O(nonzeros) and the whole fit into something a laptop finishes.
// Values are stored ALREADY DIVIDED by scale; nothing downstream rescales.
function buildDesign(X, M, y, g, N, K, D, plan) {
  const colOf = new Int32Array(D).fill(-1)
  plan.active.forEach((d, j) => {
    colOf[d] = j
  })
  const nCols = plan.active.length

  const kept = []
  const skippedNotInPool = []
  for (let n = 0; n < N; n++) {
    const yk = y[n]
    if (yk < 0 || yk >= K || !M[n * K + yk]) {
      skippedNotInPool.push(n)
      continue
    }
    kept.push(n)
  }
  const P = kept.length

  const posStart = new Int32Array(P + 1)
  let cand = 0
  for (let i = 0; i < P; i++) {
    const n = kept[i]
    let c = 0
    for (let k = 0; k < K; k++) if (M[n * K + k]) c++
    posStart[i] = cand
    cand += c
  }
  posStart[P] = cand

  // count nonzeros per candidate, then fill
  const nzStart = new Int32Array(cand + 1)
  const yLocal = new Int32Array(P)
  const gameOf = new Int32Array(P)
  const slotOf = new Int32Array(cand)
  let ci = 0
  let maxCand = 0
  for (let i = 0; i < P; i++) {
    const n = kept[i]
    gameOf[i] = g[n]
    let local = 0
    for (let k = 0; k < K; k++) {
      if (!M[n * K + k]) continue
      if (k === y[n]) yLocal[i] = local
      slotOf[ci] = n * K + k
      const base = (n * K + k) * D
      let nz = 0
      for (let d = 0; d < D; d++) {
        if (colOf[d] < 0) continue
        if (X[base + d] !== 0) nz++
      }
      nzStart[ci + 1] = nz
      ci++
      local++
    }
    if (local > maxCand) maxCand = local
  }
  for (let c = 0; c < cand; c++) nzStart[c + 1] += nzStart[c]

  const nnz = nzStart[cand]
  const nzIdx = new Int32Array(nnz)
  const nzVal = new Float64Array(nnz)
  for (let c = 0; c < cand; c++) {
    const base = slotOf[c] * D
    let at = nzStart[c]
    for (let d = 0; d < D; d++) {
      const j = colOf[d]
      if (j < 0) continue
      const v = X[base + d]
      if (v === 0) continue
      nzIdx[at] = j
      nzVal[at] = v / plan.scale[d]
      at++
    }
  }

  // Did the collapse actually lose anything? The survivor is only a faithful
  // stand-in for the group if it fires wherever any member fires.
  if (plan.collapse) {
    const { base, folded } = plan.collapse
    for (let c = 0; c < cand && !plan.collapse.lossy; c++) {
      const off = slotOf[c] * D
      if (X[off + base] !== 0) continue
      for (const d of folded) {
        if (X[off + d] !== 0) {
          plan.collapse.lossy = true
          break
        }
      }
    }
  }

  return {
    P, nCols, posStart, nzStart, nzIdx, nzVal, yLocal, gameOf, kept, slotOf,
    maxCand, nnz, skippedNotInPool,
  }
}

// ------------------------------------------------------------ the objective

// Conditional-logit negative log-likelihood and gradient over a row subset.
// Identical arithmetic to nll_and_grad() in the Python, just sparse: padded
// slots are absent from the design instead of being pushed to NEG.
function makeObjective(design, rows, D) {
  const { posStart, nzStart, nzIdx, nzVal, yLocal } = design
  const z = new Float64Array(design.maxCand)
  const grad = new Float64Array(D)
  const n = rows.length
  // the caller sets objective.lam before each solve; the grid reuses the
  // closure so the design is walked once, not once per lambda
  objective.lam = 0
  function objective(w) {
    grad.fill(0)
    let loss = 0
    for (let r = 0; r < n; r++) {
      const pos = rows[r]
      const c0 = posStart[pos]
      const c1 = posStart[pos + 1]
      const m = c1 - c0
      let zmax = -Infinity
      for (let c = c0; c < c1; c++) {
        let acc = 0
        const t1 = nzStart[c + 1]
        for (let t = nzStart[c]; t < t1; t++) acc += w[nzIdx[t]] * nzVal[t]
        z[c - c0] = acc
        if (acc > zmax) zmax = acc
      }
      const chosen = yLocal[pos]
      const zChosen = z[chosen]
      let s = 0
      for (let j = 0; j < m; j++) {
        const e = Math.exp(z[j] - zmax)
        z[j] = e
        s += e
      }
      loss += zmax + Math.log(s) - zChosen
      for (let c = c0; c < c1; c++) {
        let p = z[c - c0] / s
        if (c - c0 === chosen) p -= 1
        if (p === 0) continue
        const t1 = nzStart[c + 1]
        for (let t = nzStart[c]; t < t1; t++) grad[nzIdx[t]] += p * nzVal[t]
      }
    }
    let wsq = 0
    for (let d = 0; d < D; d++) wsq += w[d] * w[d]
    const lam = objective.lam
    const out = loss / n + 0.5 * lam * wsq
    for (let d = 0; d < D; d++) grad[d] = grad[d] / n + lam * w[d]
    return { loss: out, grad }
  }
  return objective
}

// mean p(played move), top-1 agreement, mean log-likelihood - the same three
// numbers the Python prints, and the third one is the only one that selects.
function policyMetrics(design, rows, w, temp = 1) {
  const { posStart, nzStart, nzIdx, nzVal, yLocal } = design
  const z = new Float64Array(design.maxCand)
  let sumP = 0
  let hits = 0
  let sumLog = 0
  for (let r = 0; r < rows.length; r++) {
    const pos = rows[r]
    const c0 = posStart[pos]
    const c1 = posStart[pos + 1]
    const m = c1 - c0
    let zmax = -Infinity
    for (let c = c0; c < c1; c++) {
      let acc = 0
      const t1 = nzStart[c + 1]
      for (let t = nzStart[c]; t < t1; t++) acc += w[nzIdx[t]] * nzVal[t]
      acc /= temp
      z[c - c0] = acc
      if (acc > zmax) zmax = acc
    }
    let s = 0
    let best = 0
    for (let j = 0; j < m; j++) {
      const e = Math.exp(z[j] - zmax)
      z[j] = e
      s += e
      if (e > z[best]) best = j
    }
    const chosen = yLocal[pos]
    const p = z[chosen] / s
    sumP += p
    sumLog += Math.log(p)
    if (best === chosen) hits++
  }
  const n = rows.length || 1
  return { meanP: sumP / n, top1: hits / n, logLik: sumLog / n }
}

// ------------------------------------------------------------------ L-BFGS

// Limited-memory BFGS with Armijo backtracking, two-loop recursion, no
// dependencies. Line-for-line the Python's lbfgs(), as a generator so the
// async driver can pause between iterations.
function* lbfgsGen(objective, w0, opts) {
  const iters = opts.maxIter ?? DEFAULTS.maxIter
  const mem = opts.memory ?? DEFAULTS.memory
  const tol = opts.tol ?? DEFAULTS.tol
  const yieldEvery = opts.yieldEvery ?? DEFAULTS.yieldEvery
  const D = w0.length
  let w = Float64Array.from(w0)
  let res = objective(w)
  let loss = res.loss
  let g = Float64Array.from(res.grad)
  const S = []
  const Y = []
  const rho = []
  const q = new Float64Array(D)
  let it = 0
  for (; it < iters; it++) {
    let gmax = 0
    for (let d = 0; d < D; d++) gmax = Math.max(gmax, Math.abs(g[d]))
    if (gmax < tol) break
    if (yieldEvery > 0 && it % yieldEvery === 0) yield { phase: 'lbfgs', iteration: it, loss }

    q.set(g)
    const alpha = []
    for (let i = S.length - 1; i >= 0; i--) {
      let sq = 0
      for (let d = 0; d < D; d++) sq += S[i][d] * q[d]
      const a = rho[i] * sq
      alpha.push(a)
      for (let d = 0; d < D; d++) q[d] -= a * Y[i][d]
    }
    if (S.length) {
      const last = S.length - 1
      let sy = 0
      let yy = 0
      for (let d = 0; d < D; d++) {
        sy += S[last][d] * Y[last][d]
        yy += Y[last][d] * Y[last][d]
      }
      const gamma = sy / yy
      for (let d = 0; d < D; d++) q[d] *= gamma
    }
    for (let i = 0; i < S.length; i++) {
      let yq = 0
      for (let d = 0; d < D; d++) yq += Y[i][d] * q[d]
      const b = rho[i] * yq
      const a = alpha[alpha.length - 1 - i]
      for (let d = 0; d < D; d++) q[d] += S[i][d] * (a - b)
    }

    const dir = new Float64Array(D)
    for (let d = 0; d < D; d++) dir[d] = -q[d]
    let slope = 0
    for (let d = 0; d < D; d++) slope += g[d] * dir[d]
    if (slope >= 0) {
      // curvature broke down; reset to steepest descent and empty the memory
      for (let d = 0; d < D; d++) dir[d] = -g[d]
      slope = 0
      for (let d = 0; d < D; d++) slope += g[d] * dir[d]
      S.length = 0
      Y.length = 0
      rho.length = 0
    }

    let t = 1
    let accepted = false
    let wNew = null
    let lossNew = 0
    let gNew = null
    for (let ls = 0; ls < 25; ls++) {
      wNew = new Float64Array(D)
      for (let d = 0; d < D; d++) wNew[d] = w[d] + t * dir[d]
      const r = objective(wNew)
      lossNew = r.loss
      gNew = Float64Array.from(r.grad)
      if (lossNew <= loss + 1e-4 * t * slope) {
        accepted = true
        break
      }
      t *= 0.5
    }
    if (!accepted) break // no acceptable step; we are done

    const sVec = new Float64Array(D)
    const yVec = new Float64Array(D)
    let sy = 0
    for (let d = 0; d < D; d++) {
      sVec[d] = wNew[d] - w[d]
      yVec[d] = gNew[d] - g[d]
      sy += sVec[d] * yVec[d]
    }
    if (sy > 1e-12) {
      S.push(sVec)
      Y.push(yVec)
      rho.push(1 / sy)
      if (S.length > mem) {
        S.shift()
        Y.shift()
        rho.shift()
      }
    }
    w = wNew
    loss = lossNew
    g = gNew
  }
  let gmax = 0
  for (let d = 0; d < D; d++) gmax = Math.max(gmax, Math.abs(g[d]))
  return { w, loss, gradMax: gmax, iterations: it + 1 }
}

// ------------------------------------------------------------------- folds

// Split by GAME. Positions inside one game share an opponent, an opening and
// a mood; splitting by position lets a fold grade itself on its own games.
function gameFolds(gameOf, P, k, seed) {
  const seen = new Set()
  const games = []
  for (let i = 0; i < P; i++) {
    const gid = gameOf[i]
    if (!seen.has(gid)) {
      seen.add(gid)
      games.push(gid)
    }
  }
  games.sort((a, b) => a - b)
  const rand = mulberry32(seed)
  for (let i = games.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    const tmp = games[i]
    games[i] = games[j]
    games[j] = tmp
  }
  const buckets = arraySplit(games, Math.max(1, Math.min(k, games.length)))
  const bucketOf = new Map()
  buckets.forEach((b, bi) => b.forEach((gid) => bucketOf.set(gid, bi)))
  return buckets.map((_, bi) => {
    const train = []
    const test = []
    for (let i = 0; i < P; i++) (bucketOf.get(gameOf[i]) === bi ? test : train).push(i)
    return { train: Int32Array.from(train), test: Int32Array.from(test) }
  })
}

// ------------------------------------------------------------- the fit body

function* fitGen(X, M, y, g, opts = {}) {
  const N = y.length
  const K = opts.pool ?? (M.length / N)
  const D = opts.nFeatures ?? (X.length / (N * K))
  if (!Number.isInteger(K) || !Number.isInteger(D) || K < 1 || D < 1) {
    throw new Error(`cannot infer shape: N=${N} pool=${K} nFeatures=${D}`)
  }
  if (X.length !== N * K * D) throw new Error(`X is ${X.length}, expected ${N * K * D}`)
  if (M.length !== N * K) throw new Error(`M is ${M.length}, expected ${N * K}`)
  if (g.length !== N) throw new Error(`g is ${g.length}, expected ${N}`)

  const report = (p) => {
    if (typeof opts.onProgress === 'function') opts.onProgress(p)
    return p
  }

  yield report({ phase: 'stats', done: 0, total: 1 })
  const stats = featureStats(X, M, N, K, D)
  const plan = planFeatures(stats, opts, D)
  const design = buildDesign(X, M, y, g, N, K, D, plan)
  const P = design.P
  if (P === 0) throw new Error('no usable examples: the played move was never among the candidates')
  const nCols = design.nCols
  if (nCols === 0) throw new Error('every feature was pruned: not enough games to fit anything')

  const nGames = new Set(Array.from(design.gameOf)).size
  yield report({
    phase: 'prepared', examples: P, games: nGames, features: nCols,
    dropped: plan.dropped.length, nonzeros: design.nnz,
  })

  const grid = (opts.l2Grid ?? DEFAULTS.l2Grid).slice().sort((a, b) => a - b)
  const foldCount = Math.min(opts.folds ?? DEFAULTS.folds, nGames)
  const seed = opts.seed ?? DEFAULTS.seed
  const warm = opts.warmStart ?? DEFAULTS.warmStart

  let bestL2 = opts.fallbackL2 ?? DEFAULTS.fallbackL2
  let curve = []
  let oofLogLik = null
  let oofTop1 = null
  let oofMeanP = null

  if (foldCount >= 2) {
    const folds = gameFolds(design.gameOf, P, foldCount, seed)
    const objectives = folds.map((f) => makeObjective(design, f.train, nCols))
    // warm starts run from the strongest penalty down: the heavily shrunk fit
    // is a good starting point for the next one along, never the reverse
    const order = warm ? grid.slice().reverse() : grid
    const start = folds.map(() => new Float64Array(nCols))
    const total = order.length * folds.length
    let done = 0
    const byLam = new Map()
    for (const lam of order) {
      const ps = []
      const accs = []
      const lls = []
      for (let fi = 0; fi < folds.length; fi++) {
        objectives[fi].lam = lam
        const res = yield* lbfgsGen(objectives[fi], warm ? start[fi] : new Float64Array(nCols), opts)
        if (warm) start[fi] = res.w
        const m = policyMetrics(design, folds[fi].test, res.w)
        ps.push(m.meanP)
        accs.push(m.top1)
        lls.push(m.logLik)
        done++
        yield report({ phase: 'cv', done, total, l2: lam, fold: fi })
      }
      const mean = (a) => a.reduce((x, v) => x + v, 0) / a.length
      byLam.set(lam, { l2: lam, meanP: mean(ps), top1: mean(accs), logLik: mean(lls) })
    }
    curve = grid.map((lam) => byLam.get(lam))
    // proper scoring rule, and the ONLY thing allowed to choose lambda
    let best = curve[0]
    for (const row of curve) if (row.logLik > best.logLik) best = row
    bestL2 = best.l2
    oofLogLik = best.logLik
    oofTop1 = best.top1
    oofMeanP = best.meanP
  } else {
    yield report({
      phase: 'cv-skipped', games: nGames,
      message: `only ${nGames} game(s): cannot split folds by game, using L2 ${bestL2}`,
    })
  }

  yield report({ phase: 'final', l2: bestL2 })
  const objective = makeObjective(design, Int32Array.from({ length: P }, (_, i) => i), nCols)
  objective.lam = bestL2
  const final = yield* lbfgsGen(objective, new Float64Array(nCols), opts)

  // back to RAW units - the player dots these straight against raw features
  const weights = new Float64Array(D)
  plan.active.forEach((d, j) => {
    weights[d] = final.w[j] / plan.scale[d]
  })

  const inSample = policyMetrics(design, Int32Array.from({ length: P }, (_, i) => i), final.w)

  // The behavioural gate: for each feature, the average over the moves they
  // played against the average the fitted distribution puts on it. Catches
  // changes that are behaviourally real and statistically invisible.
  const mismatch = behaviourMismatch(X, D, K, design, final.w, plan)

  const result = {
    weights,
    l2: bestL2,
    outOfFoldLogLik: oofLogLik,
    top1: oofTop1,
    behaviourMismatch: mismatch.mean,
    droppedFeatures: plan.dropped.slice(),
    // everything below is extra, for the page to show and for auditing
    outOfFoldMeanP: oofMeanP,
    inSampleMeanP: inSample.meanP,
    inSampleLogLik: inSample.logLik,
    inSampleTop1: inSample.top1,
    activeFeatures: plan.active.slice(),
    guardedFeatures: plan.guarded.slice(),
    castleCollapse: plan.collapse,
    firingRates: stats.rate,
    featureStd: stats.std,
    scale: plan.scale,
    perFeatureMismatch: mismatch.perFeature,
    l2Curve: curve,
    examples: P,
    games: nGames,
    pool: K,
    nFeatures: D,
    folds: foldCount >= 2 ? foldCount : 0,
    skippedNotInPool: design.skippedNotInPool.length,
    iterations: final.iterations,
    gradMax: final.gradMax,
    loss: final.loss,
    nonzeros: design.nnz,
  }
  return result
}

function behaviourMismatch(X, D, K, design, w, plan) {
  const { posStart, nzStart, nzIdx, nzVal, yLocal, slotOf, P } = design
  const his = new Float64Array(D)
  const bot = new Float64Array(D)
  const z = new Float64Array(design.maxCand)
  for (let i = 0; i < P; i++) {
    const c0 = posStart[i]
    const c1 = posStart[i + 1]
    const m = c1 - c0
    let zmax = -Infinity
    for (let c = c0; c < c1; c++) {
      let acc = 0
      const t1 = nzStart[c + 1]
      for (let t = nzStart[c]; t < t1; t++) acc += w[nzIdx[t]] * nzVal[t]
      z[c - c0] = acc
      if (acc > zmax) zmax = acc
    }
    let s = 0
    for (let j = 0; j < m; j++) {
      const e = Math.exp(z[j] - zmax)
      z[j] = e
      s += e
    }
    const chosenBase = slotOf[c0 + yLocal[i]] * D
    for (let d = 0; d < D; d++) his[d] += X[chosenBase + d]
    for (let c = c0; c < c1; c++) {
      const p = z[c - c0] / s
      const base = slotOf[c] * D
      for (let d = 0; d < D; d++) bot[d] += p * X[base + d]
    }
  }
  const perFeature = new Float64Array(D)
  let sum = 0
  for (let d = 0; d < D; d++) {
    perFeature[d] = Math.abs(bot[d] / P - his[d] / P) / plan.scale[d]
    sum += perFeature[d]
  }
  return { mean: sum / D, perFeature }
}

// -------------------------------------------------------------- public API

/**
 * Fit the conditional logit. Synchronous and self-contained: no DOM, no
 * globals, so this is what the Worker calls.
 *
 * @param {Float64Array} X  flattened (N * pool * nFeatures) candidate features
 * @param {Uint8Array}   M  (N * pool) 1 where the candidate slot is a real move
 * @param {Int32Array}   y  (N) index of the move actually played
 * @param {Int32Array}   g  (N) game id, so folds can split by game
 * @param {object}       opts {nFeatures, pool, l2Grid, folds, onProgress, ...}
 * @returns {object} {weights (RAW units), l2, outOfFoldLogLik, top1,
 *                    behaviourMismatch, droppedFeatures, ...}
 */
export function fit(X, M, y, g, opts = {}) {
  const it = fitGen(X, M, y, g, opts)
  let step = it.next()
  while (!step.done) step = it.next()
  return step.value
}

/**
 * The same fit, paused between folds and every few L-BFGS iterations so the
 * main thread can paint. opts.yield may supply your own pause.
 */
export async function fitAsync(X, M, y, g, opts = {}) {
  const pause = typeof opts.yield === 'function' ? opts.yield : nextTick
  const it = fitGen(X, M, y, g, opts)
  let step = it.next()
  while (!step.done) {
    await pause()
    step = it.next()
  }
  return step.value
}

/**
 * Sample one candidate from the fitted distribution.
 *
 * @param {ArrayLike<number>} scores  z per candidate (weights . raw features)
 * @param {ArrayLike<number>|null} mask 1/true where the candidate is real
 * @param {number} temp  1 plays the fitted distribution exactly; <= 0 is argmax
 * @param {function} rand  optional source of randomness, for tests
 * @returns {number} index into scores, or -1 if nothing is playable
 */
export function softmaxPick(scores, mask, temp = 1, rand = Math.random) {
  const n = scores.length
  const valid = (i) => (mask ? !!mask[i] : true) && Number.isFinite(scores[i]) && scores[i] > NEG / 2
  let best = -1
  let zmax = -Infinity
  for (let i = 0; i < n; i++) {
    if (!valid(i)) continue
    if (scores[i] > zmax) {
      zmax = scores[i]
      best = i
    }
  }
  if (best < 0) return -1
  if (!(temp > 0)) return best // temp 0 (or nonsense) means play the argmax

  const p = new Float64Array(n)
  let total = 0
  for (let i = 0; i < n; i++) {
    if (!valid(i)) continue
    const e = Math.exp((scores[i] - zmax) / temp)
    p[i] = e
    total += e
  }
  if (!(total > 0)) return best
  let r = rand() * total
  for (let i = 0; i < n; i++) {
    if (!valid(i)) continue
    r -= p[i]
    if (r <= 0) return i
  }
  return best
}

export default { fit, fitAsync, softmaxPick, DEFAULTS }
