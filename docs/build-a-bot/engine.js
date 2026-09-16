// Build-a-Bot: Stockfish WASM in a Worker.
//
// Same engine and same protocol as the Burke Bot page (docs/app.js), with two
// differences that matter here:
//
//   1. analyse(fen, depth, multipv) is a single generic entry point, because
//      Build-a-Bot runs the WHOLE pipeline in the browser - the deep pass that
//      ranks candidates and the shallow pass that models what a player can see
//      over the board are the same call at different depths.
//   2. The 7.0 MB .wasm comes from jsDelivr, verified by Subresource Integrity,
//      so a stranger's visit doesn't cost GitHub Pages bandwidth. Same-origin
//      ../vendor/stockfish/ is the fallback.
//
// Scores follow pipeline/analyse.py exactly - see toScore() - because the
// features fitted in the browser have to mean the same thing as the ones the
// python pipeline fits. A sign flip here is silent: the model still trains,
// still plays legal moves, and is wrong.

// ---------- constants ----------

// Pinned to the commit whose bytes the SRI hash below was taken from. jsDelivr
// serves a git ref immutably, so the pin and the hash can never drift apart:
// if the file at this commit ever changed, the integrity check fails and we
// fall back to the copy this site serves itself.
const COMMIT = "9b86a42fcc5385b8abaff03d62448ccae913373d"

export const CDN = {
  commit: COMMIT,
  // sha384 of docs/vendor/stockfish/stockfish-18-lite-single.wasm at COMMIT.
  // Verify with:
  //   openssl dgst -sha384 -binary <file> | openssl base64 -A
  sri: "sha384-FtRS4u+H6G00OtAtwDM34s6GnIS5R8GMKr3U2Qwtv6iH1ysFcPW+/uZzAO1xfuHb",
  wasmUrl: `https://cdn.jsdelivr.net/gh/Aburke225/burke-bot@${COMMIT}/docs/vendor/stockfish/stockfish-18-lite-single.wasm`,
  // The loader is small (21 KB) and must be same-origin anyway - a Worker
  // cannot be constructed from a cross-origin script URL - so it is never
  // fetched from the CDN. Only the 7 MB payload is.
  jsUrl: new URL("../vendor/stockfish/stockfish-18-lite-single.js", import.meta.url).href,
  localWasmUrl: new URL("../vendor/stockfish/stockfish-18-lite-single.wasm", import.meta.url).href,
  bytes: 7295411,
}

// Hash: 16 MiB, on every device including phones.
//
// Measured on the vendored binary rather than guessed. Its wasm memory section
// declares initial = maximum = 2048 pages, i.e. a FIXED 128 MiB linear memory
// with no growth, so every worker reserves 128 MiB the moment it instantiates
// whatever Hash is set to. The transposition table is malloc'd inside that
// fixed heap, on top of ~7 MB of static data (the NNUE net is embedded in the
// binary), so the lever isn't "how much memory does the tab use" - it's "how
// much room is left before an allocation fails and the engine aborts".
//
// 16 MiB is Stockfish's own default here (`option name Hash type spin default
// 16 min 1 max 33554432`) and is what pipeline/analyse.py runs with, so the
// browser ranks candidates exactly the way the reference pipeline does. Going
// to 8 would save 8 MiB out of a 128 MiB reservation - about 6% - in exchange
// for a search that explores differently from the reference. Bad trade; the
// real mobile lever is running ONE worker instead of two, which saves 128.
//
// Pass hashMiB to go lower if a page really does want two workers on a phone.
const DEFAULT_HASH_MIB = 16

// Real positions, not the start position: an opening tree is unrepresentatively
// cheap and would make the page promise a game budget it can't keep. Mix of
// opening, middlegame and endgame, all reached in ordinary play.
const CALIBRATION_FENS = [
  "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
  "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 6 4",
  "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 2 9",
  "r1bqk2r/pp2bppp/2np1n2/4p3/4P3/1NN1BP2/PPPQ2PP/2KR1B1R w kq - 2 10",
  "2r3k1/pp3pbp/6p1/q2Pp3/2P1P3/1P4P1/P1Q2PBP/3R2K1 w - - 0 22",
  "8/5pk1/6p1/3R3p/4P3/5P2/r5PP/6K1 w - - 0 35",
]

// The depth/width the site plays at - a contract with pipeline/analyse.py's
// POOL_DEPTH and MULTIPV. calibrate() must time THIS, not something cheaper.
const CALIBRATION_DEPTH = 5
const CALIBRATION_MULTIPV = 20

// mate_score=10000, from pipeline/analyse.py's score(mate_score=10000).
const MATE_SCORE = 10000

// ---------- the fallback hook ----------
//
// Called whenever the CDN copy is not what gets loaded. A beacon can be wired
// here later without touching the engine: setFallbackHook(info => fetch(...)).
// Never throws into the load path - a broken beacon must not break the engine.

let globalFallbackHook = null

export function setFallbackHook(fn) {
  globalFallbackHook = typeof fn === "function" ? fn : null
}

function reportFallback(hook, info) {
  for (const fn of [hook, globalFallbackHook]) {
    if (!fn) continue
    try { fn(info) } catch (e) { console.warn("build-a-bot/engine: fallback hook threw", e) }
  }
}

// ---------- score conversion ----------
//
// This is the one function in the file that has to be exactly right.
//
// UCI scores are already from the side to move's point of view, which is what
// analyse.py asks python-chess for (`info["score"].pov(board.turn)` where
// board.turn IS the mover - a no-op that documents the convention). So: no
// flip, ever. A caller wanting White's point of view flips it themselves.
//
// Mate folding matches python-chess's Score.score(mate_score=10000):
//   mate n > 0  ->  10000 - n     (we deliver mate in n)
//   mate n <= 0 -> -10000 - n     (we get mated in |n|; mate 0 -> -10000)
// Note mate 0 lands on -10000 in BOTH implementations, because python-chess
// tests `moves > 0`, not `moves >= 0`. It shows up on a position that is
// already checkmate, and getting it wrong would put a +10000 on a lost game.
function toScore(cpToken, mateToken) {
  if (cpToken !== null) return { cp: cpToken, mate: null }
  if (mateToken !== null) {
    return { cp: mateToken > 0 ? MATE_SCORE - mateToken : -MATE_SCORE - mateToken, mate: mateToken }
  }
  return null
}

// ---------- line parsing ----------

const RE_MULTIPV = /\bmultipv (\d+)\b/
const RE_CP = /\bscore cp (-?\d+)\b/
const RE_MATE = /\bscore mate (-?\d+)\b/
const RE_DEPTH = /\bdepth (\d+)\b/
const RE_PV = /\bpv ((?:[a-h][1-8][a-h][1-8][qrbnQRBN]?)(?: [a-h][1-8][a-h][1-8][qrbnQRBN]?)*)\s*$/

function parseInfo(line) {
  // currmove/currmovenumber updates and `info string ...` carry no pv and are
  // skipped by the pv match below, which is also why the pv regex is anchored:
  // the pv is always the last field on the line.
  const pv = RE_PV.exec(line)
  if (!pv) return null
  const cp = RE_CP.exec(line)
  const mate = RE_MATE.exec(line)
  const score = toScore(cp ? parseInt(cp[1], 10) : null, mate ? parseInt(mate[1], 10) : null)
  if (!score) return null
  const mpv = RE_MULTIPV.exec(line)
  const depth = RE_DEPTH.exec(line)
  const moves = pv[1].split(" ")
  return {
    // A MultiPV 1 search omits the multipv field entirely; treat it as rank 1,
    // the way docs/app.js does.
    multipv: mpv ? parseInt(mpv[1], 10) : 1,
    depth: depth ? parseInt(depth[1], 10) : null,
    uci: moves[0],
    cp: score.cp,
    mate: score.mate,
    pv: moves,
  }
}

// ---------- wasm acquisition ----------

async function fetchWithIntegrity(url, integrity, signal) {
  const res = await fetch(url, {
    integrity,
    mode: "cors",
    credentials: "omit",
    cache: "force-cache",
    signal,
  })
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
  return res.arrayBuffer()
}

// fetch() reports an integrity failure the same way it reports a dead network:
// a rejected promise with a TypeError. A beacon wants to tell those apart -
// "the CDN is down" and "the bytes on the CDN changed" are different incidents
// - so probe reachability with a HEAD before giving up. Cheap, best-effort,
// and never allowed to fail the load.
async function classifyFetchFailure(url) {
  try {
    const ctl = typeof AbortController === "function" ? new AbortController() : null
    const timer = ctl ? setTimeout(() => ctl.abort(), 3000) : null
    try {
      const res = await fetch(url, { method: "HEAD", mode: "cors", credentials: "omit", signal: ctl && ctl.signal })
      return res.ok ? "integrity-or-body" : `http-${res.status}`
    } finally {
      if (timer) clearTimeout(timer)
    }
  } catch {
    return "unreachable"
  }
}

// Returns a URL the worker can fetch. The worker's own loader has no SRI
// support (it calls plain fetch + instantiateStreaming), so verification has to
// happen out here: download once, let fetch enforce the hash, hand the verified
// bytes over as a blob: URL. The alternative - letting the worker fetch the CDN
// directly - would mean running 7 MB of unverified third-party wasm.
async function acquireWasm({ wasmUrl, integrity, fallbackWasmUrl, onFallback }) {
  if (!integrity) {
    // No hash to check against: don't pretend. Hand the URL straight to the
    // worker, which is the right thing for a same-origin file anyway.
    return { url: wasmUrl, source: "direct", revoke: null }
  }
  const t0 = now()
  try {
    const buf = await fetchWithIntegrity(wasmUrl, integrity)
    if (typeof Blob !== "function" || typeof URL.createObjectURL !== "function") {
      throw new Error("no Blob/createObjectURL in this environment")
    }
    // instantiateStreaming refuses anything that isn't application/wasm, and a
    // blob: response reports its blob's type, so this type is load-bearing.
    const blobUrl = URL.createObjectURL(new Blob([buf], { type: "application/wasm" }))
    return {
      url: blobUrl,
      source: "cdn",
      bytes: buf.byteLength,
      ms: Math.round(now() - t0),
      revoke: () => { try { URL.revokeObjectURL(blobUrl) } catch {} },
    }
  } catch (err) {
    const reason = await classifyFetchFailure(wasmUrl)
    reportFallback(onFallback, {
      stage: "fetch",
      reason,
      url: wasmUrl,
      fallbackUrl: fallbackWasmUrl,
      ms: Math.round(now() - t0),
      error: String(err && err.message || err),
    })
    // Same-origin: same trust domain as the page's own code, so it goes
    // straight to the worker unverified rather than being downloaded twice.
    return { url: fallbackWasmUrl, source: "local", revoke: null }
  }
}

function now() {
  return (typeof performance === "object" && performance && typeof performance.now === "function")
    ? performance.now()
    : Date.now()
}

// ---------- the engine ----------

/**
 * Boot a Stockfish worker.
 *
 * @param {object}   [opts]
 * @param {string}   [opts.wasmUrl]  where the .wasm lives (default: the CDN pin)
 * @param {string}   [opts.jsUrl]    the loader; must be same-origin (Worker rule)
 * @param {string|number} [opts.hash] SRI string for wasmUrl. A NUMBER here is
 *        read as the UCI Hash in MiB instead, because "hash" is ambiguous in
 *        chess code and silently doing the wrong one would be worse than
 *        accepting both. Pass sri/hashMiB explicitly to be unambiguous.
 * @param {string}   [opts.sri]      SRI string (default CDN.sri; null disables)
 * @param {number}   [opts.hashMiB]  UCI Hash size, default 16 - see above
 * @param {string}   [opts.fallbackWasmUrl] same-origin copy
 * @param {function} [opts.onFallback] called when the CDN copy isn't used
 * @param {number}   [opts.bootTimeoutMs]
 * @param {number}   [opts.analysisTimeoutMs]
 */
export async function createEngine(opts = {}) {
  const {
    wasmUrl = CDN.wasmUrl,
    jsUrl = CDN.jsUrl,
    hash,
    fallbackWasmUrl = CDN.localWasmUrl,
    onFallback = null,
    bootTimeoutMs = 60000,
    analysisTimeoutMs = 120000,
  } = opts

  const integrity = typeof hash === "string" && hash
    ? hash
    : ("sri" in opts ? opts.sri : CDN.sri)
  const hashMiB = typeof hash === "number" && isFinite(hash)
    ? hash
    : (typeof opts.hashMiB === "number" ? opts.hashMiB : DEFAULT_HASH_MIB)

  const acquired = await acquireWasm({ wasmUrl, integrity, fallbackWasmUrl, onFallback })

  try {
    return await bootEngine({ jsUrl, acquired, hashMiB, bootTimeoutMs, analysisTimeoutMs })
  } catch (err) {
    if (acquired.revoke) acquired.revoke()
    if (acquired.source !== "local" && acquired.url !== fallbackWasmUrl) {
      // The bytes verified but the engine never came up: a browser that can't
      // instantiate a blob: wasm, an out-of-memory tab, a stalled worker. The
      // same-origin copy is worth one try before giving up on the visitor.
      reportFallback(onFallback, {
        stage: "boot",
        reason: "engine-did-not-start",
        url: wasmUrl,
        fallbackUrl: fallbackWasmUrl,
        error: String(err && err.message || err),
      })
      return bootEngine({
        jsUrl,
        acquired: { url: fallbackWasmUrl, source: "local", revoke: null },
        hashMiB, bootTimeoutMs, analysisTimeoutMs,
      })
    }
    throw err
  }
}

function bootEngine({ jsUrl, acquired, hashMiB, bootTimeoutMs, analysisTimeoutMs }) {
  // The vendored loader takes its .wasm URL from its own location.hash:
  //   u = decodeURIComponent(location.hash.substr(1).split(",")[0])
  // (see the worker branch at the end of stockfish-18-lite-single.js). That is
  // the only injection point it offers, so it is the one we use.
  const url = new URL(jsUrl, import.meta.url)
  url.hash = encodeURIComponent(acquired.url)

  const worker = new Worker(url.href)

  // ---- state ----
  // Exactly one job owns the engine's output at a time. `active` is that job;
  // every other caller waits in `queue`. Interleaved MultiPV output from two
  // overlapping searches is the classic bug in this wrapper, and it does not
  // announce itself: you get a plausible line list with two positions' moves
  // mixed together.
  let active = null
  let queue = Promise.resolve()
  let dead = null            // Error once the engine can no longer be trusted
  let currentMultipv = null  // what the engine's MultiPV option is set to
  let blobRevoked = false

  function revokeBlob() {
    if (!blobRevoked && acquired.revoke) { acquired.revoke(); blobRevoked = true }
  }

  function fail(err) {
    dead = err instanceof Error ? err : new Error(String(err))
    const job = active
    active = null
    if (job) { clearTimeout(job.timer); job.reject(dead) }
    try { worker.terminate() } catch {}
    revokeBlob()
  }

  function handleLine(line) {
    if (!line) return
    const job = active
    if (!job) return

    if (job.kind === "boot") {
      if (line === "uciok") { job.onUciok(line); return }
      if (line === "readyok") { job.finish(); return }
      job.uciLines.push(line)
      return
    }

    // analyse
    if (line.startsWith("info ")) {
      const info = parseInfo(line)
      if (!info) return
      // Iterative deepening prints every rank once per iteration; depth only
      // ever rises within one `go`, so last-write-wins per rank leaves the
      // deepest completed iteration - which is what python-chess hands
      // analyse.py when it returns from a depth limit.
      job.lines.set(info.multipv, info)
      return
    }
    if (line.startsWith("bestmove")) {
      const bestmove = line.split(" ")[1] || null
      job.finish(bestmove)
    }
  }

  worker.onmessage = (e) => {
    const d = e && e.data
    handleLine(typeof d === "string" ? d : (d && typeof d.data === "string" ? d.data : ""))
  }
  worker.onerror = (e) => {
    fail(new Error("engine worker error: " + (e && (e.message || e.type) || "unknown")))
  }
  if (typeof worker.onmessageerror !== "undefined") {
    worker.onmessageerror = () => fail(new Error("engine worker message error"))
  }

  function send(cmd) {
    worker.postMessage(cmd)
  }

  // Run `task` with exclusive ownership of the engine. Failures don't poison
  // the chain - the next caller still gets its turn.
  function enqueue(task) {
    const run = queue.then(() => {}, () => {}).then(() => {
      if (dead) throw dead
      return task()
    })
    queue = run.then(() => {}, () => {})
    return run
  }

  const boot = enqueue(() => new Promise((resolve, reject) => {
    const job = {
      kind: "boot",
      uciLines: [],
      reject,
      timer: setTimeout(() => {
        fail(new Error(`engine did not boot within ${bootTimeoutMs}ms (wasm: ${acquired.source})`))
      }, bootTimeoutMs),
      onUciok() {
        // Threads is capped at 1 in this build; setting it anyway documents the
        // intent and keeps the command list honest if the binary is swapped.
        send("setoption name Threads value 1")
        send(`setoption name Hash value ${hashMiB}`)
        // Full strength. Humanness comes from the fitted model choosing among
        // the candidates, never from crippling the search - a weakened search
        // produces a DIFFERENT candidate set, which is the one thing the model
        // cannot be fitted against.
        send("setoption name Skill Level value 20")
        send("ucinewgame")
        send("isready")
      },
      finish() {
        clearTimeout(job.timer)
        active = null
        // The wasm is instantiated by the time readyok lands, so the 7 MB blob
        // can go back to the allocator. On a phone that matters.
        revokeBlob()
        resolve(job.uciLines)
      },
    }
    active = job
    send("uci")
  }))

  const engine = {
    // which copy of the wasm actually loaded: "cdn" | "local" | "direct"
    source: acquired.source,
    wasmUrl: acquired.url,
    hashMiB,
    ready: boot,

    /**
     * @param {string} fen
     * @param {number} depth    e.g. 5 (candidate pool) or 2 (what a human sees)
     * @param {number} multipv  e.g. 20
     * @returns {Promise<Array<{uci, cp, mate, pv, multipv, depth}>>} best first.
     *   cp is from the SIDE TO MOVE's point of view, mate folded to +-10000
     *   exactly as pipeline/analyse.py stores it. `mate` is the raw mate
     *   distance (null when the score is a real centipawn score).
     */
    analyse(fen, depth = CALIBRATION_DEPTH, multipv = CALIBRATION_MULTIPV) {
      if (typeof fen !== "string" || !fen.trim()) return Promise.reject(new Error("analyse: fen required"))
      const d = Math.max(1, Math.floor(depth))
      // Clamped to the engine's own `option name MultiPV ... max 256`. Without
      // the clamp a larger value would be rejected by the engine while this
      // wrapper recorded it as set, and the NEXT call asking for the same width
      // would skip the setoption and silently search at the old width.
      const n = Math.min(256, Math.max(1, Math.floor(multipv)))
      return enqueue(() => new Promise((resolve, reject) => {
        const job = {
          kind: "analyse",
          lines: new Map(),
          reject,
          timer: setTimeout(() => {
            // This build has no ASYNCIFY (checked: no Asyncify symbols in the
            // binary), so `go` blocks the worker thread outright and a `stop`
            // sent now would not be read until the search ended anyway. The
            // only real recovery is to kill the worker.
            fail(new Error(`analysis timed out after ${analysisTimeoutMs}ms: ${fen}`))
          }, analysisTimeoutMs),
          finish(bestmove) {
            clearTimeout(job.timer)
            active = null
            // Checkmate and stalemate: "bestmove (none)", no info lines.
            if (!job.lines.size && (!bestmove || bestmove === "(none)")) return resolve([])
            const out = [...job.lines.entries()]
              .sort((a, b) => a[0] - b[0])
              .map(([, v]) => v)
            resolve(out)
          },
        }
        active = job
        if (currentMultipv !== n) {
          send(`setoption name MultiPV value ${n}`)
          currentMultipv = n
        }
        send(`position fen ${fen}`)
        send(`go depth ${d}`)
      }))
    },

    /**
     * Time real positions at the depth and width the page actually uses, so it
     * can decide how many games fit a budget (~60s total on a phone).
     * @returns {Promise<{msPerPosition, samples, totalMs, depth, multipv}>}
     *   msPerPosition is the MEDIAN of the timed positions with the first one
     *   dropped: the first search pays for JIT warm-up and a cold TT, and the
     *   mean of a small sample is at the mercy of one scheduler hiccup.
     */
    async calibrate({ fens = CALIBRATION_FENS, depth = CALIBRATION_DEPTH, multipv = CALIBRATION_MULTIPV } = {}) {
      await boot
      const samples = []
      const t0 = now()
      for (const fen of fens) {
        const t = now()
        await engine.analyse(fen, depth, multipv)
        samples.push(now() - t)
      }
      const totalMs = now() - t0
      const timed = samples.length > 1 ? samples.slice(1) : samples
      const sorted = [...timed].sort((a, b) => a - b)
      const mid = Math.floor(sorted.length / 2)
      const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
      return {
        msPerPosition: Math.round(median * 10) / 10,
        samples: samples.map(s => Math.round(s * 10) / 10),
        totalMs: Math.round(totalMs),
        depth,
        multipv,
      }
    },

    quit() {
      if (dead) { revokeBlob(); return Promise.resolve() }
      dead = new Error("engine quit")
      const job = active
      active = null
      if (job) { clearTimeout(job.timer); job.reject(dead) }
      try { send("quit") } catch {}
      // Give the worker a tick to close itself, then make sure.
      setTimeout(() => { try { worker.terminate() } catch {} }, 50)
      revokeBlob()
      return Promise.resolve()
    },
  }

  return boot.then(() => engine)
}
