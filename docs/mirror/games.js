// Mirror Bot — games.js
// Downloads a stranger's games from chess.com and lichess, straight from the
// browser, and hands back one normalised shape the rest of the page can fit on.
//
// Both APIs answer with `access-control-allow-origin: *` (measured on both,
// 2026-09), so there is no proxy and never needs to be one.
//
// Everything here is deliberately serial. lichess allows ONE request per IP at
// a time and answers a second one with 429 plus a minute in the corner, so all
// lichess traffic — profile lookups included — goes through a single-slot lock.
// chess.com is more forgiving but is walked serially too: a parallel archive
// walk is how you get a 1.3 MB HTML error page instead of JSON.
//
// The filtering rules are measurements, not taste. Each one has the number that
// justifies it in a comment above it. If you change a rule, re-measure first.

import { Chess } from "../vendor/chess.js"

// ---------------------------------------------------------------------------
// public shape
// ---------------------------------------------------------------------------

/** The speed vocabulary this module speaks, for both sites. */
export const SPEEDS = ["bullet", "blitz", "rapid", "classical", "daily"]

/** Sites this module knows how to read. */
export const SITES = ["chesscom", "lichess"]

/**
 * Anything this module refuses to do carries a `code` so the UI can say
 * something specific: 'bad_site' | 'bad_username' | 'not_found' | 'closed' |
 * 'rate_limited' | 'http' | 'bad_json' | 'network' | 'aborted'.
 */
export class GamesError extends Error {
  constructor(code, message, extra = {}) {
    super(message)
    this.name = "GamesError"
    this.code = code
    Object.assign(this, extra)
  }
}

// ---------------------------------------------------------------------------
// tuning constants — every one of these is a measurement
// ---------------------------------------------------------------------------

// A game under 16 plies is an opening accident: a mouse-slip, a disconnect, a
// two-move scholar's attempt. There is no style in it, only noise.
const MIN_PLIES = 16

// And a game where the player barely got past the book teaches the model about
// their opening prep, not their play. Require 8 of their own moves from ply 9
// onward (0-based index 8) before the game counts.
const OPENING_PLY = 8
const MIN_MOVES_AFTER_OPENING = 8

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

// chess.com writes per-move clocks into DAILY PGNs in units of ten seconds, not
// seconds. Measured over 9 burkeley + 9 trev4ev + 19 daily-leaderboard games
// across both 1/86400 and 1/259200: sum(clk) * 10 lands within a percent or two
// of (end_time - start_time), every time, and never above it. The excess, when
// there is any, is the loser sitting on the position before resigning — time
// that belongs to no move. So the tag is time SPENT, scaled down by ten; it is
// not a remaining-time clock at all, which is why the naive prev-minus-current
// delta goes negative on daily games. Live games are ordinary countdowns.
const DAILY_CLK_UNIT_SECONDS = 10

// lichess answers 429 with "wait a minute" and means it literally.
const LICHESS_COOLDOWN_MS = 60_000
const CHESSCOM_COOLDOWN_MS = 5_000
const RETRY_BACKOFF_MS = [1_000, 2_500, 6_000]
const MAX_ATTEMPTS = 3

// lichess's own max= overshoots (asked 30 with perfType set, got 36; asked 15,
// got 21) so it is sliced client-side. We also ask for more than we need
// because roughly a third of a real pull is dropped by the filters above.
const LICHESS_OVERASK = 3
const LICHESS_OVERASK_FLOOR = 50
const LICHESS_RAW_CEILING = 6_000

const API = {
  chesscom: "https://api.chess.com",
  lichess: "https://lichess.org",
}

// lichess rates ultraBullet and correspondence as their own pools; we fold them
// into the two buckets a player actually thinks in.
const LICHESS_PERF_TO_SPEED = {
  ultraBullet: "bullet",
  bullet: "bullet",
  blitz: "blitz",
  rapid: "rapid",
  classical: "classical",
  correspondence: "daily",
}
const SPEED_TO_LICHESS_PERFS = {
  bullet: ["ultraBullet", "bullet"],
  blitz: ["blitz"],
  rapid: ["rapid"],
  classical: ["classical"],
  daily: ["correspondence"],
}
// The pool whose rating is the headline for each speed: ultraBullet games count
// towards the bullet total but its rating is not the one anybody quotes.
const HEADLINE_LICHESS_PERF = {
  bullet: "bullet",
  blitz: "blitz",
  rapid: "rapid",
  classical: "classical",
  correspondence: "daily",
}
// chess.com has no classical pool at all — its time classes stop at daily.
const CHESSCOM_TIME_CLASSES = ["bullet", "blitz", "rapid", "daily"]

// ---------------------------------------------------------------------------
// one request at a time, per site
// ---------------------------------------------------------------------------

function makeLock() {
  let tail = Promise.resolve()
  return function acquire() {
    let release
    const gate = new Promise((res) => {
      release = res
    })
    const mine = tail.then(() => release)
    tail = tail.then(() => gate)
    return mine
  }
}

const locks = { chesscom: makeLock(), lichess: makeLock() }

const sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(abortError())
    const t = setTimeout(() => {
      signal?.removeEventListener?.("abort", onAbort)
      resolve()
    }, ms)
    function onAbort() {
      clearTimeout(t)
      reject(abortError())
    }
    signal?.addEventListener?.("abort", onAbort, { once: true })
  })

const abortError = () => new GamesError("aborted", "Cancelled")

const isAbort = (err, signal) => err?.code === "aborted" || err?.name === "AbortError" || !!signal?.aborted

function throwIfAborted(signal) {
  if (signal?.aborted) throw abortError()
}

function cooldownFor(site) {
  return site === "lichess" ? LICHESS_COOLDOWN_MS : CHESSCOM_COOLDOWN_MS
}

/**
 * Fetch with the site's lock held, retrying 429s and 5xxs. When `stream` is set
 * the caller gets the live Response and must call release() when it is done
 * with the body — a half-read lichess stream still occupies the one slot.
 */
async function request(site, url, { signal, accept, stream = false } = {}) {
  throwIfAborted(signal)
  const release = await locks[site]()
  let handedOff = false
  try {
    let lastErr = null
    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      throwIfAborted(signal)
      let res
      try {
        // No User-Agent: browsers forbid setting it, so sending one from node
        // would only make the test lie about what the page does.
        res = await fetch(url, { signal, headers: accept ? { Accept: accept } : {} })
      } catch (err) {
        if (signal?.aborted || err?.name === "AbortError") throw abortError()
        lastErr = new GamesError("network", `Could not reach ${hostOf(url)}: ${err.message}`, { url })
        await sleep(RETRY_BACKOFF_MS[Math.min(attempt, RETRY_BACKOFF_MS.length - 1)], signal)
        continue
      }

      if (res.status === 404) {
        throw new GamesError("not_found", `${hostOf(url)} has no such user or archive`, { url, status: 404 })
      }
      if (res.status === 429) {
        const retryAfter = Number(res.headers.get("retry-after"))
        const waitMs = Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : cooldownFor(site)
        lastErr = new GamesError("rate_limited", `${hostOf(url)} asked us to slow down`, { url, waitMs })
        if (attempt === MAX_ATTEMPTS - 1) break
        await sleep(waitMs, signal)
        continue
      }
      if (res.status >= 500) {
        lastErr = new GamesError("http", `${hostOf(url)} returned ${res.status}`, { url, status: res.status })
        await sleep(RETRY_BACKOFF_MS[Math.min(attempt, RETRY_BACKOFF_MS.length - 1)], signal)
        continue
      }
      if (!res.ok) {
        throw new GamesError("http", `${hostOf(url)} returned ${res.status}`, { url, status: res.status })
      }

      if (stream) {
        handedOff = true
        return { res, release }
      }
      return await res.text()
    }
    throw lastErr || new GamesError("network", `Gave up on ${url}`, { url })
  } finally {
    if (!handedOff) release()
  }
}

function hostOf(url) {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}

/**
 * chess.com has been caught serving a 1.3 MB HTML error page where an archive
 * should be, with a 200 on it. Never hand a response body to JSON.parse without
 * this, and never let the raw SyntaxError reach the UI.
 */
function parseJson(text, url) {
  try {
    return JSON.parse(text)
  } catch {
    const head = text.slice(0, 80).replace(/\s+/g, " ")
    throw new GamesError("bad_json", `${hostOf(url)} sent ${text.length} bytes that are not JSON ("${head}")`, {
      url,
      bytes: text.length,
    })
  }
}

async function getJson(site, url, opts) {
  return parseJson(await request(site, url, opts), url)
}

// ---------------------------------------------------------------------------
// profiles
// ---------------------------------------------------------------------------

function cleanUsername(username) {
  const u = String(username ?? "").trim()
  if (!u || /[\s/?#]/.test(u)) {
    throw new GamesError("bad_username", `"${username}" is not a username`)
  }
  return u
}

function emptyProfile(site, username, error) {
  return {
    ok: false,
    site,
    username: String(username ?? ""),
    id: String(username ?? "").toLowerCase(),
    counts: blankCounts(),
    ratings: {},
    lastGameAt: null,
    total: 0,
    error: { code: error.code || "network", message: error.message },
  }
}

function blankCounts() {
  const c = {}
  for (const s of SPEEDS) c[s] = 0
  return c
}

/**
 * One look at a player, cheap enough to run while they are still typing.
 *
 * chess.com needs two requests (/stats for the pools, /pub/player/{u} for the
 * display spelling of the name); lichess needs one, plus an optional one-game
 * peek because /api/user/{u} carries `seenAt` — last seen online — and no
 * last-game date at all. Pass { lastGame: false } to skip that peek.
 *
 * Never throws for an ordinary failure: check `ok`, read `error`.
 */
export async function fetchProfile(site, username, opts = {}) {
  if (!SITES.includes(site)) {
    return emptyProfile(site, username, new GamesError("bad_site", `Unknown site "${site}"`))
  }
  let user
  try {
    user = cleanUsername(username)
  } catch (err) {
    return emptyProfile(site, username, err)
  }
  const signal = opts.signal
  try {
    return site === "chesscom"
      ? await chesscomProfile(user, signal)
      : await lichessProfile(user, signal, opts.lastGame !== false)
  } catch (err) {
    if (isAbort(err, signal)) throw abortError()
    return emptyProfile(site, user, err instanceof GamesError ? err : new GamesError("network", err.message))
  }
}

async function chesscomProfile(user, signal) {
  const enc = encodeURIComponent(user)
  const profile = await getJson("chesscom", `${API.chesscom}/pub/player/${enc}`, { signal })
  const stats = await getJson("chesscom", `${API.chesscom}/pub/player/${enc}/stats`, { signal })

  const counts = blankCounts()
  const ratings = {}
  let lastGameAt = null

  for (const speed of CHESSCOM_TIME_CLASSES) {
    // The key is OMITTED, not zeroed, for a pool the player has never touched:
    // burkeley has chess_daily and chess_rapid and no chess_blitz key at all.
    const pool = stats[`chess_${speed}`]
    if (!pool) continue
    const rec = pool.record || {}
    const games = (rec.win || 0) + (rec.loss || 0) + (rec.draw || 0)
    counts[speed] = games
    if (pool.last) {
      ratings[speed] = {
        rating: pool.last.rating ?? null,
        rd: pool.last.rd ?? null,
        games,
        // chess.com publishes no provisional flag. Its Glicko RD settles under
        // about 65 once a rating is established, so that is the line drawn here
        // — a display hint, not a fact from the API. `rd` is passed through so
        // a caller who disagrees can draw their own.
        provisional: (pool.last.rd ?? 0) > 65,
      }
      if (pool.last.date) {
        const ms = pool.last.date * 1000
        if (lastGameAt == null || ms > lastGameAt) lastGameAt = ms
      }
    }
  }

  // `username` in the API payload is always lower-case; the cased spelling the
  // player chose only survives in the profile URL. 52% of names are mixed-case,
  // so showing the lower-case one looks wrong to half of everybody.
  const display = (profile.url || "").split("/").filter(Boolean).pop() || profile.username || user

  return {
    ok: true,
    site: "chesscom",
    username: display,
    id: (profile.username || user).toLowerCase(),
    counts,
    ratings,
    lastGameAt,
    total: SPEEDS.reduce((a, s) => a + counts[s], 0),
    closed: profile.status === "closed" || profile.status === "closed:fair_play_violations",
    joinedAt: profile.joined ? profile.joined * 1000 : null,
    avatar: profile.avatar || null,
  }
}

async function lichessProfile(user, signal, wantLastGame) {
  const enc = encodeURIComponent(user)
  const u = await getJson("lichess", `${API.lichess}/api/user/${enc}`, { signal })

  const counts = blankCounts()
  const ratings = {}
  for (const [perf, data] of Object.entries(u.perfs || {})) {
    const speed = LICHESS_PERF_TO_SPEED[perf]
    // perfs also carries chess960, kingOfTheHill, atomic, puzzle, racer… none
    // of which are standard chess and none of which belong in these totals.
    if (!speed || !data) continue
    counts[speed] += data.games || 0
    if (HEADLINE_LICHESS_PERF[perf] === speed && (data.games || 0) > 0) {
      ratings[speed] = {
        rating: data.rating ?? null,
        rd: data.rd ?? null,
        games: data.games || 0,
        provisional: !!data.prov,
      }
    }
  }

  let lastGameAt = null
  if (wantLastGame) {
    // /api/user/{u} has seenAt (last online) and no last-game date, so this is
    // the cheapest honest answer: one game, no moves.
    try {
      const line = await request(
        "lichess",
        `${API.lichess}/api/games/user/${enc}?max=1&moves=false&clocks=false&opening=false&evals=false`,
        { signal, accept: "application/x-ndjson" }
      )
      const first = line.split("\n").find((l) => l.trim())
      if (first) {
        const g = parseJson(first, "lichess game peek")
        lastGameAt = g.lastMoveAt || g.createdAt || null
      }
    } catch (err) {
      if (isAbort(err, signal)) throw abortError()
      // A missing last-game date is not worth failing a profile over.
    }
  }

  return {
    ok: true,
    site: "lichess",
    username: u.username || user,
    id: (u.id || user).toLowerCase(),
    counts,
    ratings,
    lastGameAt,
    total: SPEEDS.reduce((a, s) => a + counts[s], 0),
    closed: !!u.disabled || !!u.tosViolation,
    joinedAt: u.createdAt || null,
    seenAt: u.seenAt || null,
  }
}

// ---------------------------------------------------------------------------
// PGN and clocks
// ---------------------------------------------------------------------------

/** "0:05:35.1" -> 335.1, "33:36:00" -> 120960. Hours are unbounded. */
export function parseClockTag(text) {
  const parts = String(text).split(":").map(Number)
  if (parts.some((n) => !Number.isFinite(n))) return null
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  if (parts.length === 1) return parts[0]
  return null
}

/** "600+5" -> {base:600, inc:5}; "1/86400" -> {daily:true, perMove:86400}. */
export function parseTimeControl(tc) {
  const s = String(tc ?? "")
  const daily = /^1\/(\d+)$/.exec(s)
  if (daily) return { daily: true, base: Number(daily[1]), inc: 0, perMove: Number(daily[1]) }
  const live = /^(\d+)(?:\+(\d+))?$/.exec(s)
  if (live) return { daily: false, base: Number(live[1]), inc: Number(live[2] || 0), perMove: null }
  return { daily: false, base: 0, inc: 0, perMove: null }
}

/**
 * Pull SAN moves out of a PGN movetext along with any [%clk] that trails them.
 * Deliberately not a real PGN parser: chess.js validates every move downstream,
 * so anything this mis-splits gets thrown out there rather than smuggled in.
 */
export function readMovetext(pgn) {
  const body = String(pgn ?? "").replace(/^\s*\[[^\]]*\]\s*$/gm, "")
  const sans = []
  const clocks = []
  const tokens = body.match(/\{[^}]*\}|\S+/g) || []
  for (const tok of tokens) {
    if (tok[0] === "{") {
      const m = /\[%clk\s+([\d:.]+)\]/.exec(tok)
      if (m && sans.length) clocks[sans.length - 1] = parseClockTag(m[1])
      continue
    }
    if (tok === "1-0" || tok === "0-1" || tok === "1/2-1/2" || tok === "*") continue
    if (/^\$\d+$/.test(tok)) continue
    if (/^\d+\.*$/.test(tok) || /^\.+$/.test(tok)) continue
    // "12.Nf3" and "12...Nf3" arrive glued in some exports.
    const glued = /^\d+\.+(.+)$/.exec(tok)
    const san = (glued ? glued[1] : tok).replace(/[?!]+$/, "")
    if (!san || san === "1-0" || san === "0-1") continue
    sans.push(san)
  }
  while (clocks.length < sans.length) clocks.push(undefined)
  return { sans, clocks: clocks.map((c) => (c == null ? null : c)) }
}

/** Seconds spent per ply for a chess.com game. */
function chesscomSpent(clocks, tc, durationSec) {
  if (!clocks.length || clocks.every((c) => c == null)) return null

  if (tc.daily) {
    // See DAILY_CLK_UNIT_SECONDS. Calibrate against the wall clock when the API
    // gave us a start_time: ten units can never exceed the game's own duration,
    // so if it does, this game does not follow the pattern and the raw seconds
    // are the safer read.
    let unit = DAILY_CLK_UNIT_SECONDS
    const sum = clocks.reduce((a, c) => a + (c || 0), 0)
    if (durationSec != null && sum > 0 && sum * unit > durationSec * 1.25) unit = 1
    return clocks.map((c) => (c == null ? null : round2(c * unit)))
  }

  // Live games: an ordinary countdown, so spent = before - after + increment.
  const out = new Array(clocks.length).fill(null)
  const prev = [tc.base, tc.base]
  for (let i = 0; i < clocks.length; i++) {
    const cur = clocks[i]
    if (cur == null) continue
    const side = i % 2
    out[i] = round2(Math.max(0, prev[side] - cur + tc.inc))
    prev[side] = cur
  }
  return out
}

/** Seconds spent per ply for a lichess game, from centisecond remainders. */
function lichessSpent(clocksCs, clock, plies) {
  if (!Array.isArray(clocksCs) || !clocksCs.length) return null
  const initCs = (clock?.initial ?? 0) * 100
  const incCs = (clock?.increment ?? 0) * 100
  const out = []
  const prev = [initCs, initCs]
  for (let i = 0; i < plies; i++) {
    const cur = clocksCs[i]
    if (cur == null) {
      out.push(null)
      continue
    }
    const side = i % 2
    // lichess hands back a few centiseconds more than the initial clock on the
    // first move (measured: initial 18000, clocks[0] 18003 — lag compensation),
    // which would make move one cost negative time.
    out.push(round2(Math.max(0, (prev[side] - cur + incCs) / 100)))
    prev[side] = cur
  }
  return out
}

const round2 = (n) => Math.round(n * 100) / 100

// ---------------------------------------------------------------------------
// the replay gate
// ---------------------------------------------------------------------------

/**
 * Replay SAN from the standard start and return UCI, or null if any move is
 * illegal there.
 *
 * This is the backstop, not the front door. King of the Hill replays 21 out of
 * 21 games full-length on a standard board (measured on german11's KotH pool),
 * so a variant filter that trusted the replay would let every one of them in
 * and quietly poison the king-walk and check features. The allow-list catches
 * variants; this catches what the allow-list cannot see — imports, truncated
 * PGNs, from-position games that still claim variant "standard".
 */
export function replayToUci(sans) {
  const board = new Chess()
  const uci = []
  for (const san of sans) {
    let mv
    try {
      mv = board.move(san)
    } catch {
      return null
    }
    if (!mv) return null
    uci.push(mv.from + mv.to + (mv.promotion || ""))
  }
  return uci
}

/** The two length rules, applied together. */
function longEnough(moves, color) {
  if (moves.length < MIN_PLIES) return false
  const mine = color === "w" ? 0 : 1
  let after = 0
  for (let i = OPENING_PLY; i < moves.length; i++) if (i % 2 === mine) after++
  return after >= MIN_MOVES_AFTER_OPENING
}

// ---------------------------------------------------------------------------
// chess.com games
// ---------------------------------------------------------------------------

function normaliseChesscom(raw, lowerUser, opts) {
  // ALLOW-list, not a deny-list. The documented rules enum is incomplete —
  // 'oddschess' turns up in live data and is not in it — so anything that is
  // not the single word we want is gone.
  if (raw.rules !== "chess") return null

  const speed = raw.time_class
  if (!opts.speeds.includes(speed)) return null
  if (opts.ratedOnly && !raw.rated) return null
  if (raw.initial_setup && raw.initial_setup !== START_FEN) return null
  if (typeof raw.pgn !== "string" || !raw.pgn) return null

  // Case-insensitive, always. Mixed-case display names are the majority, and
  // an exact-match comparison here fails silently: every game of theirs looks
  // like somebody else's and the pull comes back empty for no visible reason.
  const white = String(raw.white?.username || "").toLowerCase()
  const black = String(raw.black?.username || "").toLowerCase()
  const color = white === lowerUser ? "w" : black === lowerUser ? "b" : null
  if (!color) return null

  const { sans, clocks } = readMovetext(raw.pgn)
  const moves = replayToUci(sans)
  if (!moves || !longEnough(moves, color)) return null

  const tc = parseTimeControl(raw.time_control)
  const duration = raw.start_time && raw.end_time ? raw.end_time - raw.start_time : null
  const me = color === "w" ? raw.white : raw.black
  const them = color === "w" ? raw.black : raw.white

  return {
    site: "chesscom",
    id: String(raw.uuid || raw.url || ""),
    moves,
    color,
    speed,
    rated: !!raw.rated,
    endTime: (raw.end_time || 0) * 1000,
    clocks: chesscomSpent(clocks.slice(0, moves.length), tc, duration),
    opening: openingFromChesscom(raw),
    // extras the fitting stage wants and would otherwise have to re-download
    url: raw.url || null,
    timeControl: raw.time_control || null,
    result: resultFor(me?.result),
    opponent: them?.username || null,
    myRating: me?.rating ?? null,
    opponentRating: them?.rating ?? null,
  }
}

const CHESSCOM_WINS = new Set(["win"])
const CHESSCOM_DRAWS = new Set(["agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"])
function resultFor(code) {
  if (!code) return null
  if (CHESSCOM_WINS.has(code)) return "win"
  if (CHESSCOM_DRAWS.has(code)) return "draw"
  return "loss"
}

function openingFromChesscom(raw) {
  const url = raw.eco || ""
  const tagged = /\[ECO "([^"]+)"\]/.exec(raw.pgn || "")
  const slug = url.split("/openings/")[1]
  if (!slug && !tagged) return null
  return {
    eco: tagged ? tagged[1] : null,
    name: slug ? slug.replace(/-/g, " ").replace(/\s+/g, " ").trim() : null,
    ply: null,
  }
}

async function* chesscomGames(username, budget, opts, report) {
  const enc = encodeURIComponent(username)
  const lowerUser = username.toLowerCase()
  const index = await getJson("chesscom", `${API.chesscom}/pub/player/${enc}/games/archives`, { signal: opts.signal })
  const archives = Array.isArray(index.archives) ? index.archives.slice().reverse() : []
  report({ phase: "archives", done: 0, total: archives.length, site: "chesscom", username })

  let kept = 0
  for (let i = 0; i < archives.length; i++) {
    throwIfAborted(opts.signal)
    let month
    try {
      month = await getJson("chesscom", archives[i], { signal: opts.signal })
    } catch (err) {
      if (isAbort(err, opts.signal)) throw abortError()
      // One bad month must not kill a ten-year pull.
      report({ phase: "warning", done: kept, total: budget, site: "chesscom", username, message: `Skipped ${archives[i]}: ${err.message}` })
      continue
    }
    const games = (Array.isArray(month.games) ? month.games : []).slice().sort((a, b) => (b.end_time || 0) - (a.end_time || 0))
    for (const raw of games) {
      const g = normaliseChesscom(raw, lowerUser, opts)
      if (!g) continue
      kept++
      yield g
      if (kept >= budget) return
    }
    report({ phase: "download", done: kept, total: budget, site: "chesscom", username, scanned: i + 1, of: archives.length })
  }
}

// ---------------------------------------------------------------------------
// lichess games
// ---------------------------------------------------------------------------

function normaliseLichess(raw, lowerId, opts) {
  // ALLOW-list again, and both halves of it matter. perfType alone is not a
  // variant filter: 13 of 40 games in a real german11 pull came back with
  // perf "blitz", speed "blitz" and variant "fromPosition", every one of them
  // carrying an initialFen.
  if (raw.variant !== "standard") return null
  if (raw.initialFen && raw.initialFen !== START_FEN) return null

  const speed = LICHESS_PERF_TO_SPEED[raw.speed] || LICHESS_PERF_TO_SPEED[raw.perf]
  if (!speed || !opts.speeds.includes(speed)) return null
  if (opts.ratedOnly && !raw.rated) return null

  const wp = raw.players?.white || {}
  const bp = raw.players?.black || {}
  if (!opts.includeBots && isBotGame(raw, wp, bp)) return null

  const white = String(wp.user?.id || wp.user?.name || "").toLowerCase()
  const black = String(bp.user?.id || bp.user?.name || "").toLowerCase()
  const color = white === lowerId ? "w" : black === lowerId ? "b" : null
  if (!color) return null

  const sans = String(raw.moves || "").split(/\s+/).filter(Boolean)
  const moves = replayToUci(sans)
  if (!moves || !longEnough(moves, color)) return null

  const me = color === "w" ? wp : bp
  const them = color === "w" ? bp : wp

  return {
    site: "lichess",
    id: String(raw.id || ""),
    moves,
    color,
    speed,
    rated: !!raw.rated,
    endTime: raw.lastMoveAt || raw.createdAt || 0,
    clocks: lichessSpent(raw.clocks, raw.clock, moves.length),
    opening: raw.opening ? { eco: raw.opening.eco || null, name: raw.opening.name || null, ply: raw.opening.ply ?? null } : null,
    url: raw.id ? `https://lichess.org/${raw.id}` : null,
    timeControl: raw.clock ? `${raw.clock.initial}+${raw.clock.increment}` : raw.daysPerTurn ? `1/${raw.daysPerTurn * 86400}` : null,
    result: raw.winner ? (raw.winner[0] === color ? "win" : "loss") : raw.status === "draw" || raw.status === "stalemate" ? "draw" : null,
    opponent: them.user?.name || (them.aiLevel != null ? `Stockfish level ${them.aiLevel}` : null),
    myRating: me.rating ?? null,
    opponentRating: them.rating ?? null,
  }
}

/**
 * Bot detection is a lichess-only problem: chess.com's public API contains no
 * computer games at all, so there is nothing to filter there. Three signals,
 * because lichess has three kinds of machine — its own Stockfish (source 'ai',
 * plus an aiLevel on the machine's side) and registered bot accounts (title
 * 'BOT', verified live on maia1).
 */
function isBotGame(raw, wp, bp) {
  if (raw.source === "ai") return true
  if (wp.aiLevel != null || bp.aiLevel != null) return true
  if (wp.user?.title === "BOT" || bp.user?.title === "BOT") return true
  return false
}

async function* ndjson(res, signal) {
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  try {
    for (;;) {
      throwIfAborted(signal)
      let chunk
      try {
        chunk = await reader.read()
      } catch (err) {
        // The body is read outside request(), so an abort or a dropped
        // connection arrives raw here and has to be normalised by hand —
        // otherwise a cancelled pull looks like an ordinary failure and comes
        // back as a half-finished result instead of a cancellation.
        if (signal?.aborted || err?.name === "AbortError") throw abortError()
        throw new GamesError("network", `lichess cut the stream short: ${err.message}`)
      }
      const { done, value } = chunk
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let nl
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim()
        buf = buf.slice(nl + 1)
        if (line) yield line
      }
    }
    buf += decoder.decode()
    if (buf.trim()) yield buf.trim()
  } finally {
    try {
      await reader.cancel()
    } catch {
      /* the consumer walked away mid-stream; nothing to do */
    }
  }
}

async function* lichessGames(username, budget, opts, report) {
  const enc = encodeURIComponent(username)
  const lowerId = username.toLowerCase()

  const perfs = []
  for (const s of opts.speeds) for (const p of SPEED_TO_LICHESS_PERFS[s] || []) perfs.push(p)

  // Ask for more than we need — a third of a real pull falls to the filters —
  // then slice client-side, because max= overshoots whenever perfType is set.
  const rawTarget = Math.min(LICHESS_RAW_CEILING, Math.max(budget * LICHESS_OVERASK, budget + LICHESS_OVERASK_FLOOR))

  const params = new URLSearchParams({
    max: String(rawTarget),
    moves: "true",
    clocks: "true",
    opening: "true",
    evals: "false",
    accuracy: "false",
    division: "false",
    sort: "dateDesc",
  })
  // Always send perfType, even when every speed is wanted: it is what keeps
  // chess960, King of the Hill, crazyhouse and the rest off the wire. It does
  // NOT keep fromPosition games out — that is the allow-list's job.
  if (perfs.length) params.set("perfType", perfs.join(","))
  if (opts.ratedOnly) params.set("rated", "true")

  const url = `${API.lichess}/api/games/user/${enc}?${params}`
  const { res, release } = await request("lichess", url, {
    signal: opts.signal,
    accept: "application/x-ndjson",
    stream: true,
  })

  let kept = 0
  let seen = 0
  try {
    for await (const line of ndjson(res, opts.signal)) {
      // The client-side slice. lichess answered a max=30 request with 36 games
      // and a max=15 request with 21; without this the caller's cap is a lie.
      if (seen >= rawTarget) break
      seen++
      let raw
      try {
        raw = JSON.parse(line)
      } catch {
        report({ phase: "warning", done: kept, total: budget, site: "lichess", username, message: "Skipped a line lichess did not send as JSON" })
        continue
      }
      const g = normaliseLichess(raw, lowerId, opts)
      if (!g) continue
      kept++
      yield g
      if (kept >= budget) return
      if (kept % 25 === 0) report({ phase: "download", done: kept, total: budget, site: "lichess", username, scanned: seen })
    }
  } finally {
    // Releases the one lichess slot whether we finished, hit the cap, or the
    // caller aborted. Without this a cancelled pull locks out the next one.
    release()
  }
}

// ---------------------------------------------------------------------------
// fetchGames
// ---------------------------------------------------------------------------

/**
 * Download and normalise a player's games.
 *
 * opts: { accounts: [{site, username}], speeds, ratedOnly, includeBots, max, signal }
 * onProgress: ({ phase, done, total, ... }) with phase one of
 *   'archives' | 'download' | 'warning' | 'done'.
 *
 * Accounts are drained one at a time, never in parallel — two overlapping
 * lichess streams is exactly the 429 the docs warn about. Each account gets an
 * even share of what is still missing, so a thin first account is made up for
 * by the second rather than leaving the caller short.
 *
 * Returns games newest-first. The array also carries a `warnings` property
 * listing anything that was skipped along the way.
 */
export async function fetchGames(opts = {}, onProgress) {
  const accounts = (opts.accounts || []).filter((a) => a && SITES.includes(a.site) && String(a.username || "").trim())
  const speeds = (opts.speeds && opts.speeds.length ? opts.speeds : SPEEDS).filter((s) => SPEEDS.includes(s))
  const max = Math.max(0, Math.floor(opts.max ?? 400))
  const settings = {
    speeds,
    ratedOnly: opts.ratedOnly !== false,
    includeBots: !!opts.includeBots,
    signal: opts.signal,
  }
  const warnings = []
  const report = (msg) => {
    if (msg.phase === "warning") warnings.push(msg.message)
    if (typeof onProgress !== "function") return
    try {
      onProgress(msg)
    } catch {
      /* a UI that throws in its own progress handler is not our problem */
    }
  }

  const out = []
  let left = accounts.length
  for (const acct of accounts) {
    throwIfAborted(settings.signal)
    const need = max - out.length
    if (need <= 0) break
    const budget = Math.ceil(need / left)
    left--
    let username = String(acct.username).trim()
    try {
      username = cleanUsername(acct.username)
      const stream =
        acct.site === "chesscom"
          ? chesscomGames(username, budget, settings, report)
          : lichessGames(username, budget, settings, report)
      for await (const g of stream) out.push(g)
    } catch (err) {
      if (isAbort(err, settings.signal)) throw abortError()
      report({
        phase: "warning",
        done: out.length,
        total: max,
        site: acct.site,
        username,
        message: `${acct.site}/${username}: ${err.message}`,
      })
    }
    report({ phase: "download", done: out.length, total: max, site: acct.site, username })
  }

  out.sort((a, b) => b.endTime - a.endTime)
  report({ phase: "done", done: out.length, total: out.length })
  out.warnings = warnings
  return out
}
