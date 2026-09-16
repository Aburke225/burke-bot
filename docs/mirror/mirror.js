// Mirror Bot - the page controller.
//
// It owns the four views (setup, building, result, board), the routing between
// them, and the wiring to the five modules that do the actual work:
//   games.js   fetching a stranger's games from chess.com and lichess
//   engine.js  Stockfish WASM in a Worker
//   v9.js      the 58-feature contract, shared with the trainer and Burke Bot
//   train.js   the conditional-logit fit
//   share.js   packing a whole bot into a URL fragment
//
// The bot IS the URL. There is no server and nothing is stored anywhere, so
// "play your own bot" and "open a link someone sent you" are the same code path
// reading the same fragment; they differ only in one line of copy.

import * as Games from "./games.js"
import * as Share from "./share.js"
import { Chess } from "../vendor/chess.js"

const $ = (id) => document.getElementById(id)
const SPEED_LABEL = {
  ultrabullet: "UltraBullet", bullet: "Bullet", blitz: "Blitz",
  rapid: "Rapid", classical: "Classical", daily: "Daily",
}
// chess.com has no ultrabullet or classical; lichess has both. Only speeds the
// player actually has any games in are ever rendered.
const SPEED_ORDER = ["ultrabullet", "bullet", "blitz", "rapid", "classical", "daily"]

const state = {
  profiles: {},        // site -> profile from Games.fetchProfile
  counts: {},          // speed -> pooled game count
  speeds: new Set(),
  msPerPosition: null, // from engine.calibrate()
  cap: 400,            // device-calibrated ceiling
  bot: null,           // { weights, book, levers, stats, meta }
  engine: null,
  game: null,          // the live play session
  abort: null,
  muted: false,
  sliderTouched: false,
}

/* ---------------------------------------------------------------- routing */

const VIEWS = ["setup", "building", "result", "shared"]

function show(view, opts = {}) {
  for (const v of VIEWS) $(v).hidden = v !== view
  document.body.dataset.route = view
  // Someone who built this bot was not sent it by anyone.
  document.querySelector("#shared .eyebrow").hidden = !!opts.own
  if (opts.top) window.scrollTo(0, 0)
}

/* ------------------------------------------------------------ boot veil */

function lift() {
  document.documentElement.classList.remove("booting")
}

/* --------------------------------------------------------------- theme */

const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'

function setTheme(light) {
  const root = document.documentElement
  if (light) root.setAttribute("data-theme", "light")
  else root.removeAttribute("data-theme")
  const b = $("theme-toggle")
  b.innerHTML = light ? MOON : SUN
  b.setAttribute("aria-label", light ? "Switch to dark mode" : "Switch to light mode")
  b.title = b.getAttribute("aria-label")
}

/* --------------------------------------------------------------- sound */

const SPK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/></svg>'
const MUTE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M22 9l-6 6M16 9l6 6"/></svg>'

function setMuted(m) {
  state.muted = m
  const b = $("sound-toggle")
  b.innerHTML = m ? MUTE : SPK
  b.setAttribute("aria-label", m ? "Unmute sounds" : "Mute sounds")
  b.title = b.getAttribute("aria-label")
  if (state.game) state.game.setMuted(m)
  try { localStorage.setItem("mirror-sound", m ? "off" : "on") } catch (e) {}
}

/* ------------------------------------------------------- the copy control */

const CLIP_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
const TICK_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4 12 9 17 20 6" pathLength="100"/></svg>'

function shareUrl() {
  const frag = state.bot && state.bot.fragment
  if (!frag) return location.href
  return location.origin + location.pathname + "#" + frag
}

async function copyShare() {
  const url = shareUrl()
  try {
    await navigator.clipboard.writeText(url)
    return true
  } catch (e) {
    // clipboard needs a secure context and permission; fall back rather than
    // failing silently, exactly as the parent site does
    try {
      const ta = document.createElement("textarea")
      ta.value = url
      ta.style.cssText = "position:fixed;opacity:0"
      document.body.appendChild(ta)
      ta.select()
      const ok = document.execCommand("copy")
      ta.remove()
      return ok
    } catch (e2) { return false }
  }
}

function wireCopyButtons() {
  const copy = $("copy-link")
  copy.innerHTML = CLIP_ICON
  let t = null
  copy.addEventListener("click", async () => {
    const ok = await copyShare()
    copy.innerHTML = ok ? TICK_ICON : CLIP_ICON
    copy.classList.toggle("done", ok)
    copy.title = ok ? "Copied" : "Could not copy"
    clearTimeout(t)
    t = setTimeout(() => {
      copy.innerHTML = CLIP_ICON
      copy.classList.remove("done")
      copy.title = "Copy the link"
    }, 1400)
  })

  const share = $("share-bot")
  const label = share.querySelector("span")
  let st = null
  share.addEventListener("click", async () => {
    const ok = await copyShare()
    if (!ok) { label.textContent = "Could not copy"; return }
    share.classList.add("done")
    share.querySelector("svg").outerHTML = TICK_ICON
    label.textContent = "Link copied"
    clearTimeout(st)
    st = setTimeout(() => {
      share.classList.remove("done")
      share.querySelector("svg").outerHTML = CLIP_ICON
      label.textContent = "Share this bot"
    }, 1400)
  })
}

/* --------------------------------------------------- account lookup + speeds */

let lookupTimer = null

function scheduleLookup() {
  clearTimeout(lookupTimer)
  lookupTimer = setTimeout(runLookup, 450)
}

async function runLookup() {
  const wanted = [
    { site: "chesscom", username: $("cc").value.trim(), ok: $("cc-ok") },
    { site: "lichess", username: $("li").value.trim(), ok: $("li-ok") },
  ].filter(a => a.username)

  for (const el of [$("cc-ok"), $("li-ok")]) el.hidden = true
  state.profiles = {}
  const msg = $("acct-msg")
  msg.hidden = true
  msg.className = "note"

  if (!wanted.length) {
    $("card-speeds").hidden = true
    $("card-count").hidden = true
    return
  }

  const failures = []
  for (const a of wanted) {
    try {
      const p = await Games.fetchProfile(a.site, a.username, { wantLastGame: true })
      if (p && p.ok) { state.profiles[a.site] = p; a.ok.hidden = false }
      else failures.push(`${a.site}: ${a.username} not found`)
    } catch (err) {
      failures.push(`${a.site}: ${err.message}`)
    }
  }

  if (failures.length) {
    msg.textContent = failures.join(" · ")
    msg.className = "note bad"
    msg.hidden = false
  }
  if (!Object.keys(state.profiles).length) {
    $("card-speeds").hidden = true
    $("card-count").hidden = true
    return
  }
  renderSpeeds()
}

function renderSpeeds() {
  // Pool the counts across whichever accounts were given, and show only the
  // speeds the player has actually played. A row reading "0" is noise.
  const counts = {}
  for (const p of Object.values(state.profiles)) {
    for (const [sp, n] of Object.entries(p.counts || {})) {
      if (n > 0) counts[sp] = (counts[sp] || 0) + n
    }
  }
  state.counts = counts

  const box = $("speeds")
  box.innerHTML = ""
  state.speeds = new Set()
  for (const sp of SPEED_ORDER) {
    const n = counts[sp]
    if (!n) continue
    state.speeds.add(sp)            // present means on, per the agreed rule
    const lab = document.createElement("label")
    lab.className = "speed"
    lab.innerHTML =
      `<input type="checkbox" data-speed="${sp}" checked>` +
      `<span class="nm">${SPEED_LABEL[sp] || sp}</span>` +
      `<span class="ct">${n.toLocaleString("en-US")}</span>`
    box.appendChild(lab)
  }
  box.addEventListener("change", onSpeedChange, { once: true })

  // the bots control is lichess-only: chess.com's public API has no computer
  // games at all, so offering it there would describe a filter that does nothing
  $("bots-wrap").hidden = !state.profiles.lichess
  $("card-speeds").hidden = false
  $("card-count").hidden = false
  updateSlider()
}

function onSpeedChange(ev) {
  const box = $("speeds")
  box.addEventListener("change", onSpeedChange, { once: true })
  const cb = ev.target.closest("input[type=checkbox]")
  if (cb) {
    cb.closest(".speed").classList.toggle("off", !cb.checked)
    if (cb.checked) state.speeds.add(cb.dataset.speed)
    else state.speeds.delete(cb.dataset.speed)
  }
  updateSlider()
}

/* ------------------------------------------------------------ the slider */

function selectedTotal() {
  let n = 0
  for (const sp of state.speeds) n += state.counts[sp] || 0
  return n
}

// The ladder, as data. BEST_FROM is read by the slider so its opening value is
// the smallest number of games that earns the top verdict - the two can never
// drift apart the way a hardcoded default would.
const BANDS = [
  { from: 0, q: "not really you", cls: "bad" },
  { from: 25, q: "a little like you", cls: "rough" },
  { from: 50, q: "recognisably you", cls: "ok" },
  { from: 120, q: "definitely you", cls: "good" },
  { from: 250, q: "unmistakably you", cls: "best" },
]
const BEST_FROM = BANDS[BANDS.length - 1].from
// Below this there is genuinely nothing to fit: 10 games is about 277 decision
// points against 58 features, and fewer is under five examples per parameter.
// The honest answer is to refuse rather than hand back noise with a rating on it.
const MIN_GAMES = 10

function quality(n) {
  let hit = BANDS[0]
  for (const b of BANDS) if (n >= b.from) hit = b
  return hit
}

function prettyTime(secs) {
  if (secs < 60) return Math.max(1, Math.round(secs)) + "s"
  const m = Math.floor(secs / 60), s = Math.round(secs % 60)
  return s ? `${m}m ${s}s` : `${m}m`
}

function grandTotal() {
  return Object.values(state.counts).reduce((a, b) => a + b, 0)
}

// Distinguishes "you have almost no games" from "you have plenty, just not in
// the types you ticked" - the same number means two different things and only
// one of them is the user's to fix right now.
function shortfallMessage(total, grand) {
  const names = Object.values(state.profiles).map(p => titleCase(p.username))
  const who = names.length === 1 ? names[0] : names.join(" and ")
  if (grand < MIN_GAMES) {
    const n = grand === 0 ? "no games" : `only ${grand} game${grand === 1 ? "" : "s"}`
    return {
      text: `${who} has ${n} we can read. Mirror Bot needs at least ${MIN_GAMES} ` +
            `to build anything worth calling a bot of you.`,
      cls: "note bad",
    }
  }
  const spare = grand - total
  return {
    text: `Only ${total} game${total === 1 ? "" : "s"} in the types you have picked. ` +
          `Mirror Bot needs at least ${MIN_GAMES} to build a bot of you &mdash; ` +
          `tick another type, there ${spare === 1 ? "is" : "are"} ${spare} more.`,
    cls: "note warn",
  }
}

function updateSlider() {
  const total = selectedTotal()
  const grand = grandTotal()
  const msg = $("count-msg")

  // One predicate drives the slider, the assurances and the button together.
  // Promises about how carefully we handle a build are noise next to a build
  // that cannot happen.
  const blocked = total < MIN_GAMES
  const assure = document.querySelector("#setup .assure")

  // at boot there is no account yet - nothing to say and nothing to reveal
  if (!Object.keys(state.profiles).length) {
    msg.hidden = true
    if (assure) assure.hidden = true
    $("build").disabled = true
    return
  }

  if (assure) assure.hidden = blocked
  $("slider-wrap").hidden = blocked
  $("build").disabled = blocked

  if (blocked) {
    // say what is wrong and, where there is one, what would fix it
    const m = shortfallMessage(total, grand)
    msg.innerHTML = m.text
    msg.className = m.cls
    msg.hidden = false
    $("card-count").hidden = false
    return
  }
  msg.hidden = true

  const slider = $("games")
  const max = Math.max(10, Math.min(total, state.cap))
  const wasAtFloor = +slider.max <= 10
  slider.max = String(max)
  if (wasAtFloor && !state.sliderTouched) {
    // first real counts: open at a sensible default rather than the floor the
    // empty state clamped us to
    // open at the fewest games that still reads "unmistakably you", or at
    // everything they have if that is less
    slider.value = String(Math.min(BEST_FROM, max))
  } else if (+slider.value > max) {
    slider.value = String(max)
  }
  $("tick-max").textContent = total > max
    ? `${max.toLocaleString("en-US")} max here`
    : `all ${total.toLocaleString("en-US")}`

  const n = +slider.value
  // 27.7 decision points per game, two searches each
  const ms = state.msPerPosition || 11
  const secs = (n * 27.7 * ms) / 1000
  const e = quality(n)
  $("readout").innerHTML =
    `<b>${n.toLocaleString("en-US")}</b> games` +
    `<span class="sep">&middot;</span><b>${prettyTime(secs)}</b>` +
    `<span class="sep">&middot;</span><span class="q ${e.cls}">${e.q}</span>`
  $("build").disabled = n < 10
}

/* ------------------------------------------------------------ the build */

async function ensureEngine() {
  if (state.engine) return state.engine
  const { createEngine } = await import("./engine.js")
  state.engine = await createEngine({})
  try {
    const cal = await state.engine.calibrate()
    state.msPerPosition = cal.msPerPosition
    // iOS suspends a page the moment the user switches apps, so a phone run has
    // to fit inside one uninterrupted look at the screen. ~60s of engine time.
    const phone = window.matchMedia("(max-width: 860px)").matches
    if (phone) {
      state.cap = Math.max(25, Math.min(150, Math.round(60000 / (state.msPerPosition * 27.7))))
      const note = $("mobile-note")
      note.textContent = "To train on all of your games, use a desktop or laptop."
      note.hidden = false
    } else {
      state.cap = Math.max(50, Math.round(120000 / (state.msPerPosition * 27.7)))
    }
    updateSlider()
  } catch (e) { /* calibration is an optimisation, not a requirement */ }
  return state.engine
}

async function build() {
  const accounts = Object.entries(state.profiles).map(([site, p]) => ({ site, username: p.username }))
  if (!accounts.length) return

  const n = +$("games").value
  const who = accounts.map(a => a.username).join(" + ")
  const cap = titleCase(accounts[0].username)
  $("recap-building").innerHTML =
    `<b>${cap}</b> <span>&middot;</span> <b>${n}</b> games <span>&middot;</span> ` +
    ($("rated").checked ? "rated only" : "rated and casual")
  show("building", { top: true })

  state.abort = new AbortController()
  try {
    const engine = await ensureEngine()
    const { buildBot } = await import("./build.js")
    const bot = await buildBot({
      accounts,
      speeds: [...state.speeds],
      ratedOnly: $("rated").checked,
      includeBots: $("bots").checked,
      maxGames: n,
      signal: state.abort.signal,
      engine,
    }, onProgress)
    await finish(bot, accounts)
  } catch (err) {
    if (err && (err.name === "AbortError" || /abort/i.test(err.message || ""))) {
      show("setup", { top: true })
      return
    }
    $("phase").textContent = "Could not build that bot: " + (err.message || err)
    $("left").textContent = ""
  }
}

function onProgress(p) {
  if (p.label) $("phase").textContent = p.label
  if (p.total) {
    const pct = Math.max(0, Math.min(100, Math.round((p.done / p.total) * 100)))
    $("fill").style.width = pct + "%"
  }
  if (typeof p.secondsLeft === "number") $("left").textContent = prettyTime(p.secondsLeft) + " left"
}

function titleCase(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

async function finish(bot, accounts) {
  // whichever account played most recently is the one the bot is named for
  let best = accounts[0], bestAt = -1
  for (const a of accounts) {
    const at = (state.profiles[a.site] || {}).lastGameAt || 0
    if (at > bestAt) { bestAt = at; best = a }
  }
  const name = titleCase(best.username)
  const profileUrl = best.site === "chesscom"
    ? `https://www.chess.com/member/${best.username}`
    : `https://lichess.org/@/${best.username}`

  // The levers the probe measured for THIS player have to travel with the bot.
  // play.js reads meta.poolDepth / horizonDepth / multipv / temp and otherwise
  // falls back to its own constants, which would quietly hand every user the
  // settings fitted to one 1300-rated player.
  const lv = bot.levers || {}
  // The probe deliberately does NOT pick the horizon - it computes depths 1, 2
  // and 3 and lets the regulariser choose, so the winner lands on fit.horizon
  // and levers.horizon stays null. Reading only levers left horizonDepth
  // undefined, and play.js would then fall back to its own 2 while the model
  // had been fitted at 1 or 3 - every score feature computed at a depth the
  // weights were never trained against, silently.
  const horizon = lv.horizonDepth || lv.horizon || (bot.fit && bot.fit.horizon) || null
  bot.meta = {
    username: best.username, site: best.site, name, profileUrl,
    poolDepth: lv.poolDepth, horizonDepth: horizon,
    multipv: lv.multipv, temp: lv.temp,
    rating: bot.stats.botRating,
  }
  bot.fragment = await Share.encodeAsync({
    version: Share.FORMAT_VERSION,
    weights: bot.weights,
    book: bot.book,
    meta: {
      u: best.username, s: best.site,
      r: bot.stats.botRating, g: bot.stats.gamesUsed, v: "9.3",
      pd: lv.poolDepth, hd: horizon, mp: lv.multipv, t: lv.temp,
    },
  })
  state.bot = bot

  $("bot-name").textContent = `${name} Bot`
  $("recap-result").innerHTML = $("recap-building").innerHTML.replace(
    /$/, ` <span>&middot;</span> <a href="#" id="again-link">build another bot</a>`)
  $("st-rating").textContent = bot.stats.botRating ? bot.stats.botRating.toLocaleString("en-US") : "—"
  $("st-top1").textContent = bot.stats.top1 != null ? Math.round(bot.stats.top1 * 100) + "%" : "—"
  // favouriteOpening is an object - { name, colour, games, ofGames, share } -
  // and only the family name belongs in the tile. Reading it as a string put
  // "[object Object]" on screen.
  const fav = bot.stats.favouriteOpening
  $("st-opening").textContent = (fav && fav.name) || "—"
  $("share-url").textContent = shareUrl().replace(/^https?:\/\//, "")
  const again = $("again-link")
  if (again) again.addEventListener("click", (e) => { e.preventDefault(); show("setup", { top: true }) })

  show("result", { top: true })
}

/* ------------------------------------------------------------- the board */

async function openBoard(own) {
  const bot = state.bot
  if (!bot) return
  $("nm-top").innerHTML =
    `${bot.meta.name} Bot <span class="rat">(${bot.stats.botRating ? bot.stats.botRating.toLocaleString("en-US") : "—"})</span>`
  $("from-shared").innerHTML =
    `Same machinery as <a href="../">Burke&nbsp;Bot</a> &mdash; pointed at ` +
    `<a href="${bot.meta.profileUrl}" rel="noopener">${bot.meta.name}&rsquo;s</a> games instead of mine.`

  show("shared", { own, top: true })
  if (own) {
    // land on the same URL a shared link carries, so copying the address bar
    // works and a refresh keeps the bot
    try { history.replaceState(null, "", "#" + bot.fragment) } catch (e) {}
  }

  const engine = await ensureEngine()
  const { createGame } = await import("./play.js")
  if (state.game) state.game.destroy()
  // buildBot returns the book twice: `book` is the packed trie destined for the
  // URL, `bookMap` is the lookup table. play.js wants the lookup table. A book
  // that came from a shared link is a RAW trie and has to be hydrated first -
  // handing either the packed or the raw form straight over builds a book keyed
  // "undefined" whose every lookup misses in silence.
  const playBook = bot.bookMap
    ? bot.bookMap
    : (bot.book && Array.isArray(bot.book.kids)
        ? Share.hydrateTrie(bot.book, { Chess })
        : bot.book)

  state.game = await createGame({
    boardEl: $("board"),
    evalBarEl: $("eval-bar"), evalFillEl: $("eval-fill"), evalNumEl: $("eval-num"),
    topRow: { capsEl: $("cap-top") },   // nameEl withheld on purpose - see below
    bottomRow: { capsEl: $("cap-bot") },
    pickOverlay: $("spick"), confirmOverlay: $("sconfirm"), endOverlay: $("send"),
    endTitleEl: $("end-title"), endLineEl: $("end-line"),
    resignBtn: $("resign-btn"), resignNo: $("resign-no"), resignYes: $("resign-yes"),
    againBtn: $("again"), pickWhite: $("pick-white"), pickBlack: $("pick-black"),
    bot: { ...bot, book: playBook }, engine, muted: state.muted,
  })
}

/* ------------------------------------------------- the Burke Bot promo card */

async function loadPromo() {
  try {
    const [stats, model] = await Promise.all([
      fetch("../stats.json").then(r => r.json()),
      fetch("../style-v9.json").then(r => r.json()),
    ])
    const fmt = (n) => Number(n).toLocaleString("en-US")
    $("bb-games").textContent = fmt(stats.games)
    $("bb-book").textContent = fmt(stats.book_positions)
    $("bb-feats").textContent = fmt(model.n_features)
  } catch (e) {
    // the promo is advertising, not function - leave the dashes rather than
    // showing numbers that might be wrong
  }
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  setTheme(false)
  let muted = false
  try { muted = localStorage.getItem("mirror-sound") === "off" } catch (e) {}
  setMuted(muted)
  wireCopyButtons()
  loadPromo()

  $("theme-toggle").addEventListener("click", () =>
    setTheme(document.documentElement.getAttribute("data-theme") !== "light"))
  $("sound-toggle").addEventListener("click", () => setMuted(!state.muted))
  for (const id of ["cc", "li"]) $(id).addEventListener("input", scheduleLookup)
  $("games").addEventListener("input", () => { state.sliderTouched = true; updateSlider() })
  $("build").addEventListener("click", build)
  $("cancel").addEventListener("click", () => state.abort && state.abort.abort())
  $("play-it").addEventListener("click", () => openBoard(true))
  $("build-own").addEventListener("click", () => {
    history.replaceState(null, "", location.pathname)
    show("setup", { top: true })
  })

  // A fragment means someone was sent this bot. Decode it and go straight to
  // the board; the setup flow is never shown to them.
  const frag = location.hash.slice(1)
  if (frag.length > 8) {
    try {
      const payload = await Share.decodeAsync(frag)
      state.bot = {
        weights: payload.weights,
        book: payload.book,
        fragment: frag,
        stats: {
          botRating: payload.meta && payload.meta.r,
          gamesUsed: payload.meta && payload.meta.g,
        },
        meta: {
          username: (payload.meta && payload.meta.u) || "someone",
          site: (payload.meta && payload.meta.s) || "chesscom",
          // the sender's measured levers, so their bot plays here as it did there
          poolDepth: payload.meta && payload.meta.pd,
          horizonDepth: payload.meta && payload.meta.hd,
          multipv: payload.meta && payload.meta.mp,
          temp: payload.meta && payload.meta.t,
        },
      }
      state.bot.meta.name = titleCase(state.bot.meta.username)
      state.bot.meta.profileUrl = state.bot.meta.site === "chesscom"
        ? `https://www.chess.com/member/${state.bot.meta.username}`
        : `https://lichess.org/@/${state.bot.meta.username}`
      lift()
      await openBoard(false)
      return
    } catch (e) {
      // a mangled link should not strand someone on a blank page
      history.replaceState(null, "", location.pathname)
    }
  }

  show("setup")
  updateSlider()
  lift()
}

boot()
