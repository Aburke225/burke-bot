// Build-a-Bot - the page controller.
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
  autoOff: new Set(),   // speeds the rated filter emptied, to be revived when it lifts
  msPerPosition: null, // one pool search, from engine.calibrate()
  msPerDecision: null, // a whole decision point: pool search + every horizon
  cap: 400,            // device-calibrated ceiling
  bot: null,           // { weights, book, levers, stats, meta }
  engine: null,
  game: null,          // the live play session
  abort: null,
  muted: false,
  sliderTouched: false,
  // PGN the player dropped in: chess.com bot games, or anything else no API
  // will hand over. Parsed on arrival, merged at build time.
  scan: null,          // exact chess.com counts, once the archive has been read
  allCounts: null,     // speed -> every game
  ratedCounts: null,   // speed -> the rated ones only
  upload: { games: [], read: 0, texts: [], players: null, identity: null, source: null, mod: null, showAllNames: false, skipped: { unplaceable: 0, speed: 0, untimed: 0, variant: 0, tooShort: 0, unreadable: 0 } },
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
  try { localStorage.setItem("build-a-bot-sound", m ? "off" : "on") } catch (e) {}
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
  state.scan = null
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
  startArchiveScan()
}

// ---------- exact chess.com counts ----------

// /stats answers instantly but only knows rated ladder records, so the first
// numbers on screen are a floor. This reads every month of the archive and
// replaces them with the real ones - the only place a casual game exists. It
// runs in the background because the page is already useful without it, and it
// is cancelled and restarted whenever the username changes.
let scanToken = 0

async function startArchiveScan() {
  const prof = state.profiles.chesscom
  const token = ++scanToken
  if (!prof) return

  const done = Games.getArchiveScan(prof.username)
  if (done) { applyScan(done); return }

  const note = $("speeds-msg")
  note.hidden = false
  note.innerHTML = '<li class="busy"><span class="t">Counting your casual games&hellip;</span></li>'
  // Write into the SAME element on every tick. Replacing the <li> replaces the
  // pseudo-element the spin is attached to, which restarts the animation from
  // zero - so the wheel stuttered once per month scanned, turning an "it is
  // working" signal into a progress readout nobody asked for. The text changes;
  // the wheel is left alone to spin.
  const busyText = note.querySelector(".t")
  try {
    const scan = await Games.scanChesscomArchive(prof.username, {}, (p) => {
      if (token !== scanToken) return
      busyText.innerHTML = `Counting your casual games&hellip; ` +
        `<b>${p.found.toLocaleString("en-US")}</b> so far (${p.done} of ${p.total} months)`
    })
    if (token !== scanToken) return
    applyScan(scan)
  } catch (err) {
    if (token !== scanToken) return
    // The floor is still a usable number, so a failed scan is a note, not an
    // error state: say what is missing rather than pretending nothing happened.
    note.innerHTML = '<li class="no"><span class="t">Could not read your full archive, so these are ' +
      "chess.com's rated totals only &mdash; casual games are still downloaded " +
      "and learned from.</span></li>"
  }
}

function applyScan(scan) {
  state.scan = scan
  renderSpeeds()          // which calls syncSpeedCounts, which draws the facts
  updateSlider()
}

/**
 * What the scan found, as two statements: what came in and what did not.
 *
 * The first line follows the rated checkbox, because ticking it changes the
 * answer. Casual games are the whole reason this scan exists - chess.com's
 * profile totals cannot see one - so a row still claiming they are included
 * while the checkbox excludes them would be the page contradicting itself one
 * line above the box that did it.
 */
function renderFacts() {
  const note = $("speeds-msg")
  const scan = state.scan
  if (!scan) return
  const ratedOnly = $("rated").checked
  const casual = scan.total - SPEEDS_ALL.reduce((a, sp) => a + (scan.ratedCounts[sp] || 0), 0)
  const short = scan.tooShort || 0

  const rows = []
  if (ratedOnly) {
    rows.push(`<li class="no"><span class="t">Read from your whole archive, ` +
      `casual games not included.</span></li>`)
  } else {
    rows.push(`<li class="yes"><span class="t">Read from your whole archive, casual games included` +
      (casual > 0 ? ` &mdash; <b>${casual.toLocaleString("en-US")}</b> of these are casual.` : ".") +
      `</span></li>`)
  }
  // A six-move resignation counts on a ladder but teaches nothing, so it is out
  // either way - the rated checkbox has no bearing on this line.
  if (short) {
    rows.push(`<li class="no"><span class="t"><b>${short.toLocaleString("en-US")}</b> ` +
      `game${short === 1 ? " was" : "s were"} too short to learn from.</span></li>`)
  }
  note.innerHTML = rows.join("")
  note.hidden = false
}

const SPEEDS_ALL = ["bullet", "blitz", "rapid", "classical", "daily"]

// Two sets of counts, not one: every game, and only the rated ones. The rated
// checkbox swaps which set is on screen, so ticking it shows the player exactly
// what it costs them instead of silently shrinking the build later.
function poolCounts(which) {
  const out = {}
  for (const [site, p] of Object.entries(state.profiles)) {
    // For chess.com an exact archive scan replaces the profile's rated-only
    // ladder totals outright rather than adding to them - the two count the
    // same games, and the scan is the one that can see a casual game.
    let src
    if (site === "chesscom" && state.scan) {
      src = which === "rated" ? state.scan.ratedCounts : state.scan.counts
    } else {
      // lichess publishes no rated/casual split we can get cheaply, so its
      // counts stand for both. Claiming a split we cannot see would be worse
      // than not narrowing them.
      src = p.counts || {}
    }
    for (const [sp, n] of Object.entries(src || {})) {
      if (n > 0) out[sp] = (out[sp] || 0) + n
    }
  }
  return out
}

function renderSpeeds() {
  // Rows exist for every speed the player has EVER played, so a row can read 0
  // under "rated only" rather than vanishing - a row that disappears looks like
  // a bug, and the player still needs its checkbox to get the games back.
  state.allCounts = poolCounts("all")
  state.ratedCounts = poolCounts("rated")

  const box = $("speeds")
  box.innerHTML = ""
  state.speeds = new Set()
  state.autoOff = new Set()
  for (const sp of SPEED_ORDER) {
    if (!state.allCounts[sp]) continue
    state.speeds.add(sp)            // present means on, per the agreed rule
    const lab = document.createElement("label")
    lab.className = "speed"
    lab.dataset.for = sp
    lab.innerHTML =
      `<input type="checkbox" data-speed="${sp}" checked>` +
      `<span class="nm">${SPEED_LABEL[sp] || sp}</span>` +
      `<span class="ct"></span>`
    box.appendChild(lab)
  }
  box.addEventListener("change", onSpeedChange)

  showBotControls()
  // The note is owned by the scan (startArchiveScan / applyScan), which knows
  // whether these numbers are the rated floor, a count in progress, or exact.
  if (!state.profiles.chesscom) $("speeds-msg").hidden = true
  $("card-speeds").hidden = false
  $("card-count").hidden = false
  syncSpeedCounts()
}

/**
 * Put the right number beside each speed and keep the checkboxes honest.
 *
 * Ticking "rated games only" can take a speed to zero - burkeley's only bullet
 * and only blitz game are both casual. A ticked box against a zero is a lie
 * about what the build will contain, so those speeds untick themselves. The
 * inverse lives in onSpeedChange: ticking such a speed back on is a clear
 * instruction to stop excluding casual games, so it releases the rated filter
 * rather than bouncing straight back to zero.
 */
function syncSpeedCounts() {
  const ratedOnly = $("rated").checked
  state.counts = ratedOnly ? state.ratedCounts : state.allCounts
  for (const lab of $("speeds").querySelectorAll(".speed")) {
    const sp = lab.dataset.for
    const n = state.counts[sp] || 0
    const cb = lab.querySelector("input")
    lab.querySelector(".ct").textContent = n.toLocaleString("en-US")
    // Untick what the rated filter emptied - and tick it back when the filter
    // lifts. Only speeds THIS code turned off are revived: a speed the player
    // unticked themselves stays off, because coming back to life under them
    // would be the page overruling a choice they made on purpose.
    if (!n && cb.checked) { cb.checked = false; state.speeds.delete(sp); state.autoOff.add(sp) }
    else if (n && !cb.checked && state.autoOff.has(sp)) {
      cb.checked = true; state.speeds.add(sp); state.autoOff.delete(sp)
    }
    lab.classList.toggle("off", !cb.checked)
    lab.classList.toggle("empty", !n)
  }
  renderFacts()
  updateSlider()
}

function onSpeedChange(ev) {
  const cb = ev.target.closest("input[type=checkbox]")
  if (!cb || !cb.dataset.speed) return
  const sp = cb.dataset.speed

  // Turning a speed back on that "rated games only" had emptied: the player is
  // asking for those games, and the only way to give them any is to stop
  // filtering casual ones out. Release the filter and restore every count.
  if (cb.checked && $("rated").checked && !(state.ratedCounts[sp] || 0)) {
    $("rated").checked = false
  }

  // Touching the box at all makes it the player's, so it stops being ours to
  // revive later.
  state.autoOff.delete(sp)
  if (cb.checked) state.speeds.add(sp)
  else state.speeds.delete(sp)
  // uploaded games are filtered by speed too, so they have to be re-read
  if (state.upload.texts && state.upload.texts.length) reparseUpload()
  else syncSpeedCounts()
}

// ---------- uploaded PGN ----------

// chess.com publishes no games against its own bots, so a file the player
// exports is the only way they reach the fit. Kept in state rather than parsed
// at build time so the count can be shown the moment a file lands and a bad
// file is reported while there is still something to do about it.
function uploadNames() {
  return Object.values(state.profiles).map((p) => p.username).filter(Boolean)
}

// Who shows what.
//
//   lichess only      - the checkbox. lichess publishes its bot games, so there
//                       is nothing to upload.
//   chess.com only    - the upload button, because chess.com publishes none.
//                       Once a file is in, the button has done its job: it
//                       becomes the same checked checkbox, which is now what
//                       decides whether those games are used.
//   both              - the checkbox (for lichess) and the button (for
//                       chess.com) side by side, until the upload lands and
//                       the button goes the same way.
function showBotControls() {
  const uploaded = state.upload.games.length > 0
  const lichess = !!state.profiles.lichess
  const chesscom = !!state.profiles.chesscom
  $("bots-wrap").hidden = !(lichess || uploaded)
  $("drop-wrap").hidden = !chesscom || uploaded
  // Uploading is itself the decision to include them; arriving unticked would
  // throw away the file the player just went and fetched.
  if (uploaded) $("bots").checked = true
}

function describeUpload() {
  const msg = $("pgn-msg")
  const u = state.upload
  if (!u || (!u.games.length && !u.read)) { msg.hidden = true; msg.className = "upload-result"; return }
  const n = u.games.length

  // The headline is the whole point: a file went in and games came out. It gets
  // the size and the colour. Everything else - what was untimed, what was too
  // short - is a footnote to that, and reads as one.
  // ONLY games that did not make it. "30 untimed" was noise here: those games
  // were added, so listing them beside the exclusions invited the reader to
  // subtract a number that is already inside the count.
  const s = u.skipped
  const notes = []
  if (s.unplaceable) notes.push(`${s.unplaceable} with neither player recognised`)
  if (s.speed) notes.push(`${s.speed} outside the game types above`)
  if (s.variant) notes.push(`${s.variant} not standard chess`)
  if (s.tooShort) notes.push(`${s.tooShort} too short`)
  if (s.unreadable) notes.push(`${s.unreadable} unreadable`)

  // Only guess at a cause when the guess is a good one. Every game belonging to
  // someone else is the identity case; nothing added for any other reason is
  // not, and saying so would send the reader after the wrong thing.
  if (!n && s.unplaceable) {
    notes.push("there is no way to tell which side was you")
  }

  msg.className = "upload-result " + (n ? "ok" : "none")
  msg.innerHTML =
    `<span class="head"><b>${n.toLocaleString("en-US")}</b> ` +
    `bot game${n === 1 ? "" : "s"} added</span>` +
    (notes.length ? `<span class="why">${notes.join(" &middot; ")}</span>` : "")
  msg.hidden = false
}

async function takeFiles(files) {
  const list = [...files].filter((f) => f && f.size)
  if (!list.length) return
  const G = state.upload.mod || (state.upload.mod = await import("./games.js"))

  state.upload.texts = state.upload.texts || []
  for (const f of list) {
    try { state.upload.texts.push(await f.text()) }
    catch { state.upload.skipped.unreadable++ }
  }
  // Identity is worked out from the FILES, not demanded of the player. Only if
  // the file has no clear protagonist does anything get asked.
  const all = state.upload.texts.join("\n\n")
  state.upload.players = G.pgnPlayers(all)
  const found = G.inferPgnIdentity(all, uploadNames())
  if (found && (!state.upload.identity || found.source === "account")) {
    state.upload.identity = found.name
    state.upload.source = found.source
  }
  reparseUpload()
}

// Re-reads every file held so far under the current identity. Cheap - the text
// is already in memory - and it means changing who "you" are re-decides every
// game's colour at once instead of only the next file's.
function reparseUpload() {
  const G = state.upload.mod
  const u = state.upload
  u.games = []
  u.read = 0
  for (const k of Object.keys(u.skipped)) u.skipped[k] = 0
  // Which speeds an upload may use. A class the player HAS and has unticked is
  // a real instruction, so it is honoured. A class with no row on screen at all
  // - classical, when their online account has never played one - is not a
  // choice they declined, it is a choice they were never offered, and dropping
  // an over-the-board game for failing it would be the untimed bug again.
  const onScreen = new Set(Object.keys(state.allCounts || {}))
  const speeds = SPEEDS_ALL.filter((sp) => !onScreen.has(sp) || state.speeds.has(sp))

  const seen = new Set()
  for (const text of u.texts || []) {
    const r = G.parsePgn(text, {
      identity: u.identity,
      usernames: uploadNames(),
      speeds,
    })
    u.read += r.read
    for (const k of Object.keys(u.skipped)) u.skipped[k] += r.skipped[k] || 0
    for (const g of r.games) {
      const k = G.gameKey(g)          // the same file twice must not double up
      if (seen.has(k)) continue
      seen.add(k)
      u.games.push(g)
    }
  }
  describeUpload()
  renderIdentity()
  showBotControls()
  updateSlider()
}

/**
 * Say who we took the games to belong to, and let that be corrected.
 *
 * Silent is not an option here. Reading the wrong side of the board does not
 * fail - it fits the opponent's style and every number still looks right - so
 * whenever the name was inferred rather than matched to an account they gave
 * us, it is stated plainly with the other names in the file one click away.
 */
const IDENTITY_SHOWN = 6

function renderIdentity() {
  const box = $("pgn-who")
  const u = state.upload
  if (!u.identity || !u.players || u.players.length < 2) { box.hidden = true; return }
  // Nothing to confirm when the name in the file is an account they typed - we
  // are not guessing, so asking would be noise. Their own chess.com export names
  // three hundred opponents; every one of them would be offered as "not you?".
  if (u.source === "account") { box.hidden = true; return }

  // Six, because the list is sorted by how many games each player appears in and
  // the real alternative is always near the top. A personal export makes that
  // stark: burkeley is in 331 of 331 games, the next name in 10, and 306 of the
  // 313 others appear exactly once. Showing all of them would be a wall of
  // strangers. But the cap is stated rather than silent - a truncated list that
  // does not admit it is how someone concludes their name is not there at all.
  const rest = u.players.filter((p) => p.name !== u.identity)
  const others = u.showAllNames ? rest : rest.slice(0, IDENTITY_SHOWN)
  const hidden = rest.length - others.length
  // An inference from a clear protagonist and a coin toss between two equally
  // present names are not the same claim, and until now they were worded as if
  // they were. "ambiguous" means the file had no protagonist at all - a single
  // game between two strangers - so the name shown is White of the first game
  // and nothing more. Saying so is what makes the buttons beside it look worth
  // reading, which matters here more than anywhere else on the page: picking
  // the wrong side does not fail, it fits the opponent's style and every
  // number still looks right.
  const lead = u.source === "ambiguous"
    ? `Could not detect which player is you &mdash; reading as <b>${escapeHtml(u.identity)}</b>.`
    : `Read as <b>${escapeHtml(u.identity)}</b>'s games.`
  box.classList.toggle("guess", u.source === "ambiguous")
  box.innerHTML =
    lead +
    (others.length ? ` Not you? ` + others.map((p) =>
      `<button type="button" class="who" data-name="${escapeHtml(p.name)}">${escapeHtml(p.name)}</button>`
    ).join(" ") : "") +
    (hidden > 0 ? ` <button type="button" class="who more" data-more="1">+${hidden.toLocaleString("en-US")} more</button>` : "")
  box.hidden = false
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]))
}

function wireUpload() {
  const wrap = $("drop-wrap")
  const input = $("pgn")
  // The box IS the control now that the button is gone. Without this the whole
  // feature would be drag-only, which is the same as saying it does not exist
  // on a phone - there is nothing to drag from on a touch screen.
  wrap.addEventListener("click", () => input.click())
  // it now moves the game total, so the slider and the build button have to
  // hear about it
  $("bots").addEventListener("change", () => { describeUpload(); updateSlider() })
  $("pgn-who").addEventListener("click", (ev) => {
    const b = ev.target.closest("button.who")
    if (!b) return
    if (b.dataset.more) { state.upload.showAllNames = true; renderIdentity(); return }
    state.upload.identity = b.dataset.name
    state.upload.source = "chosen"
    reparseUpload()
  })
  // ticking it re-counts every speed and may untick the ones it empties
  $("rated").addEventListener("change", () => { if (state.allCounts) syncSpeedCounts() })
  input.addEventListener("change", () => { takeFiles(input.files); input.value = "" })

  // dragover must be cancelled or the browser navigates to the file instead
  for (const ev of ["dragenter", "dragover"]) {
    wrap.addEventListener(ev, (e) => { e.preventDefault(); wrap.classList.add("over") })
  }
  for (const ev of ["dragleave", "drop"]) {
    wrap.addEventListener(ev, (e) => { e.preventDefault(); wrap.classList.remove("over") })
  }
  wrap.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files) takeFiles(e.dataTransfer.files)
  })
  // a file dropped anywhere else would otherwise replace the page
  for (const ev of ["dragover", "drop"]) {
    window.addEventListener(ev, (e) => { if (e.target.closest && !e.target.closest("#drop-wrap")) e.preventDefault() })
  }
}

/* ------------------------------------------------------------ the slider */

function selectedTotal() {
  let n = 0
  for (const sp of state.speeds) n += state.counts[sp] || 0
  // Uploaded games are real games and they count. parsePgn already applied the
  // speed choices above, and the build dedupes against the archive, so at worst
  // this over-counts by the overlap and the slider offers a few games the
  // download quietly folds together.
  return n + uploadedInPlay()
}

// Uploaded games only count while the include-bot-games box is ticked, because
// that is the box that decides whether the build uses them.
function uploadedInPlay() {
  const box = document.getElementById("bots")
  return box && box.checked ? state.upload.games.length : 0
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

/**
 * How long a build of n games takes.
 *
 * ~27.7 decision points a game, each costing one CALIBRATED DECISION - the pool
 * search plus one search per horizon, which is what engine.calibrate() now
 * times. The old version priced a decision at one search and was about five
 * times light; worse, the shortfall was machine-shaped, so no constant could
 * fix it for every device. Nothing here is tuned to a particular processor now.
 *
 * OVERHEAD_SECS is the rest of the job - download, probe, fit, opening book -
 * which a scoring-only figure omits entirely. It is roughly flat in n and small
 * beside the scoring pass, so one measured number covers it.
 */
// 34.3, counted over 72 real games rather than assumed: every move the player
// made after the opening ply. The old 27.7 was 24% light on its own.
const DECISIONS_PER_GAME = 34.3
const OVERHEAD_SECS = 4

/**
 * What calibration cannot see.
 *
 * scoreAll yields to the browser once per decision point so the page keeps
 * painting. On a quiet page that yield is free, which is exactly why measuring
 * it during calibration does not help - the setup screen is idle. During a
 * build the spinner and the progress bar are animating, so the same yield waits
 * for a render and costs about 28ms. Measured per decision:
 *
 *     engine, four searches   9.0ms   calibrated per machine
 *     v9 feature extraction   3.6ms   calibrated per machine
 *     yield while animating  ~28ms    this factor
 *
 * The two that scale with the device are now measured on the device. What is
 * left is the browser's rendering cadence, which is far more uniform across
 * machines than wasm or JS throughput - so this travels much better than the
 * single 5.3x constant it replaces, which was standing in for all four of the
 * engine, the JavaScript, the yield, AND a 24% error in decisions per game.
 */
const LOOP_FACTOR = 3.0
// Download is done by the time this is used, so it covers probe + fit + book.
const POST_DOWNLOAD_SECS = 4
// What is left after scoring finishes: the fit and the opening book.
const TAIL_SECS = 2

function estimateFromDecisions(points) {
  const ms = state.msPerDecision || 12.5
  return POST_DOWNLOAD_SECS + (points * ms * LOOP_FACTOR) / 1000
}

function estimateSeconds(n) {
  // msPerDecision is the real cost of one decision point - pool search plus
  // every horizon - so there is no correction factor left to apply and nothing
  // tuned to the machine this was written on. 50 is the fallback before the
  // engine has calibrated, which is this machine's measured figure and only
  // ever stands in for the second or two before the real one arrives.
  // 12.5 is this machine's calibrated figure, standing in for the second or two
  // before the real one arrives.
  const ms = state.msPerDecision || 12.5
  return OVERHEAD_SECS + (n * DECISIONS_PER_GAME * ms * LOOP_FACTOR) / 1000
}

// A non-breaking space BEFORE the dot and an ordinary one after it. The recap
// can now run to two lines, and this is what decides where they break: the dot
// is glued to the phrase it follows, so a wrapped line can end "rapid, daily ·"
// but can never begin with an orphaned separator.
const SEP = "&nbsp;<span class=\"sep\">&middot;</span> "

function joinBits(bits) {
  return bits.join(SEP)
}

function prettyTime(secs) {
  if (secs < 60) return Math.max(1, Math.round(secs)) + "s"
  const m = Math.floor(secs / 60), s = Math.round(secs % 60)
  return s ? `${m}m ${s}s` : `${m}m`
}

function grandTotal() {
  return Object.values(state.counts).reduce((a, b) => a + b, 0) + uploadedInPlay()
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
      text: `${who} has ${n} we can read. Build-a-Bot needs at least ${MIN_GAMES} ` +
            `to build anything worth calling a bot of you.`,
      cls: "note bad",
    }
  }
  const spare = grand - total
  return {
    text: `Only ${total} game${total === 1 ? "" : "s"} in the types you have picked. ` +
          `Build-a-Bot needs at least ${MIN_GAMES} to build a bot of you &mdash; ` +
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
  const secs = estimateSeconds(n)
  state.etaSecs = secs
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
    // Calibrate on a WHOLE decision - engine plus this app's own feature work -
    // because the engine is under a fifth of it. build.js owns that path, so it
    // owns the measurement; engine.calibrate() is kept for the engine-only
    // figure, which is still worth having for diagnostics.
    const { calibrateDecision } = await import("./build.js")
    const cal = await calibrateDecision(state.engine)
    state.msPerDecision = cal.msPerDecision
    state.msPerPosition = null
    // iOS suspends a page the moment the user switches apps, so a phone run has
    // to fit inside one uninterrupted look at the screen. ~60s of engine time.
    const phone = window.matchMedia("(max-width: 860px)").matches
    if (phone) {
      state.cap = Math.max(25, Math.min(150, Math.round(60000 / (state.msPerDecision * LOOP_FACTOR * DECISIONS_PER_GAME))))
      const note = $("mobile-note")
      note.textContent = "To train on all of your games, use a desktop or laptop."
      note.hidden = false
    } else {
      state.cap = Math.max(50, Math.round(120000 / (state.msPerDecision * LOOP_FACTOR * DECISIONS_PER_GAME)))
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
  // The estimate rides along here and nowhere else: it is the one screen where
  // "how long is this going to take" is the live question. It is the figure the
  // setup page quoted, frozen - a number that keeps revising itself while you
  // watch is a worse answer than a slightly wrong one that holds still.
  // Every choice that shaped this build, not just the rated one. Someone who
  // ticked three speeds and uploaded a pile of bot games was being told only
  // "rated and casual", which described the least of what they did.
  const chosen = SPEED_ORDER
    .filter((sp) => state.speeds.has(sp))
    .map((sp) => (SPEED_LABEL[sp] || sp).toLowerCase())
  const uploaded = $("bots").checked ? state.upload.games.length : 0
  const bits = [`<b>${cap}</b>`, `<b>${n}</b> games`]
  if (chosen.length) bits.push(chosen.join(", "))
  bits.push($("rated").checked ? "rated only" : "rated and casual")
  if (uploaded) bits.push(`plus <b>${uploaded.toLocaleString("en-US")}</b> bot games`)
  else if ($("bots").checked) bits.push("bot games included")

  $("recap-building").innerHTML = joinBits(bits) +
    `<span class="eta">${SEP}about <b>${prettyTime(state.etaSecs || estimateSeconds(n))}</b> to build</span>`
  state.buildSize = n
  state.scoreT0 = null
  resetSteps()
  // One run, the length of the whole estimate. Overrun and it closes on 99.2%
  // by halves without ever arriving; only the build finishing fills it.
  startBar(state.etaSecs || estimateSeconds(n))
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
      // The same checkbox that governs lichess bot games governs these: after
      // an upload it IS the control that replaced the button.
      extraGames: $("bots").checked ? state.upload.games : [],
      maxGames: n,
      signal: state.abort.signal,
      engine,
    }, onProgress)
    finishSteps()
    await finish(bot, accounts)
  } catch (err) {
    if (err && (err.name === "AbortError" || /abort/i.test(err.message || ""))) {
      show("setup", { top: true })
      return
    }
    stopBar()
    const li = stepAt >= 0 && $("step-" + STEPS[stepAt][0])
    if (li) li.className = "step failed"
    setNote("Could not build that bot: " + (err.message || err))
  }
}

// The build in the order it actually happens. build.js reports one of these
// phase names; the bar belongs to whichever is in flight, and everything above
// it has a tick and stays on screen. A reader can see what is done, what is
// running, and that there is more to come - which one relabelled bar could not
// show. `finish` covers book and stats, which are one wait as far as anyone
// watching is concerned.
const STEPS = [
  ["download", "Downloading your games"],
  ["probe", "Measuring how far you see"],
  ["score", "Scoring your moves"],
  ["fit", "Fitting your style"],
  ["finish", "Building your opening book"],
]

// -1 = nothing started. Phases only ever move forward, and one can be skipped
// (probe does not run when an engine is handed in), so arriving at step N marks
// every earlier step done rather than assuming N-1 was the last one seen.
let stepAt = -1

/**
 * The bar, on a clock rather than on the data.
 *
 * Driving the width straight off done/total made it lurch: scoring reports once
 * per GAME, so on a 14-game build the bar jumped in 7% steps and sat still in
 * between, and the probe and fit phases report a handful of times in total. The
 * information was honest and the motion was useless.
 *
 * So the bar walks the WHOLE build's estimate, once, start to finish. Two rules
 * keep it from lying. It never passes CEILING on the clock alone, so a build
 * that overruns leaves the bar closing on 99.2% by halves - always moving,
 * never arriving - instead of claiming to be done. And when the build really
 * does finish, the bar goes to 100% from wherever it had got to: early or late,
 * completion is the only thing that fills it.
 */
const BAR_CEILING = 0.94
const BAR_ASYMPTOTE = 99.2

let barTimer = null
let barFrom = 0        // width the current step started at (always 0 today)
let barStart = 0       // when the current step began
let barSpan = 1000     // how long we think it will take, in ms

// The note row only exists when it carries something; an always-present empty
// row would be a fourth, invisible gap in a stack whose spacing is meant to be
// even.
function setNote(html) {
  const note = $("note")
  note.innerHTML = html || ""
  note.parentElement.hidden = !html
}

function paintBar(pct) {
  $("fill").style.width = Math.max(0, Math.min(100, pct)) + "%"
}

function stopBar() {
  if (barTimer) { clearInterval(barTimer); barTimer = null }
}

// An ease-out: quick off the mark, slowing as it approaches the ceiling, which
// is how a progress bar reads as "working" rather than as "counting down".
function currentPct() {
  return parseFloat($("fill").style.width) || 0
}

/**
 * Re-time the bar mid-flight without moving it.
 *
 * The setup estimate is games x an average decisions-per-game, and game LENGTH
 * varies far more than game count does - a 36-game build measured 0.78s a game
 * against 1.41s for a 72-game one, purely because the games were shorter. Once
 * the download lands, the exact number of decision points is known, so the bar
 * stops extrapolating and starts running on the real figure.
 *
 * It re-anchors rather than restarting: the curve picks up from whatever width
 * is already painted, so the bar never jumps or goes backwards - the only thing
 * that changes is how fast it moves from here.
 */
function repaceBar(seconds) {
  startBar(seconds, currentPct())
}

function startBar(seconds, from = 0) {
  stopBar()
  barFrom = from
  barStart = performance.now()
  barSpan = Math.max(600, seconds * 1000)
  // paint the anchor, not zero - re-pacing mid-build must not snap the bar back
  paintBar(barFrom)
  barTimer = setInterval(() => {
    const t = (performance.now() - barStart) / barSpan
    if (t <= 1) {
      // ease-out to the ceiling: quick off the mark, slowing as it approaches
      const eased = 1 - Math.pow(1 - t, 2)
      paintBar(barFrom + eased * (BAR_CEILING * 100 - barFrom))
    } else {
      // Past the estimate. An estimate is a guess and some machines are slow,
      // so rather than freezing at the ceiling - which reads as a hang - it
      // keeps closing on 99.2% by halves. Always moving, never arriving; only
      // the step actually ending fills it.
      const over = t - 1
      const ceil = Math.max(BAR_CEILING * 100, barFrom)
      paintBar(BAR_ASYMPTOTE - (BAR_ASYMPTOTE - ceil) * Math.exp(-over))
    }
  }, 50)
}

function completeBar() {
  stopBar()
  paintBar(100)
}

function resetSteps() {
  stepAt = -1
  stopBar()
  $("steps").innerHTML = ""
  setNote("")
  paintBar(0)
}

function stepRow(i) {
  const li = document.createElement("li")
  li.className = "step running"
  li.id = "step-" + STEPS[i][0]
  li.innerHTML = '<span class="tick" aria-hidden="true"></span>' +
    '<span class="what"></span><span class="detail"></span>'
  li.querySelector(".what").textContent = STEPS[i][1]
  return li
}

// build.js writes its labels as "Scoring your moves - game 5 of 309", so the
// half after the dash is the live detail and the half before it is the step
// name this row already shows. A label with no dash carries no detail.
function detailOf(label) {
  if (!label) return ""
  const i = label.indexOf(" - ")
  return i === -1 ? "" : label.slice(i + 3)
}

function enterStep(i) {
  // The bar is NOT touched here. It runs once, across the whole estimate, so
  // that its position answers "how much longer" rather than "how far into a
  // stage whose length you have no idea about". Restarting it per step also
  // meant any step the clock mispriced showed an empty bar going nowhere - the
  // fit stage, at a flat guess of eight seconds, was reliably that step.
  for (let k = Math.max(stepAt, 0); k < i; k++) {
    const done = $("step-" + STEPS[k][0])
    if (done) { done.className = "step done"; done.querySelector(".detail").textContent = "" }
  }
  for (let k = stepAt + 1; k <= i; k++) {
    if (!$("step-" + STEPS[k][0])) $("steps").appendChild(stepRow(k))
  }
  stepAt = i
}

function finishSteps() {
  for (let k = 0; k <= stepAt; k++) {
    const li = $("step-" + STEPS[k][0])
    if (li) { li.className = "step done"; li.querySelector(".detail").textContent = "" }
  }
  completeBar()
}

function onProgress(p) {
  // Not a step - it is the moment the guesswork can stop. buildBot reports the
  // true number of decision points as soon as the games are parsed, which is
  // the only figure that actually predicts how long scoring will take.
  if (p.phase === "plan") {
    state.decisions = p.total
    repaceBar(estimateFromDecisions(p.total))
    return
  }

  // Scoring is the long pole, and it reports once per game - which is enough to
  // measure this build's ACTUAL rate instead of predicting it. Two builds of
  // the same size differ by more than a third depending on how long the games
  // are and how the engine warms, so a figure fixed before the first move was
  // read can only ever be close. From here the bar runs on observed throughput.
  if (p.phase === "score" && p.total) {
    if (state.scoreT0 == null) { state.scoreT0 = performance.now(); state.lastRepace = 0 }
    const frac = p.done / p.total
    const elapsed = (performance.now() - state.scoreT0) / 1000
    // Wait for a tenth of the work before believing the rate, and re-pace at
    // most twice a second: an estimate that twitches is worse than one slightly
    // behind.
    if (frac > 0.1 && elapsed - state.lastRepace > 2) {
      state.lastRepace = elapsed
      const projectedScoring = elapsed / frac
      repaceBar(Math.max(1, projectedScoring - elapsed + TAIL_SECS))
    }
    // fall through: the step row still wants its "game 5 of 24" detail
  }
  const i = STEPS.findIndex((s) => s[0] === p.phase)
  // An unknown phase still deserves its label rather than being dropped.
  if (i === -1) { if (p.label) setNote(p.label); return }
  if (i > stepAt) enterStep(i)
  if (i === stepAt) {
    const li = $("step-" + p.phase)
    if (li) li.querySelector(".detail").textContent = detailOf(p.label)
  }
  // Deliberately does NOT move the bar. The bar is on its own clock; these
  // events are far too lumpy to animate from - one per game, or three in total
  // for a whole phase - and that lumpiness was the choppiness.
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
  // Strip the estimate: on the result page the wait is over, so how long it was
  // going to take is no longer a question anyone has. Done by removing the NODE
  // rather than by pattern-matching the markup - the first attempt at this used
  // a regex that assumed the span ended in </span></span> when it actually ends
  // in </b></span>, so it silently matched nothing and the estimate shipped
  // through to the result page.
  const recap = $("recap-building").cloneNode(true)
  const eta = recap.querySelector(".eta")
  if (eta) eta.remove()
  $("recap-result").innerHTML = recap.innerHTML +
    ` <span>&middot;</span> <a href="#" id="again-link">build another bot</a>`
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
  wireUpload()
  setTheme(false)
  let muted = false
  try { muted = localStorage.getItem("build-a-bot-sound") === "off" } catch (e) {}
  setMuted(muted)
  wireCopyButtons()
  loadPromo()

  $("theme-toggle").addEventListener("click", () =>
    setTheme(document.documentElement.getAttribute("data-theme") !== "light"))
  $("sound-toggle").addEventListener("click", () => setMuted(!state.muted))
  for (const id of ["cc", "li"]) $(id).addEventListener("input", scheduleLookup)
  $("games").addEventListener("input", () => { state.sliderTouched = true; updateSlider() })
  $("build").addEventListener("click", build)
  $("cancel").addEventListener("click", () => { stopBar(); state.abort && state.abort.abort() })
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
