// Mirror Bot — packing a whole bot into a URL fragment.
//
// A shared bot is a link and nothing else. No server, no storage, no download.
// The payload rides in the fragment (after the #), which browsers never put on
// the wire, so a shared bot never touches anyone's logs.
//
// ---------------------------------------------------------------------------
// WIRE FORMAT (all multi-byte fields little-endian)
//
//   offset      size      field
//   0           1         format version                    (FORMAT_VERSION)
//   1           1         flags: bit0 hasBook, bit1 bookDeflated
//   2           2         uint16 weight scale = round(maxabs * 10000)
//   4           2*58      int16 weights, round(w / unit * 32767), unit = scale/1e4
//   120         1         meta length, in bytes of UTF-8 JSON
//   121         M         meta JSON {u, s, r, g, v}
//   121+M       B         opening book, as a move trie rooted at the start
//   len-2       2         uint16 CRC-16/CCITT-FALSE over every preceding byte
//
// The trie is pre-order DFS. Each node is one byte, its child count; each edge
// is two bytes, packed from6 | to6 | promo2 | own1 | reserved1. A node's
// children follow it immediately, each child's own subtree inline. That is
// self-delimiting, so no length prefix is needed. When bit1 is set the trie
// span is deflate-raw and runs to len-2 instead.
//
// from6|to6|promo2 is 14 bits and the edge word is 16, so bit 14 was free. It
// carries `own`: whether the OWNER played this move here, as against an edge
// that only exists to walk through the opponent's reply. Measured on the
// owner's book at the >=2x threshold, 60 of 238 edges standing at his own
// positions are moves he never played there — a quarter of the bot's book would
// have been invented. The bit costs nothing and removes all of it. See
// buildTrie() for why those edges have to exist at all, and trieToMoveMap()
// for how the bot is kept off them.
//
// THERE IS NO nFeatures FIELD. The count is a constant of the format, pinned
// to the version byte: format 1 means exactly 58 features, in the order
// style-v9.json lists them. A model with a different feature count needs a new
// FORMAT_VERSION, not a new header field — that is what the version byte is
// for. encode() refuses a weight vector of the wrong length rather than
// writing something decode() would silently misread.
//
// ---------------------------------------------------------------------------
// WHY THESE CHOICES — all measured, none guessed. Do not undo them.
//
//   int16 weights. Quantising to int16 changed the bot's chosen move in 0.0000%
//   of 20,000 simulated pools, against the model's own 0.125% behaviour-mismatch
//   floor. Three orders of magnitude below the model's own noise, so it is free.
//
//   No deflate on the weights. Measured on style-v9.json: 116 bytes raw, 121
//   deflated — 156 base64 chars becoming 164. Quantised floats are
//   near-maximum-entropy and deflate can only add its own framing.
//
//   Deflate on the BOOK, but only after checking. Measured on the owner's book
//   at >=2x: 1,123 bytes of trie became 494, and the whole URL went from 1,788
//   characters to 903. Opposite outcome to the weights, and for the opposite
//   reason — a trie is mostly repeated 0x00 and 0x01 node bytes and repeated
//   square patterns. encodeAsync() still measures rather than assumes, and
//   keeps the compressed form only when it is genuinely smaller; flag bit1
//   records which way it went, so decode never has to guess.
//
//   The book is a trie, not a FEN-keyed map. In the owner's real book the FEN
//   keys alone were 245 KB of 468 KB, and every byte of that is waste: a book
//   is natively a trie, and the position is implied by the path. Measured on
//   docs/book.json, 468,505 bytes packs to a few hundred.
//
//   base64url, not base91. base91's quotes and backslashes break link
//   auto-detection in chat clients, and it buys only ~74 characters.
//
//   1800 characters, because DISCORD hard-rejects a message over 2,000 and the
//   message silently vanishes — no error, no post, the user thinks they shared
//   it. Browsers tolerate far more. Discord is the binding constraint.
//
// ---------------------------------------------------------------------------
// WHAT THE FORMAT DELIBERATELY DROPS
//
// The trie carries moves, not counts: two bytes per edge has no room for the
// owner's n/w/d. So a shared book knows WHICH moves its owner played from a
// position, but not how often. To keep as much of that as fits in zero bytes,
// encode writes each node's children in descending play count, so the first
// child is the owner's most-played move. trieToMoveMap() can turn that ranking
// back into selection weights — see its `weight` option, and note that those
// weights are reconstructed from rank, not the owner's real counts.
//
// It also drops the win/draw record. Nothing built on a shared book should show
// a score for a line, because the number is not in the link.

// ---------------------------------------------------------------------------
// constants

/** Longest share URL we will emit. Discord's 2,000-character message cap is
 *  the binding constraint, not the browser. */
export const MAX_URL_CHARS = 1800

/** Wire format version. Bump this if the layout or the feature count changes. */
export const FORMAT_VERSION = 1

/** Feature count pinned to FORMAT_VERSION 1. Not stored on the wire. */
export const N_FEATURES = 58

/** Largest maxabs weight the uint16 scale can hold: 65535 / 10000. */
export const MAX_WEIGHT_MAGNITUDE = 6.5535

/** Standard opening position, and the root of every trie. */
export const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

const FLAG_HAS_BOOK = 1
const FLAG_BOOK_DEFLATED = 2
const HEADER_BYTES = 4
const WEIGHT_BYTES = N_FEATURES * 2
const META_OFFSET = HEADER_BYTES + WEIGHT_BYTES   // 120
const MAX_META_BYTES = 255
const CRC_BYTES = 2

/** Promotion codes for the 2-bit field. Only meaningful on a promoting move —
 *  which the reader knows, because it is replaying the line on a real board. */
const PROMO_TO_CODE = { q: 0, r: 1, b: 2, n: 3 }
const CODE_TO_PROMO = ["q", "r", "b", "n"]

export class ShareError extends Error {
  constructor(message, code) {
    super(message)
    this.name = "ShareError"
    this.code = code
  }
}

// ---------------------------------------------------------------------------
// small primitives

/** The book's position key: the first three FEN fields. Matches app.js's
 *  bookKey() exactly, and the two must stay in step. */
export function bookKey(fen) {
  return fen.split(" ").slice(0, 3).join(" ")
}

/** Square index, a1 = 0, h8 = 63. */
export function squareToIndex(sq) {
  const file = sq.charCodeAt(0) - 97
  const rank = sq.charCodeAt(1) - 49
  if (file < 0 || file > 7 || rank < 0 || rank > 7) {
    throw new ShareError("bad square: " + sq, "BAD_SQUARE")
  }
  return rank * 8 + file
}

export function indexToSquare(i) {
  return String.fromCharCode(97 + (i & 7)) + String.fromCharCode(49 + (i >> 3))
}

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor.
// Bitwise rather than table-driven — the payload is under 2 KB and a 512-byte
// table would cost more than it saves.
export function crc16(bytes, end) {
  const stop = end === undefined ? bytes.length : end
  let crc = 0xffff
  for (let i = 0; i < stop; i++) {
    crc ^= bytes[i] << 8
    for (let b = 0; b < 8; b++) {
      crc = crc & 0x8000 ? ((crc << 1) ^ 0x1021) & 0xffff : (crc << 1) & 0xffff
    }
  }
  return crc & 0xffff
}

export function toBase64Url(bytes) {
  let bin = ""
  // chunked: String.fromCharCode(...big array) blows the argument limit
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000))
  }
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
}

export function fromBase64Url(str) {
  const clean = String(str).trim().replace(/-/g, "+").replace(/_/g, "/")
  if (!/^[A-Za-z0-9+/]*$/.test(clean)) {
    throw new ShareError("link is not valid base64url", "BAD_BASE64")
  }
  const padded = clean + "=".repeat((4 - (clean.length % 4)) % 4)
  let bin
  try {
    bin = atob(padded)
  } catch {
    throw new ShareError("link is not valid base64url", "BAD_BASE64")
  }
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

// ---------------------------------------------------------------------------
// weights

/**
 * Quantise to int16 against a uint16 scale.
 *
 * The scale stores round(maxabs * 10000), so the value that maps to ±32767 is
 * scale/10000 — a hair off the true maxabs. Quantising against that same
 * rounded unit rather than the raw maxabs keeps encode and decode symmetric;
 * otherwise the largest weight could quantise to 32768 and wrap.
 *
 * A maxabs above MAX_WEIGHT_MAGNITUDE cannot be expressed by a uint16 scale.
 * Rather than refuse to make a link, we saturate the scale and clamp. A weight
 * that large has already pinned the softmax, so clamping moves the bot far less
 * than handing the user a dead share button. `clamped` reports when it happened.
 */
export function quantiseWeights(weights) {
  if (!weights || weights.length !== N_FEATURES) {
    throw new ShareError(
      "expected " + N_FEATURES + " weights for format " + FORMAT_VERSION +
      ", got " + (weights ? weights.length : "none"),
      "BAD_FEATURE_COUNT"
    )
  }
  let maxabs = 0
  for (const w of weights) {
    if (!Number.isFinite(w)) throw new ShareError("weight is not finite", "BAD_WEIGHT")
    const a = Math.abs(w)
    if (a > maxabs) maxabs = a
  }
  const rawScale = Math.round(maxabs * 10000)
  const scale = Math.min(65535, rawScale)
  const clamped = rawScale > 65535
  const ints = new Int16Array(N_FEATURES)
  if (scale > 0) {
    const unit = scale / 10000
    for (let i = 0; i < N_FEATURES; i++) {
      let q = Math.round((weights[i] / unit) * 32767)
      if (q > 32767) q = 32767
      if (q < -32767) q = -32767
      ints[i] = q
    }
  }
  const back = dequantiseWeights(scale, ints)
  let maxErr = 0
  for (let i = 0; i < N_FEATURES; i++) {
    const e = Math.abs(back[i] - weights[i])
    if (e > maxErr) maxErr = e
  }
  return { scale, ints, maxabs, maxErr, clamped, step: scale > 0 ? scale / 10000 / 32767 : 0 }
}

export function dequantiseWeights(scale, ints) {
  const unit = scale / 10000
  const out = new Array(N_FEATURES)
  for (let i = 0; i < N_FEATURES; i++) out[i] = (ints[i] / 32767) * unit
  return out
}

// ---------------------------------------------------------------------------
// meta

const te = new TextEncoder()
const td = new TextDecoder()

/** Pack meta to at most 255 bytes, shortening the username before dropping
 *  anything — the name is the part a recipient recognises. */
export function packMeta(meta) {
  const m = meta || {}
  const build = (u) => {
    const o = {}
    if (u) o.u = u
    if (m.s != null) o.s = m.s
    if (m.r != null) o.r = m.r
    if (m.g != null) o.g = m.g
    if (m.v != null) o.v = m.v
    // The measured levers. Without these a shared bot falls back to play.js's
    // defaults - pool depth 5, horizon 2, MultiPV 20 - which is a 1300-rated
    // player's settings applied to everyone. They cost ~28 URL characters
    // against an 1800 budget, so carrying them is close to free.
    if (m.pd != null) o.pd = m.pd
    if (m.hd != null) o.hd = m.hd
    if (m.mp != null) o.mp = m.mp
    if (m.t != null) o.t = m.t
    return te.encode(JSON.stringify(o))
  }
  let name = m.u == null ? "" : String(m.u)
  let bytes = build(name)
  // trim by code point, never by byte — halving a multi-byte character would
  // produce JSON that does not decode
  while (bytes.length > MAX_META_BYTES && name.length > 0) {
    name = Array.from(name).slice(0, -1).join("")
    bytes = build(name)
  }
  if (bytes.length > MAX_META_BYTES) {
    throw new ShareError("meta will not fit in 255 bytes", "META_TOO_BIG")
  }
  return bytes
}

// ---------------------------------------------------------------------------
// the book trie
//
// Canonical shape, and what encode reads and decode writes:
//
//   TrieNode = { kids: Edge[] }
//   Edge     = { from: 0-63, to: 0-63, promo: 0-3, own: bool, node: TrieNode }
//
// buildTrie() also hangs `uci`, `n` and `san` on each edge for convenience;
// encode ignores those.

/**
 * Walk a FEN-keyed book (docs/book.json's shape) out from the start position
 * and return a trie of the lines its owner played at least `minCount` times.
 *
 * WHY THE TRIE HAS MOVES THE OWNER NEVER PLAYED
 *
 * The book only stores positions where its OWNER had the move, so a trie rooted
 * at the start cannot be built from his moves alone — with him as Black, the
 * whole first ply is the opponent's. So at every node the edge set is the union
 * of two things:
 *
 *   his book moves from this position          (own = true), and
 *   every legal move landing on a book position (own = false).
 *
 * The second kind bridges the replies he faced, reconstructed from the positions
 * he actually reached. Those edges are structure, not opinion: they exist so the
 * trie can be walked, never so the bot can play them. `own` is what keeps the
 * two apart, and trieToMoveMap() only ever offers the bot an own edge.
 *
 * Bridging happens at his own positions too, not only at the opponent's. A
 * position key fixes the side to move but not which side HE was, so one key can
 * be his-move in a game he had White and opponent-to-move in a game he had
 * Black. Measured: dropping the bridges at his own positions costs 174 of 266
 * positions at the >=2x threshold. Keeping them and flagging them costs nothing.
 *
 * WHY PRUNING HAPPENS DURING THE WALK, NOT AFTER
 *
 * A line is not followed through a position he reached fewer than `minCount`
 * times, even when something further down that line is common by another move
 * order. That is path-wise pruning rather than aggregate, and it is a deliberate
 * choice twice over. It keeps the book on lines he genuinely repeated instead of
 * on transposition artifacts reached through a move he played once. And it is
 * what makes this runnable in a browser: building the whole tree and pruning
 * afterwards measured 28 SECONDS on the owner's book against 0.8s this way, and
 * at the URL budget it delivered FEWER positions (239 at >=3x, against 266 at
 * >=2x here) because it spent its bytes on the rare paths.
 *
 * Transpositions are expanded, since a trie is a tree. A key already on the
 * current path is not re-entered, which is what stops a repetition looping
 * forever.
 *
 * Returns null when nothing clears the threshold.
 */
export function buildTrie(bookMap, opts = {}) {
  const Chess = opts.Chess
  if (!Chess) throw new ShareError("buildTrie needs a Chess constructor", "NO_CHESS")
  const minCount = opts.minCount == null ? 2 : opts.minCount
  const maxDepth = opts.maxDepth == null ? 30 : opts.maxDepth
  const maxNodes = opts.maxNodes == null ? 100000 : opts.maxNodes
  const startFen = opts.startFen || START_FEN

  const count = opts.counts || countPositions(bookMap)
  const chess = new Chess(startFen)
  const onPath = new Set()
  let nodes = 0
  let truncated = false

  function rec(depth) {
    nodes++
    const key = bookKey(chess.fen())
    const node = { kids: [], n: count[key] || 0 }
    if (depth >= maxDepth || nodes >= maxNodes) {
      if (nodes >= maxNodes) truncated = true
      return node
    }
    onPath.add(key)
    const mine = bookMap[key] || null
    const candidates = []
    // one verbose move generation per node, and m.after gives the child position
    // for free — making and unmaking each move to read its FEN was most of the
    // 28 seconds
    for (const m of chess.moves({ verbose: true })) {
      const uci = m.from + m.to + (m.promotion || "")
      const own = mine && mine[uci] !== undefined ? mine[uci] : null
      const childKey = bookKey(m.after)
      const isBook = bookMap[childKey] !== undefined
      if (!own && !isBook) continue
      if (onPath.has(childKey)) continue
      if (isBook && (count[childKey] || 0) < minCount) continue
      candidates.push({ m, uci, n: own ? own.n || 0 : 0, own: !!own, san: m.san })
    }
    // descending play count, so the wire order records the owner's preference;
    // uci breaks ties so the encoding is deterministic
    candidates.sort((a, b) => b.n - a.n || (a.uci < b.uci ? -1 : 1))
    for (const c of candidates) {
      chess.move(c.m)
      const child = rec(depth + 1)
      chess.undo()
      node.kids.push({
        from: squareToIndex(c.uci.slice(0, 2)),
        to: squareToIndex(c.uci.slice(2, 4)),
        promo: c.uci.length > 4 ? PROMO_TO_CODE[c.uci[4]] : 0,
        own: c.own,
        uci: c.uci,
        san: c.san,
        n: c.n,
        node: child,
      })
    }
    onPath.delete(key)
    return node
  }

  // the walk keeps branches that turn out to be dead ends — a bridge into a
  // position that led nowhere above the threshold — so it still needs a prune
  const pruned = pruneTrie(rec(0), minCount)
  if (pruned) pruned.truncated = truncated
  return pruned
}

/** How many times the owner had the move at each book position. */
export function countPositions(bookMap) {
  const count = Object.create(null)
  for (const k in bookMap) {
    let s = 0
    for (const u in bookMap[k]) s += bookMap[k][u].n || 0
    count[k] = s
  }
  return count
}

/**
 * Keep only nodes reached at least `minCount` times, plus every node on a path
 * to one. Interior nodes where the opponent had the move have no count of their
 * own and survive purely as connectors. Returns null if nothing survives.
 */
export function pruneTrie(node, minCount) {
  const kids = []
  for (const e of node.kids) {
    const child = pruneTrie(e.node, minCount)
    if (child) kids.push({ ...e, node: child })
  }
  if (kids.length === 0 && (node.n || 0) < minCount) return null
  return { kids, n: node.n || 0 }
}

export function trieStats(node) {
  let nodes = 0, edges = 0, ownEdges = 0, positions = 0, maxDepth = 0
  ;(function go(x, d) {
    nodes++
    if ((x.n || 0) > 0) positions++
    if (d > maxDepth) maxDepth = d
    for (const e of x.kids) { edges++; if (e.own) ownEdges++; go(e.node, d + 1) }
  })(node, 0)
  // 1 byte per node, 2 per edge
  return { nodes, edges, ownEdges, positions, maxDepth, bytes: nodes + 2 * edges }
}

function writeTrie(node, out) {
  let kids = node.kids
  if (kids.length > 255) {
    // no real position comes close to 255 book moves, but a malformed trie must
    // not corrupt the byte stream — keep the owner's most-played and drop the
    // tail, since the children are already in preference order
    kids = kids.slice(0, 255)
  }
  out.push(kids.length)
  for (const e of kids) {
    const packed =
      (e.from & 63) | ((e.to & 63) << 6) | ((e.promo & 3) << 12) | (e.own ? 1 << 14 : 0)
    out.push(packed & 0xff, (packed >> 8) & 0xff)
    writeTrie(e.node, out)
  }
}

function readTrie(bytes, cursor) {
  if (cursor.i >= bytes.length) throw new ShareError("book trie ended early", "TRUNCATED_BOOK")
  const count = bytes[cursor.i++]
  const node = { kids: [] }
  for (let c = 0; c < count; c++) {
    if (cursor.i + 1 >= bytes.length) throw new ShareError("book trie ended early", "TRUNCATED_BOOK")
    const packed = bytes[cursor.i] | (bytes[cursor.i + 1] << 8)
    cursor.i += 2
    node.kids.push({
      from: packed & 63,
      to: (packed >> 6) & 63,
      promo: (packed >> 12) & 3,
      own: !!(packed & (1 << 14)),
      node: readTrie(bytes, cursor),
    })
  }
  return node
}

/**
 * Replay a decoded trie on a real board so its edges get their UCI, SAN and
 * position key back.
 *
 * This is where the 2-bit promotion field is resolved. Two bits cannot hold
 * five states (none, q, r, b, n) — but they do not have to. Whether a move
 * promotes is a property of the position, not of the move's encoding, so the
 * board tells us; the two bits then only have to choose among the four pieces.
 * That is why the reader needs a Chess constructor and the writer needed one too.
 *
 * Edges that are not legal in the position they were reached at are dropped.
 * That is the corruption backstop past the CRC: a payload that survives the
 * checksum but describes an impossible line loses the impossible part, rather
 * than handing the caller moves that will throw when played.
 */
export function hydrateTrie(trie, opts = {}) {
  const Chess = opts.Chess
  if (!Chess) throw new ShareError("hydrateTrie needs a Chess constructor", "NO_CHESS")
  const startFen = opts.startFen || START_FEN
  const chess = new Chess(startFen)
  let dropped = 0

  function rec(node) {
    const out = { key: bookKey(chess.fen()), fen: chess.fen(), kids: [] }
    const legal = new Map()
    for (const m of chess.moves({ verbose: true })) {
      legal.set(m.from + m.to + (m.promotion || ""), m)
    }
    for (const e of node.kids) {
      const from = indexToSquare(e.from)
      const to = indexToSquare(e.to)
      let m = legal.get(from + to)
      let uci = from + to
      if (!m) {
        // no quiet move here, so this must be the promoting form
        uci = from + to + CODE_TO_PROMO[e.promo]
        m = legal.get(uci)
      }
      if (!m) { dropped++; continue }
      chess.move(m)
      const child = rec(e.node)
      chess.undo()
      out.kids.push({ from: e.from, to: e.to, promo: e.promo, own: !!e.own, uci, san: m.san, node: child })
    }
    return out
  }

  const root = rec(trie)
  root.dropped = dropped
  return root
}

/**
 * Turn a decoded trie into the FEN-keyed map app.js's pickBookMove() consumes.
 *
 * ONLY OWNER EDGES BY DEFAULT. The trie also holds bridge edges, which exist to
 * walk the opponent's replies and were never the owner's choices; putting those
 * in the bot's book is how a quarter of its opening moves would become moves the
 * owner never made. Pass ownerOnly:false to get the navigation edges too, and
 * then do not hand the result to a move picker.
 *
 * The wire format has no room for the owner's real play counts, so `weight`
 * decides what to put in their place:
 *
 *   "rank"    — the wire order is the owner's preference order, so give the
 *               first child the most weight. Reconstructed from rank, NOT the
 *               owner's counts. Closer to how he actually chose than uniform.
 *   "uniform" — every book move equally likely. Invents nothing.
 *
 * w and d come back as 0 because they were never encoded. Anything showing a
 * score for a line off a shared book would be showing a number that is not there.
 */
export function trieToMoveMap(trie, opts = {}) {
  const hydrated = opts.hydrated || hydrateTrie(trie, opts)
  const weight = opts.weight || "rank"
  const ownerOnly = opts.ownerOnly !== false
  const map = Object.create(null)
  ;(function go(node) {
    const usable = ownerOnly ? node.kids.filter(e => e.own) : node.kids
    if (usable.length) {
      const entry = map[node.key] || (map[node.key] = Object.create(null))
      usable.forEach((e, i) => {
        entry[e.uci] = {
          san: e.san,
          n: weight === "uniform" ? 1 : usable.length - i,
          w: 0,
          d: 0,
        }
      })
    }
    for (const e of node.kids) go(e.node)
  })(hydrated)
  return map
}

// ---------------------------------------------------------------------------
// encode / decode

function assemble(payload, bookBytes, deflated) {
  const version = payload.version == null ? FORMAT_VERSION : payload.version
  if (!Number.isInteger(version) || version < 0 || version > 255) {
    throw new ShareError("format version must be a byte", "BAD_VERSION")
  }
  const { scale, ints } = quantiseWeights(payload.weights)
  const metaBytes = packMeta(payload.meta)
  const hasBook = bookBytes !== null && bookBytes.length > 0

  const total = META_OFFSET + 1 + metaBytes.length + (hasBook ? bookBytes.length : 0) + CRC_BYTES
  const out = new Uint8Array(total)
  const view = new DataView(out.buffer)

  out[0] = version
  out[1] = (hasBook ? FLAG_HAS_BOOK : 0) | (hasBook && deflated ? FLAG_BOOK_DEFLATED : 0)
  view.setUint16(2, scale, true)
  for (let i = 0; i < N_FEATURES; i++) view.setInt16(HEADER_BYTES + i * 2, ints[i], true)
  out[META_OFFSET] = metaBytes.length
  out.set(metaBytes, META_OFFSET + 1)
  if (hasBook) out.set(bookBytes, META_OFFSET + 1 + metaBytes.length)
  view.setUint16(total - CRC_BYTES, crc16(out, total - CRC_BYTES), true)
  return out
}

function serialiseBook(book) {
  if (!book) return null
  const out = []
  writeTrie(book, out)
  return Uint8Array.from(out)
}

/**
 * Pack a bot into a base64url string. Synchronous, and never deflates — see
 * encodeAsync() for the deflate attempt.
 *
 *   payload.version  wire format version (defaults to FORMAT_VERSION)
 *   payload.weights  exactly N_FEATURES numbers
 *   payload.meta     {u, s, r, g, v}
 *   payload.book     a trie (buildTrie), or null for no book
 */
export function encode(payload) {
  return toBase64Url(assemble(payload, serialiseBook(payload.book), false))
}

async function deflateRaw(bytes) {
  if (typeof CompressionStream === "undefined") return null
  try {
    const cs = new CompressionStream("deflate-raw")
    const w = cs.writable.getWriter()
    w.write(bytes)
    w.close()
    const buf = await new Response(cs.readable).arrayBuffer()
    return new Uint8Array(buf)
  } catch {
    return null
  }
}

async function inflateRaw(bytes) {
  if (typeof DecompressionStream === "undefined") {
    throw new ShareError("this browser cannot read a compressed book", "NO_INFLATE")
  }
  const ds = new DecompressionStream("deflate-raw")
  const w = ds.writable.getWriter()
  w.write(bytes)
  w.close()
  const buf = await new Response(ds.readable).arrayBuffer()
  return new Uint8Array(buf)
}

/**
 * encode(), but it also tries deflate-raw on the book and keeps it only when it
 * actually comes out smaller. Weights are never deflated: measured, deflate
 * makes them bigger, because quantised floats are near-maximum-entropy.
 */
export async function encodeAsync(payload) {
  const raw = serialiseBook(payload.book)
  if (!raw || raw.length === 0) return toBase64Url(assemble(payload, raw, false))
  const packed = await deflateRaw(raw)
  if (packed && packed.length < raw.length) {
    return toBase64Url(assemble(payload, packed, true))
  }
  return toBase64Url(assemble(payload, raw, false))
}

function parse(bytes) {
  if (bytes.length < META_OFFSET + 1 + CRC_BYTES) {
    throw new ShareError("link is too short to be a bot", "TOO_SHORT")
  }
  const stated = bytes[bytes.length - 2] | (bytes[bytes.length - 1] << 8)
  const actual = crc16(bytes, bytes.length - CRC_BYTES)
  if (stated !== actual) {
    throw new ShareError("link is damaged (checksum failed)", "BAD_CRC")
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const version = bytes[0]
  if (version !== FORMAT_VERSION) {
    throw new ShareError("link was made by a different version of Mirror Bot", "BAD_VERSION")
  }
  const flags = bytes[1]
  const scale = view.getUint16(2, true)
  const ints = new Int16Array(N_FEATURES)
  for (let i = 0; i < N_FEATURES; i++) ints[i] = view.getInt16(HEADER_BYTES + i * 2, true)
  const metaLen = bytes[META_OFFSET]
  const metaStart = META_OFFSET + 1
  const metaEnd = metaStart + metaLen
  if (metaEnd > bytes.length - CRC_BYTES) {
    throw new ShareError("link is damaged (meta runs past the end)", "BAD_META")
  }
  let meta
  try {
    meta = metaLen ? JSON.parse(td.decode(bytes.subarray(metaStart, metaEnd))) : {}
  } catch {
    throw new ShareError("link is damaged (meta is not readable)", "BAD_META")
  }
  return {
    version,
    flags,
    weights: dequantiseWeights(scale, ints),
    scale,
    meta,
    bookBytes: flags & FLAG_HAS_BOOK ? bytes.subarray(metaEnd, bytes.length - CRC_BYTES) : null,
    deflated: !!(flags & FLAG_BOOK_DEFLATED),
  }
}

function finish(p, bookBytes) {
  const out = { version: p.version, weights: p.weights, meta: p.meta, book: null }
  if (bookBytes) {
    const cursor = { i: 0 }
    out.book = readTrie(bookBytes, cursor)
  }
  return out
}

/**
 * Unpack a base64url string back into { version, weights, meta, book }.
 *
 * `book` comes back as the canonical trie — edges carry square indices and a
 * promotion code, not UCI, because resolving a promotion needs the board. Run
 * it through hydrateTrie() or trieToMoveMap() with a Chess constructor to get
 * playable moves.
 *
 * Synchronous. Throws ShareError("...", "DEFLATED") for a compressed book,
 * which only encodeAsync() produces — use decodeAsync() to read those.
 */
export function decode(str) {
  const p = parse(fromBase64Url(str))
  if (p.deflated) {
    throw new ShareError("this book is compressed — use decodeAsync()", "DEFLATED")
  }
  return finish(p, p.bookBytes)
}

/** decode(), and it can also read a deflated book. */
export async function decodeAsync(str) {
  const p = parse(fromBase64Url(str))
  const book = p.bookBytes && p.deflated ? await inflateRaw(p.bookBytes) : p.bookBytes
  return finish(p, book)
}

// ---------------------------------------------------------------------------
// URLs

function baseOf(base) {
  if (base) return String(base).split("#")[0]
  if (typeof location !== "undefined") return location.href.split("#")[0]
  throw new ShareError("no base URL to build a share link from", "NO_BASE")
}

/** Build the full share URL. The payload goes in the fragment, which is never
 *  sent to any server — not to GitHub Pages, not to anyone. */
export function buildShareURL(payload, base) {
  return baseOf(base) + "#" + encode(payload)
}

export async function buildShareURLAsync(payload, base) {
  return baseOf(base) + "#" + (await encodeAsync(payload))
}

/** Pull the payload text out of a URL or a bare fragment. Accepts "#<payload>"
 *  and "#b=<payload>"; we emit the bare form because it is two characters
 *  cheaper, but accept the keyed one so a future page can add real anchors.
 *
 *  With no "#" at all the argument is only treated as a bare payload when it
 *  actually looks like one. A URL with no fragment is not a payload, and
 *  returning the whole URL as if it were is how "no bot in this link" turns
 *  into a checksum error. */
export function fragmentOf(urlOrHash) {
  const s = String(urlOrHash == null ? "" : urlOrHash)
  const at = s.indexOf("#")
  if (at < 0) return /^[A-Za-z0-9_-]+$/.test(s) ? s : ""
  const hash = s.slice(at + 1)
  if (!hash) return ""
  const keyed = /^b=(.*)$/.exec(hash)
  return keyed ? keyed[1] : hash
}

/** Read a shared bot back off a URL. Returns null when there is no fragment;
 *  throws ShareError when there is one and it is broken. */
export function readShareFromURL(url) {
  const frag = fragmentOf(url)
  if (!frag) return null
  return decode(frag)
}

/** As readShareFromURL, and it can also read a deflated book. This is the one
 *  a page should call on load, since it handles every link encode can make. */
export async function readShareFromLocation(loc) {
  const l = loc || (typeof location !== "undefined" ? location : null)
  if (!l) throw new ShareError("no location to read", "NO_LOCATION")
  const frag = fragmentOf(l.hash || "")
  if (!frag) return null
  return decodeAsync(frag)
}

// ---------------------------------------------------------------------------
// fitting a book into the budget

/**
 * Pack the biggest book that still fits under MAX_URL_CHARS.
 *
 * Starts at "reached at least twice" and raises the threshold until the whole
 * URL fits. If even the harshest threshold is too big, ships the bot with no
 * book at all rather than no link — the weights are the bot; the book is the
 * opening manners. If the weights alone do not fit, that is a real failure and
 * it throws.
 *
 * The trie is rebuilt at each threshold rather than pruned down from one big
 * tree, because buildTrie prunes path-wise as it walks and the two are not the
 * same tree. Rebuilding is also the cheap direction: the first build dominates
 * and every later one is smaller (measured 0.77s, then 0.25s, then 0.14s).
 *
 * Pass `bookMap` for the FEN-keyed book, or `book` for a trie you already built
 * (which is then used as-is, at no threshold).
 *
 * Returns { url, chars, threshold, payload, stats, tried } where `tried` is the
 * measurement for every threshold attempted, so a caller can show its work.
 */
export function fitShare(payload, opts = {}) {
  const base = baseOf(opts.base)
  const max = opts.max == null ? MAX_URL_CHARS : opts.max
  const start = opts.startThreshold == null ? 2 : opts.startThreshold
  const ceiling = opts.maxThreshold == null ? 64 : opts.maxThreshold

  const tried = []
  const attempt = (book, threshold) => {
    const p = { version: payload.version, weights: payload.weights, meta: payload.meta, book }
    const url = base + "#" + encode(p)
    const row = { threshold, chars: url.length, fits: url.length <= max }
    if (book) Object.assign(row, trieStats(book))
    tried.push(row)
    return { url, payload: p, row }
  }

  if (payload.bookMap) {
    const counts = countPositions(payload.bookMap)
    for (let t = start; t <= ceiling; t++) {
      const trie = buildTrie(payload.bookMap, { ...opts, counts, minCount: t })
      if (!trie) break
      const a = attempt(trie, t)
      if (a.row.fits) {
        return { url: a.url, chars: a.row.chars, threshold: t, payload: a.payload, stats: trieStats(trie), tried }
      }
    }
  } else if (payload.book) {
    const a = attempt(payload.book, null)
    if (a.row.fits) {
      return { url: a.url, chars: a.row.chars, threshold: null, payload: a.payload, stats: trieStats(payload.book), tried }
    }
  }

  const bare = attempt(null, Infinity)
  if (!bare.row.fits) {
    throw new ShareError(
      "the weights alone need " + bare.row.chars + " characters, over the " + max + " budget",
      "OVER_BUDGET"
    )
  }
  return { url: bare.url, chars: bare.row.chars, threshold: null, payload: bare.payload, stats: null, tried }
}

/**
 * fitShare, but each candidate is measured through encodeAsync — so the book
 * gets its deflate attempt and the link that comes out is the one the page
 * should actually hand the user.
 *
 * This is what a share button wants. On the owner's book it is the difference
 * between a 1,788-character link and a 903-character one for the same book, and
 * on a bigger book it is the difference between keeping the >=2x threshold and
 * being forced up to >=3x or >=4x. The synchronous fitShare stays for callers
 * that cannot await.
 */
export async function fitShareAsync(payload, opts = {}) {
  const base = baseOf(opts.base)
  const max = opts.max == null ? MAX_URL_CHARS : opts.max
  const start = opts.startThreshold == null ? 2 : opts.startThreshold
  const ceiling = opts.maxThreshold == null ? 64 : opts.maxThreshold

  const tried = []
  const attempt = async (book, threshold) => {
    const p = { version: payload.version, weights: payload.weights, meta: payload.meta, book }
    const url = base + "#" + (await encodeAsync(p))
    const row = { threshold, chars: url.length, fits: url.length <= max }
    if (book) Object.assign(row, trieStats(book))
    tried.push(row)
    return { url, payload: p, row }
  }

  if (payload.bookMap) {
    const counts = countPositions(payload.bookMap)
    for (let t = start; t <= ceiling; t++) {
      const trie = buildTrie(payload.bookMap, { ...opts, counts, minCount: t })
      if (!trie) break
      const a = await attempt(trie, t)
      if (a.row.fits) {
        return { url: a.url, chars: a.row.chars, threshold: t, payload: a.payload, stats: trieStats(trie), tried }
      }
    }
  } else if (payload.book) {
    const a = await attempt(payload.book, null)
    if (a.row.fits) {
      return { url: a.url, chars: a.row.chars, threshold: null, payload: a.payload, stats: trieStats(payload.book), tried }
    }
  }

  const bare = await attempt(null, Infinity)
  if (!bare.row.fits) {
    throw new ShareError(
      "the weights alone need " + bare.row.chars + " characters, over the " + max + " budget",
      "OVER_BUDGET"
    )
  }
  return { url: bare.url, chars: bare.row.chars, threshold: null, payload: bare.payload, stats: null, tried }
}
