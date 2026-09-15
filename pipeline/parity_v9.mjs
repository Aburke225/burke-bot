// Prove the two halves of the v9 contract agree.
//
// pipeline/features_v9.py and the v9 block in docs/app.js must produce
// identical vectors for the same position, or the bot plays a different game
// than the one it was trained on. 33 of the 57 features are new, and a skew
// here produces legal, plausible, WRONG moves with nothing in the training
// metrics catching it - so this runs the REAL source out of app.js rather than
// a copy, extracted by brace-matching.
//
//   node pipeline/parity_v9.mjs /tmp/v9_parity_cases.json
import { readFileSync } from "fs"
import { Chess } from "../docs/vendor/chess.js"

const app = readFileSync(new URL("../docs/app.js", import.meta.url), "utf8")
function extract(name, kind = "function") {
  const needle = kind === "function" ? `function ${name}(` : `const ${name} =`
  const start = app.indexOf(needle)
  if (start < 0) throw new Error(`${name} not found in app.js`)
  if (kind === "const") {
    // a const may span lines (V9_HOME), so brace-match when one is open
    const nl = app.indexOf("\n", start)
    const firstBrace = app.indexOf("{", start)
    if (firstBrace < 0 || firstBrace > nl) return app.slice(start, nl)
    let d = 0
    for (let j = firstBrace; j < app.length; j++) {
      if (app[j] === "{") d++
      if (app[j] === "}" && --d === 0) return app.slice(start, j + 1)
    }
    throw new Error(`unterminated const ${name}`)
  }
  let depth = 0
  const open = app.indexOf("{", start)
  for (let j = open; j < app.length; j++) {
    if (app[j] === "{") depth++
    if (app[j] === "}" && --depth === 0) return app.slice(start, j + 1)
  }
  throw new Error(`unterminated ${name}`)
}

const consts = ["V9_N", "V9_PIECE_VAL", "V9_ATTACKER_VAL", "V9_ORDER", "V9_HOME",
                "V9_FIANCHETTO", "sqFile", "sqRank", "mkSq", "cheb"]
const fns = ["tryMove", "enPriseV9", "neighbours", "decisionContextV9",
             "horizonScores", "moveFeaturesV9"]
const src = [...consts.map(c => extract(c, "const")), ...fns.map(f => extract(f)),
             "return { decisionContextV9, moveFeaturesV9, horizonScores, V9_N }"].join("\n")

const chess = new Chess()
const api = new Function("chess", src)(chess)

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"))
let checked = 0, bad = 0
const perFeature = new Array(api.V9_N).fill(0)
let lastKey = null, ctx = null

for (const c of cases) {
  const key = c.moves.join(" ")
  if (key !== lastKey) {
    chess.reset()
    for (const u of c.moves) {
      chess.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u.slice(4) || undefined })
    }
    ctx = api.decisionContextV9()
    lastKey = key
  }
  const x = api.moveFeaturesV9(c.uci, ctx, c.sh, c.bestSh, c.rank)
  checked++
  if (!x) { bad++; console.log("NULL vector for", c.uci); continue }
  let mismatch = false
  for (let k = 0; k < api.V9_N; k++) {
    if (Math.abs(x[k] - c.x[k]) > 1e-6) {
      perFeature[k]++
      mismatch = true
    }
  }
  if (mismatch) {
    bad++
    if (bad <= 3) {
      const diffs = []
      for (let k = 0; k < api.V9_N; k++) {
        if (Math.abs(x[k] - c.x[k]) > 1e-6) diffs.push(`${k}: js ${x[k]} vs py ${c.x[k]}`)
      }
      console.log(`MISMATCH ${c.uci} after ${c.moves.length} plies -> ${diffs.join(", ")}`)
    }
  }
}
console.log(`\n${checked} candidate vectors checked, ${bad} mismatching`)
const offenders = perFeature.map((n, k) => [k, n]).filter(([, n]) => n > 0)
if (offenders.length) {
  console.log("mismatches by feature index:", offenders.map(([k, n]) => `${k}:${n}`).join("  "))
} else {
  console.log("every feature agrees to 1e-6")
}
