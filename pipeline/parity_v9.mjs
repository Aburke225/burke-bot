// Prove every copy of the v9 contract agrees.
//
// pipeline/features_v9.py and the v9 block in docs/app.js must produce
// identical vectors for the same position, or the bot plays a different game
// than the one it was trained on. 33 of the 57 features are new, and a skew
// here produces legal, plausible, WRONG moves with nothing in the training
// metrics catching it - so this runs the REAL source out of app.js rather than
// a copy, extracted by brace-matching.
//
// THERE IS NOW A THIRD COPY: docs/build-a-bot/v9.js, the module the Build-a-Bot page
// uses to build a bot from a stranger's games. It is checked here against the
// same python-generated vectors, so one run proves all three agree. If it is
// absent the check is skipped and the original two-way guarantee is unchanged -
// this file must never start failing because of a folder it does not require.
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
             "horizonScores", "aimlessEdgePawnV9", "moveFeaturesV9"]
const src = [...consts.map(c => extract(c, "const")), ...fns.map(f => extract(f)),
             "return { decisionContextV9, moveFeaturesV9, horizonScores, V9_N }"].join("\n")

const chess = new Chess()
const api = new Function("chess", src)(chess)

// The third copy is a real ES module with explicit exports, so it imports
// rather than needing the brace-extraction above.
let bab = null
try {
  bab = await import("../docs/build-a-bot/v9.js")
} catch (e) {
  if (e.code !== "ERR_MODULE_NOT_FOUND") throw e
}

// docs/build-a-bot/v9.js takes the previous-move facts explicitly instead of digging
// them out of module state the way app.js does. Passing nulls when history DOES
// exist silently changes features 12, 13, 40, 43 and 44 and raises no error, so
// this mapping is the load-bearing part of the third-copy check.
function babContext() {
  const hist = chess.history({ verbose: true })
  const oppLast = hist.length ? hist[hist.length - 1] : null
  const myLast = hist.length > 1 ? hist[hist.length - 2] : null
  const mine = myLast && myLast.color === chess.turn() ? myLast : null
  return bab.makeContext(
    chess,
    mine ? mine.to : null,
    mine ? mine.from : null,
    oppLast && oppLast.captured ? oppLast.to : null,
    oppLast ? oppLast.to : null,
  )
}

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"))
let checked = 0, bad = 0
const perFeature = new Array(api.V9_N).fill(0)
let lastKey = null, ctx = null, mctx = null
let mChecked = 0, mBad = 0
const mPerFeature = new Array(api.V9_N).fill(0)

for (const c of cases) {
  // a case is either a move list (history intact, so the context has a real
  // previous move) or a bare FEN, used for endgames that would take forty
  // moves of SAN to reach. With a FEN there is no history and the context's
  // previous-move fields are null on BOTH sides, which is the point.
  const key = c.fen ? "fen:" + c.fen : c.moves.join(" ")
  if (key !== lastKey) {
    if (c.fen) {
      chess.load(c.fen)
    } else {
      chess.reset()
      for (const u of c.moves) {
        chess.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u.slice(4) || undefined })
      }
    }
    ctx = api.decisionContextV9()
    if (bab) mctx = babContext()
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
  if (bab) {
    const mx = bab.features(chess, c.uci, mctx, c.sh, c.bestSh, c.rank)
    mChecked++
    if (!mx) {
      mBad++
      console.log("BUILD-A-BOT NULL vector for", c.uci)
    } else {
      let mm = false
      for (let k = 0; k < api.V9_N; k++) {
        if (Math.abs(mx[k] - c.x[k]) > 1e-6) { mPerFeature[k]++; mm = true }
      }
      if (mm) {
        mBad++
        if (mBad <= 3) {
          const d = []
          for (let k = 0; k < api.V9_N; k++) {
            if (Math.abs(mx[k] - c.x[k]) > 1e-6) d.push(`${k}: bab ${mx[k]} vs py ${c.x[k]}`)
          }
          console.log(`BUILD-A-BOT MISMATCH ${c.uci} in ${c.fen || c.moves.length + " plies"} -> ${d.join(", ")}`)
        }
      }
    }
  }
  if (mismatch) {
    bad++
    if (bad <= 3) {
      const diffs = []
      for (let k = 0; k < api.V9_N; k++) {
        if (Math.abs(x[k] - c.x[k]) > 1e-6) diffs.push(`${k}: js ${x[k]} vs py ${c.x[k]}`)
      }
      console.log(`MISMATCH ${c.uci} in ${c.fen || c.moves.length + " plies"} -> ${diffs.join(", ")}`)
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

if (!bab) {
  console.log("docs/build-a-bot/v9.js absent - third copy not checked")
} else {
  console.log(`\ndocs/build-a-bot/v9.js: ${mChecked} vectors checked, ${mBad} mismatching`)
  const mo = mPerFeature.map((n, k) => [k, n]).filter(([, n]) => n > 0)
  if (mo.length) {
    console.log("build-a-bot mismatches by feature:", mo.map(([k, n]) => `${k}:${n}`).join("  "))
  } else {
    console.log("the third copy agrees with python to 1e-6")
  }
}

// a non-zero exit so the python wrapper cannot pass on a string match alone
process.exitCode = bad || mBad ? 1 : 0
