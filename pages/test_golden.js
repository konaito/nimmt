/* Python実装が書き出した golden.json と JS 移植の数値照合。
   Usage: node pages/test_golden.js */

"use strict";

const fs = require("fs");
const path = require("path");
const E = require("./features.js");
const { NimmtNetJS } = require("./net.js");

const dir = __dirname;
const nets = {
  4: new NimmtNetJS(JSON.parse(fs.readFileSync(path.join(dir, "weights_exp4.json")))),
  2: new NimmtNetJS(JSON.parse(fs.readFileSync(path.join(dir, "weights_exp2.json")))),
};
const cases = JSON.parse(fs.readFileSync(path.join(dir, "golden.json")));

let bad = 0;
let maxGerr = 0, maxLerr = 0;
for (const [ci, c] of cases.entries()) {
  const seen = new Array(105).fill(false);
  for (const s of c.state.seen) seen[s] = true;
  const view = {
    rows: c.state.rows,
    handSlots: c.state.hand_slots[c.seat],
    seen,
    taken: c.state.taken,
    turn: c.state.turn,
    players: c.state.players,
    seat: c.seat,
    playedCard: c.kind === "row" ? c.pending_card : undefined,
  };
  const g = E.globalFeatures(view);
  for (let i = 0; i < 266; i++) {
    const err = Math.abs(g[i] - c.g[i]);
    maxGerr = Math.max(maxGerr, err);
    if (err > 1e-5) { console.log(`case ${ci} (${c.kind}): g[${i}] js=${g[i]} py=${c.g[i]}`); bad++; break; }
  }
  const net = nets[c.players];
  const h = net.encode(g);
  let logits, pyLogits;
  if (c.kind === "card") {
    const { feats, mask } = E.cardFeatures(view);
    logits = net.cardLogits(h, feats, mask);
    pyLogits = c.logits.map((x) => (x === null ? -Infinity : x));
  } else {
    logits = net.rowLogits(h, E.rowFeatures(view));
    pyLogits = c.logits;
  }
  let ok = true;
  for (let i = 0; i < logits.length; i++) {
    if (!isFinite(pyLogits[i]) && !isFinite(logits[i])) continue;
    const err = Math.abs(logits[i] - pyLogits[i]);
    maxLerr = Math.max(maxLerr, err);
    if (err > 1e-3) { console.log(`case ${ci} (${c.kind}): logit[${i}] js=${logits[i]} py=${pyLogits[i]}`); ok = false; }
  }
  const argmax = logits.indexOf(Math.max(...logits.filter(isFinite)));
  if (argmax !== c.argmax) { console.log(`case ${ci} (${c.kind}): argmax js=${argmax} py=${c.argmax}`); ok = false; }
  if (!ok) bad++;
}
console.log(`cases=${cases.length} failures=${bad} max_g_err=${maxGerr.toExponential(2)} max_logit_err=${maxLerr.toExponential(2)}`);
process.exit(bad ? 1 : 0);
