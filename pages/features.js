/* src/nimmt/features.py の JS 移植（1局面・1視点分）。
   数値は Python 実装と golden テストで照合する。 */

"use strict";

const N_CARDS = 104, N_ROWS = 4, ROW_CAP = 5, HAND_SIZE = 10;
const G_DIM = 266, C_DIM = 12, R_DIM = 9;
const BULL_SCALE = 14.0, TAKEN_SCALE = 20.0;

function bullheads(c) {
  if (c === 55) return 7;
  if (c % 11 === 0) return 5;
  if (c % 10 === 0) return 3;
  if (c % 5 === 0) return 2;
  return 1;
}

/* view: {rows: [[cards...]x4], handSlots: int[10] (0=出済み), seen: bool[105],
          taken: int[P], turn, players, seat, playedCard(optional, row相のみ)} */

function rowEnds(rows) { return rows.map((r) => r[r.length - 1]); }
function rowBulls(rows) { return rows.map((r) => r.reduce((s, c) => s + bullheads(c), 0)); }

function globalFeatures(view) {
  const out = new Float32Array(G_DIM);
  const ends = rowEnds(view.rows), bulls = rowBulls(view.rows);
  for (let r = 0; r < N_ROWS; r++) {
    const len = view.rows[r].length, o = r * 8;
    out[o] = ends[r] / N_CARDS;
    out[o + 1] = len / ROW_CAP;
    out[o + 2] = bulls[r] / BULL_SCALE;
    out[o + 3 + (len - 1)] = 1.0;
  }
  for (const c of view.handSlots) if (c > 0) out[32 + c - 1] = 1.0;
  for (let c = 1; c <= N_CARDS; c++) if (view.seen[c]) out[136 + c - 1] = 1.0;
  out[240] = view.taken[view.seat] / TAKEN_SCALE;
  const others = view.taken.filter((_, p) => p !== view.seat).sort((a, b) => b - a).slice(0, 3);
  for (let i = 0; i < others.length; i++) out[241 + i] = others[i] / TAKEN_SCALE;
  out[244 + Math.min(view.turn, HAND_SIZE - 1)] = 1.0;
  out[254] = view.players / 10.0;
  out[255 + Math.max(view.players - 2, 0)] = 1.0;
  if (view.playedCard !== undefined && view.playedCard !== null) {
    out[264] = view.playedCard / N_CARDS;
    out[265] = bullheads(view.playedCard) / 7.0;
  }
  return out;
}

function placementInfo(view) {
  const ends = rowEnds(view.rows), bulls = rowBulls(view.rows);
  const minBull = Math.min(...bulls);
  // 未見カード累積（自分の手札は除外）
  const unseen = new Int32Array(N_CARDS + 1);
  for (let c = 1; c <= N_CARDS; c++) unseen[c] = view.seen[c] ? 0 : 1;
  for (const c of view.handSlots) if (c > 0) unseen[c] = 0;
  const prefix = new Int32Array(N_CARDS + 1);
  for (let c = 1; c <= N_CARDS; c++) prefix[c] = prefix[c - 1] + unseen[c];
  const nUnseen = Math.max(prefix[N_CARDS], 1);

  return view.handSlots.map((card, slot) => {
    if (card <= 0) return null;
    let tgt = -1, tEnd = -1;
    for (let r = 0; r < N_ROWS; r++) {
      if (ends[r] < card && ends[r] > tEnd) { tgt = r; tEnd = ends[r]; }
    }
    const belowAll = tgt < 0;
    const tLen = belowAll ? 0 : view.rows[tgt].length;
    const tBull = belowAll ? 0 : bulls[tgt];
    const isSixth = !belowAll && tLen === ROW_CAP;
    const takeNow = belowAll ? minBull : (isSixth ? tBull : 0);
    const gap = belowAll ? 0 : card - tEnd;
    const lower = prefix[Math.max(card - 1, 0)];
    const inGap = belowAll ? 0 : lower - prefix[Math.min(Math.max(tEnd, 0), N_CARDS)];
    return { card, slot, tgt, belowAll, tLen, tBull, isSixth, takeNow, gap,
             lowerFrac: lower / nUnseen, inGapFrac: inGap / nUnseen };
  });
}

function cardFeatures(view) {
  const info = placementInfo(view);
  const feats = [], mask = [];
  for (let slot = 0; slot < HAND_SIZE; slot++) {
    const f = new Float32Array(C_DIM);
    const i = info[slot];
    if (i) {
      const hasTarget = i.belowAll ? 0 : 1;
      f[0] = i.card / N_CARDS;
      f[1] = bullheads(i.card) / 7.0;
      f[2] = (i.tLen / ROW_CAP) * hasTarget;
      f[3] = (i.tBull / BULL_SCALE) * hasTarget;
      f[4] = i.gap / N_CARDS;
      f[5] = i.isSixth ? 1 : 0;
      f[6] = i.belowAll ? 1 : 0;
      f[7] = ((ROW_CAP - i.tLen) / ROW_CAP) * hasTarget;
      f[8] = i.takeNow / BULL_SCALE;
      f[9] = i.lowerFrac;
      f[10] = i.inGapFrac;
      f[11] = slot / HAND_SIZE;
    }
    feats.push(f);
    mask.push(!!i);
  }
  return { feats, mask };
}

function rowFeatures(view) {
  const ends = rowEnds(view.rows), bulls = rowBulls(view.rows);
  const minB = Math.min(...bulls);
  const out = [];
  for (let r = 0; r < N_ROWS; r++) {
    const f = new Float32Array(R_DIM);
    const len = view.rows[r].length;
    f[0] = len / ROW_CAP;
    f[1] = bulls[r] / BULL_SCALE;
    f[2] = ends[r] / N_CARDS;
    f[3 + (len - 1)] = 1.0;
    f[8] = bulls[r] === minB ? 1 : 0;
    out.push(f);
  }
  return out;
}

const FeaturesJS = { globalFeatures, cardFeatures, rowFeatures, bullheads,
                     N_CARDS, N_ROWS, ROW_CAP, HAND_SIZE };
if (typeof module !== "undefined") module.exports = FeaturesJS;
if (typeof globalThis !== "undefined") globalThis.FeaturesJS = FeaturesJS;
