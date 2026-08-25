/* ニムトのルールエンジン + AI（ブラウザ内で完結）。
   webapp/server.py の Match と同じ API 形状（state / events）を返すので UI を共用できる。 */

"use strict";

/* global FeaturesJS, NimmtNetJS */

const E = FeaturesJS;

function shuffledDeck() {
  const d = Array.from({ length: 104 }, (_, i) => i + 1);
  for (let i = d.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [d[i], d[j]] = [d[j], d[i]];
  }
  return d;
}

class LocalMatch {
  constructor(players, net) {
    this.players = players;
    this.net = net;
    this.total = new Array(players).fill(0);
    this.dealNo = 0;
    this.target = 66;
    this.matchOver = false;
    this._newDeal();
  }

  _newDeal() {
    this.dealNo += 1;
    const deck = shuffledDeck();
    this.handSlots = [];
    for (let p = 0; p < this.players; p++) {
      this.handSlots.push(deck.slice(p * 10, p * 10 + 10).sort((a, b) => a - b));
    }
    this.rows = [];
    this.seen = new Array(105).fill(false);
    for (let r = 0; r < 4; r++) {
      const c = deck[this.players * 10 + r];
      this.rows.push([c]);
      this.seen[c] = true;
    }
    this.dealTaken = new Array(this.players).fill(0);
    this.turn = 0;
    this.dealOver = false;
    this.pending = null;   // {queue, idx, played} 解決中の状態
  }

  _view(seat, playedCard) {
    return { rows: this.rows, handSlots: this.handSlots[seat], seen: this.seen,
             taken: this.dealTaken, turn: this.turn, players: this.players,
             seat, playedCard };
  }

  _aiCard(seat) {
    const view = this._view(seat);
    const g = E.globalFeatures(view);
    const { feats, mask } = E.cardFeatures(view);
    const h = this.net.encode(g);
    const lg = this.net.cardLogits(h, feats, mask);
    let best = -1, bestV = -Infinity;
    for (let i = 0; i < lg.length; i++) if (lg[i] > bestV) { bestV = lg[i]; best = i; }
    return best;
  }

  _aiRow(seat, card) {
    const view = this._view(seat, card);
    const g = E.globalFeatures(view);
    const rf = E.rowFeatures(view);
    const h = this.net.encode(g);
    const lg = this.net.rowLogits(h, rf);
    let best = 0, bestV = -Infinity;
    for (let r = 0; r < 4; r++) if (lg[r] > bestV) { bestV = lg[r]; best = r; }
    return best;
  }

  rowPoints(r) { return this.rows[r].reduce((s, c) => s + E.bullheads(c), 0); }

  _place(player, card, row, forced) {
    const before = [...this.rows[row]];
    if (forced || this.rows[row].length >= 5) {
      const pts = this.rowPoints(row);
      this.rows[row] = [card];
      this.dealTaken[player] += pts;
      return { type: "take", player, card, row, points: pts, row_before: before, forced };
    }
    this.rows[row].push(card);
    return { type: "place", player, card, row, row_before: before };
  }

  /* 解決を進める。人間の行選択が必要になったら events を返して中断 */
  _resolve(events) {
    const q = this.pending;
    while (q.idx < q.queue.length) {
      const { player, card } = q.queue[q.idx];
      const ends = this.rows.map((r) => r[r.length - 1]);
      let tgt = -1, tEnd = -1;
      for (let r = 0; r < 4; r++) if (ends[r] < card && ends[r] > tEnd) { tgt = r; tEnd = ends[r]; }
      if (tgt >= 0) {
        events.push(this._place(player, card, tgt, false));
        q.idx += 1;
        continue;
      }
      if (player === 0) {
        this.waitingCard = card;
        return events;              // 人間の行選択待ち
      }
      const row = this._aiRow(player, card);
      events.push(this._place(player, card, row, true));
      q.idx += 1;
    }
    // ターン終了
    this.pending = null;
    this.turn += 1;
    if (this.handSlots[0].every((c) => c === 0)) {
      this.dealOver = true;
      for (let p = 0; p < this.players; p++) this.total[p] += this.dealTaken[p];
      if (Math.max(...this.total) >= this.target) this.matchOver = true;
    }
    return events;
  }

  playCard(slot) {
    if (this.pending || this.dealOver || this.matchOver) throw new Error("今は出せない");
    if (!(slot >= 0 && slot < 10) || this.handSlots[0][slot] <= 0) throw new Error("そのスロットは出せない");
    const played = [];
    played[0] = this.handSlots[0][slot];
    this.handSlots[0][slot] = 0;
    for (let p = 1; p < this.players; p++) {
      const s = this._aiCard(p);
      played[p] = this.handSlots[p][s];
      this.handSlots[p][s] = 0;
    }
    for (const c of played) this.seen[c] = true;
    const queue = played
      .map((card, player) => ({ card, player }))
      .sort((a, b) => a.card - b.card);
    this.pending = { queue, idx: 0 };
    const events = [{ type: "reveal", cards: played }];
    return this._resolve(events);
  }

  chooseRow(row) {
    if (!this.pending || this.waitingCard === undefined) throw new Error("行選択待ちではない");
    const card = this.waitingCard;
    delete this.waitingCard;
    const events = [this._place(0, card, row, true)];
    this.pending.idx += 1;
    return this._resolve(events);
  }

  nextDeal() {
    if (!this.dealOver || this.matchOver) throw new Error("ディールは終了していない");
    this._newDeal();
  }

  state() {
    const waiting = this.pending !== null && this.waitingCard !== undefined;
    return {
      match_id: "local",
      players: this.players,
      deal_no: this.dealNo,
      turn: this.turn,
      rows: this.rows.map((r) => r.map((c) => ({ card: c, bulls: E.bullheads(c) }))),
      hand: this.handSlots[0]
        .map((c, i) => ({ slot: i, card: c, bulls: c > 0 ? E.bullheads(c) : 0 }))
        .filter((h) => h.card > 0),
      deal_taken: [...this.dealTaken],
      total: [...this.total],
      target: this.target,
      row_points: [0, 1, 2, 3].map((r) => this.rowPoints(r)),
      waiting_row: waiting,
      pending_card: waiting ? this.waitingCard : null,
      deal_over: this.dealOver,
      match_over: this.matchOver,
    };
  }
}

if (typeof module !== "undefined") module.exports = { LocalMatch };
if (typeof globalThis !== "undefined") globalThis.LocalMatch = LocalMatch;
