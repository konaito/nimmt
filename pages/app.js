/* 6 Nimmt! フロント。サーバーのイベント列を順に再生する */

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let PLAYERS = 4;
let state = null;
let busy = false;

const names = (p) => (p === 0 ? "あなた" : `AI ${p}`);

let match = null;
const netCache = {};

async function loadNet(players) {
  const name = players === 4 ? "exp4" : "exp2";
  if (!netCache[name]) {
    const res = await fetch(`./weights_${name}.json`);
    if (!res.ok) throw new Error(`モデル読み込み失敗: ${res.status}`);
    netCache[name] = new NimmtNetJS(await res.json());
  }
  return netCache[name];
}

async function api(path, body) {
  // 旧サーバーAPIと同じ形を返すローカル実装
  if (path === "/api/game") {
    match = new LocalMatch(body.players, await loadNet(body.players));
    return match.state();
  }
  if (!match) throw new Error("マッチが始まっていない");
  if (path === "/api/state") return match.state();
  if (path === "/api/play") return { events: match.playCard(body.slot), state: match.state() };
  if (path === "/api/row") return { events: match.chooseRow(body.row), state: match.state() };
  if (path === "/api/next_deal") { match.nextDeal(); return match.state(); }
  throw new Error(`unknown path: ${path}`);
}

/* ---------- カードDOM ---------- */

function tierOf(bulls) {
  return bulls >= 7 ? 7 : bulls >= 5 ? 5 : bulls >= 3 ? 3 : bulls >= 2 ? 2 : 1;
}

function cardEl(card, bulls) {
  const el = document.createElement("div");
  el.className = "card";
  el.dataset.card = card;
  el.dataset.tier = tierOf(bulls);
  const dots = Array.from({ length: Math.min(bulls, 7) }, () => "<i></i>").join("");
  el.innerHTML = `<div class="bulls">${dots}</div><div class="num">${card}</div><div class="bhead">${bulls}</div>`;
  return el;
}

const BULLS = (c) =>
  c === 55 ? 7 : c % 11 === 0 ? 5 : c % 10 === 0 ? 3 : c % 5 === 0 ? 2 : 1;

/* ---------- 描画 ---------- */

function renderTop() {
  $("#deal-label").textContent =
    `DEAL ${state.deal_no} · TURN ${Math.min(state.turn + 1, 10)}/10`;
  const sc = $("#scores");
  sc.innerHTML = "";
  for (let p = 0; p < state.players; p++) {
    const total = state.total[p] + state.deal_taken[p] * (state.deal_over ? 0 : 1);
    // deal_over 時は total に合算済みなので二重に足さない
    const shown = state.deal_over ? state.total[p] : state.total[p] + state.deal_taken[p];
    const el = document.createElement("div");
    el.className = "score" + (p === 0 ? " human" : "") + (shown >= 50 ? " danger" : "");
    el.innerHTML = `
      <div class="score-name"><b>${names(p)}</b><span>今回 +${state.deal_taken[p]}</span></div>
      <div class="score-num">${shown}<small>/ ${state.target}</small></div>
      <div class="gauge"><i style="width:${Math.min(100, (shown / state.target) * 100)}%"></i></div>`;
    sc.appendChild(el);
  }
}

function renderRows() {
  const wrap = $("#rows");
  wrap.innerHTML = "";
  state.rows.forEach((row, r) => {
    const div = document.createElement("div");
    div.className = "row";
    div.dataset.row = r;
    const meta = document.createElement("div");
    meta.className = "row-meta";
    meta.innerHTML = `ROW ${r + 1}<b>${state.row_points[r]}🐮</b>`;
    div.appendChild(meta);
    const slots = document.createElement("div");
    slots.className = "row-slots";
    for (let i = 0; i < 5; i++) {
      if (i < row.length) {
        slots.appendChild(cardEl(row[i].card, row[i].bulls));
      } else {
        const s = document.createElement("div");
        s.className = "slot" + (i === row.length && row.length === 5 ? "" : "");
        slots.appendChild(s);
      }
    }
    div.appendChild(slots);
    wrap.appendChild(div);
  });
}

function renderHand() {
  const hand = $("#hand");
  hand.innerHTML = "";
  const n = state.hand.length;
  state.hand.forEach((h, i) => {
    const el = cardEl(h.card, h.bulls);
    const mid = (n - 1) / 2;
    el.style.setProperty("--rot", `${(i - mid) * 2.2}deg`);
    el.style.setProperty("--fan", `${Math.abs(i - mid) * 3.5}px`);
    el.addEventListener("click", () => playCard(h.slot, el));
    hand.appendChild(el);
  });
}

function renderAll() {
  renderTop();
  renderRows();
  renderHand();
}

/* ---------- アニメーション ---------- */

function flyClone(fromEl, toRect, ms = 500) {
  const from = fromEl.getBoundingClientRect();
  const clone = fromEl.cloneNode(true);
  clone.classList.add("fly");
  clone.style.left = `${from.left}px`;
  clone.style.top = `${from.top}px`;
  clone.style.width = `${from.width}px`;
  clone.style.height = `${from.height}px`;
  document.body.appendChild(clone);
  requestAnimationFrame(() => {
    const dx = toRect.left - from.left + (toRect.width - from.width) / 2;
    const dy = toRect.top - from.top + (toRect.height - from.height) / 2;
    clone.style.transform = `translate(${dx}px, ${dy}px)`;
  });
  return sleep(ms).then(() => clone.remove());
}

function slotRect(row, idx) {
  const rowEl = document.querySelector(`.row[data-row="${row}"] .row-slots`);
  const kids = rowEl.children;
  const el = kids[Math.min(idx, kids.length - 1)];
  return el.getBoundingClientRect();
}

function floatPts(rect, pts) {
  const el = document.createElement("div");
  el.className = "pts-float";
  el.textContent = `+${pts}🐮`;
  el.style.left = `${rect.left + rect.width / 2 - 24}px`;
  el.style.top = `${rect.top}px`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 1000);
}

async function playEvents(events, stateAfter) {
  const strip = $("#reveal-strip");
  for (const e of events) {
    if (e.type === "reveal") {
      strip.innerHTML = "";
      e.cards.forEach((c, p) => {
        const item = document.createElement("div");
        item.className = "reveal-item";
        item.style.animationDelay = `${p * 90}ms`;
        item.dataset.player = p;
        const who = document.createElement("div");
        who.className = "who" + (p === 0 ? " me" : "");
        who.textContent = names(p);
        item.appendChild(who);
        item.appendChild(cardEl(c, BULLS(c)));
        strip.appendChild(item);
      });
      await sleep(e.cards.length * 90 + 500);
      continue;
    }
    const src = strip.querySelector(`.reveal-item[data-player="${e.player}"] .card`);
    if (e.type === "place") {
      const rect = slotRect(e.row, e.row_before.length);
      if (src) { await flyClone(src, rect); src.classList.add("resolved"); }
      // 行にカードを差し込む（部分再描画）
      state.rows[e.row].push({ card: e.card, bulls: BULLS(e.card) });
      renderRows();
      await sleep(120);
    } else if (e.type === "take") {
      const rowEl = document.querySelector(`.row[data-row="${e.row}"]`);
      const rect = rowEl.getBoundingClientRect();
      rowEl.querySelectorAll(".card").forEach((c) => c.classList.add("taken-flash"));
      await sleep(430);
      floatPts(rect, e.points);
      if (src) { await flyClone(src, slotRect(e.row, 0)); src.classList.add("resolved"); }
      state.rows[e.row] = [{ card: e.card, bulls: BULLS(e.card) }];
      renderRows();
      await sleep(200);
    }
  }
  state = stateAfter;
  renderAll();
  if (!state.deal_over) strip.innerHTML = "";
}

/* ---------- 操作 ---------- */

async function recover(err) {
  // alert はダイアログでページ全体を止めるので使わない。
  // サーバー状態を取り直して再描画し、行選択待ちならモーダルを開き直す。
  console.error("nimmt:", err);
  try {
    state = await api("/api/state");
    renderAll();
    await afterResolve();
    $("#hint").classList.add("wait");
    setTimeout(() => $("#hint").classList.remove("wait"), 1500);
  } catch (e2) {
    $("#hint").textContent = "サーバーに接続できない。リロードして";
    console.error("nimmt recover:", e2);
  }
}

async function playCard(slot, el) {
  if (busy || state.waiting_row || state.deal_over || state.match_over) return;
  busy = true;
  $("#hand").classList.add("locked");
  $("#hint").textContent = "解決中…";
  try {
    const res = await api("/api/play", { slot, match_id: state.match_id });
    await playEvents(res.events, res.state);
    await afterResolve();
  } catch (err) {
    await recover(err);
  } finally {
    busy = false;
    $("#hand").classList.remove("locked");
  }
}

async function afterResolve() {
  if (state.waiting_row) {
    openRowPick();
    return;
  }
  if (state.deal_over) {
    showResult();
    return;
  }
  $("#hint").textContent = "出すカードを選んで";
}

function openRowPick() {
  $("#rowpick-msg").textContent =
    `あなたの ${state.pending_card} はどの行末よりも小さい。引き取る行を選ぶ（その牛頭があなたの失点になる）`;
  const wrap = $("#rowpick-buttons");
  wrap.innerHTML = "";
  state.rows.forEach((row, r) => {
    const b = document.createElement("button");
    b.className = "btn-row";
    b.innerHTML = `<span>ROW ${r + 1} — ${row.map((c) => c.card).join(" · ")}</span>
                   <span class="pts">+${state.row_points[r]}🐮</span>`;
    b.addEventListener("click", () => chooseRow(r));
    wrap.appendChild(b);
  });
  $("#rowpick").hidden = false;
}

async function chooseRow(r) {
  $("#rowpick").hidden = true;
  busy = true;
  try {
    const res = await api("/api/row", { row: r, match_id: state.match_id });
    await playEvents(res.events, res.state);
    await afterResolve();
  } catch (err) {
    await recover(err);
  } finally {
    busy = false;
  }
}

function showResult() {
  const over = state.match_over;
  $("#result-title").textContent = over ? "マッチ終了" : `ディール ${state.deal_no} 終了`;
  const body = $("#result-body");
  body.innerHTML = "";
  const order = [...Array(state.players).keys()].sort((a, b) => state.total[a] - state.total[b]);
  const best = state.total[order[0]];
  order.forEach((p) => {
    const line = document.createElement("div");
    line.className = "res-line" + (over && state.total[p] === best ? " winner" : "");
    line.innerHTML = `<span>${over && state.total[p] === best ? "🏆 " : ""}${names(p)}
        <small style="color:var(--ink-soft)">（今回 +${state.deal_taken[p]}）</small></span>
      <span class="n">${state.total[p]}🐮</span>`;
    body.appendChild(line);
  });
  const btn = $("#result-next");
  btn.textContent = over ? "もう一度遊ぶ" : "次のディールへ";
  btn.onclick = async () => {
    $("#result").hidden = true;
    if (over) {
      $("#game").hidden = true;
      $("#start").hidden = false;
      return;
    }
    state = await api("/api/next_deal", { match_id: state.match_id });
    renderAll();
    $("#reveal-strip").innerHTML = "";
    $("#hint").textContent = "出すカードを選んで";
  };
  $("#result").hidden = false;
}

/* ---------- 起動 ---------- */

document.querySelectorAll(".btn-mode").forEach((btn) => {
  btn.addEventListener("click", async () => {
    PLAYERS = Number(btn.dataset.players);
    state = await api("/api/game", { players: PLAYERS });
    $("#start").hidden = true;
    $("#game").hidden = false;
    $("#reveal-strip").innerHTML = "";
    renderAll();
  });
});
