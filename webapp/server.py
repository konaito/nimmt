"""ニムト対戦Web UIのサーバー。

Usage: uv run --group web uvicorn webapp.server:app --port 8765

人間は席0。4人戦は runs/exp4/latest.pt、2人戦は runs/exp2/latest.pt が相手。
66点マッチ: 誰かの累積が66牛頭に達したディールで終了、最少失点が勝ち。

解決アニメーション用のイベント列は自前のシミュレーションで生成し、
毎ターン終了時に env の盤面・失点と照合する。ズレたら黙って進めず 500 で落とす。
"""

import secrets
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nimmt.bots.neural import NeuralBot
from nimmt.cards import BULL, N_ROWS, ROW_CAP
from nimmt.rl.model import load
from nimmt.vec import PHASE_ROW, VecNimmt

ROOT = Path(__file__).resolve().parent.parent
TARGET = 66
HUMAN = 0

app = FastAPI(title="6 Nimmt!")

_nets: dict[int, object] = {}


def _net(players: int):
    if players not in _nets:
        ckpt = ROOT / ("runs/exp4/latest.pt" if players == 4 else "runs/exp2/latest.pt")
        _nets[players] = load(ckpt)
    return _nets[players]


class Match:
    def __init__(self, players: int):
        self.match_id = secrets.token_hex(8)
        self.players = players
        self.bot = NeuralBot(_net(players), temperature=0.0)
        self.total = np.zeros(players, dtype=np.int64)
        self.deal_no = 0
        self.rng = np.random.default_rng(secrets.randbits(63))
        self.match_over = False
        self._new_deal()

    def _new_deal(self):
        self.deal_no += 1
        self.env = VecNimmt(1, self.players, seed=int(self.rng.integers(1 << 30)))
        self.events: list[dict] = []
        self.played: np.ndarray | None = None
        # 自前シミュレーションの盤面（env と毎ターン照合する）
        self.sim_rows = [list(self.env.rows[0, r, : self.env.row_len[0, r]]) for r in range(N_ROWS)]
        self.deal_over = False

    # ---------- 解決 ----------

    def _row_points(self, r: int) -> int:
        return int(sum(int(BULL[c]) for c in self.sim_rows[r]))

    def _sim_place(self, player: int, card: int, row: int, forced: bool) -> dict:
        """自前盤面に1枚置いてイベントを返す。"""
        before = [int(c) for c in self.sim_rows[row]]
        if forced or len(self.sim_rows[row]) >= ROW_CAP:
            pts = self._row_points(row)
            self.sim_rows[row] = [card]
            self.turn_pts[player] += pts
            return {"type": "take", "player": player, "card": card, "row": row,
                    "points": pts, "row_before": before, "forced": forced}
        self.sim_rows[row].append(card)
        return {"type": "place", "player": player, "card": card, "row": row,
                "row_before": before}

    def _verify(self):
        env_rows = [[int(c) for c in self.env.rows[0, r, : self.env.row_len[0, r]]]
                    for r in range(N_ROWS)]
        if env_rows != [[int(c) for c in row] for row in self.sim_rows]:
            raise RuntimeError(f"盤面シミュレーションが env と不一致: sim={self.sim_rows} env={env_rows}")

    def _resolve(self) -> bool:
        """カード公開後の解決を進める。人間の行選択待ちになったら True を返して中断。"""
        assert self.played is not None
        order = np.argsort(self.played, kind="stable")
        while self._k < self.players:
            p = int(order[self._k])
            card = int(self.played[p])
            ends = [int(row[-1]) for row in self.sim_rows]
            valid = [r for r in range(N_ROWS) if ends[r] < card]
            if valid:
                tgt = max(valid, key=lambda r: ends[r])
                self.events.append(self._sim_place(p, card, tgt, forced=False))
                self._k += 1
                continue
            # どの行末よりも小さい → 行選択。env 側も同じ地点で PHASE_ROW で止まっているはず
            if self.env.phase != PHASE_ROW or int(self.env.pending_players[0]) != p:
                raise RuntimeError("解決順序が env と不一致")
            if p == HUMAN:
                self.pending_card = card
                return True
            choice = int(self.bot.pick_rows(self.env, np.array([0], dtype=np.intp))[0])
            self.env.step_rows(np.array([choice], dtype=np.intp))
            self.events.append(self._sim_place(p, card, choice, forced=True))
            self._k += 1
        self._finish_turn()
        return False

    def _finish_turn(self):
        self._verify()
        env_taken = self.env.taken[0].astype(int)
        self.deal_taken_check = self.deal_taken_check + self.turn_pts
        if not np.array_equal(env_taken, self.deal_taken_check):
            raise RuntimeError(
                f"失点がenvと不一致: sim={self.deal_taken_check.tolist()} env={env_taken.tolist()}")
        self.played = None
        if self.env.done():
            self.deal_over = True
            self.total += env_taken
            if int(self.total.max()) >= TARGET:
                self.match_over = True

    # ---------- 操作 ----------

    def play_card(self, slot: int) -> list[dict]:
        if self.deal_over or self.match_over:
            raise HTTPException(409, "ディールは終了している")
        if self.played is not None:
            raise HTTPException(409, "行選択待ち。/api/row を呼ぶこと")
        hand = self.env.hands[0, HUMAN]
        if not (0 <= slot < hand.shape[0]) or hand[slot] <= 0:
            raise HTTPException(400, "そのスロットは出せない")
        if not hasattr(self, "deal_taken_check"):
            self.deal_taken_check = np.zeros(self.players, dtype=np.int64)
        self.turn_pts = np.zeros(self.players, dtype=np.int64)
        slots = np.zeros((1, self.players), dtype=np.intp)
        slots[0, HUMAN] = slot
        g0 = np.array([0], dtype=np.intp)
        for s in range(1, self.players):
            slots[0, s] = self.bot.pick_cards(self.env, g0, np.array([s], dtype=np.intp))[0]
        self.played = self.env.hands[0, np.arange(self.players), slots[0]].copy().astype(int)
        self.events = [{"type": "reveal",
                        "cards": [int(self.played[p]) for p in range(self.players)]}]
        self._k = 0
        self.env.step_cards(slots)
        self._resolve()
        return self.events

    def choose_row(self, row: int) -> list[dict]:
        if self.played is None or not hasattr(self, "pending_card"):
            raise HTTPException(409, "行選択待ちではない")
        if not (0 <= row < N_ROWS):
            raise HTTPException(400, "行は0-3")
        if self.env.phase != PHASE_ROW or int(self.env.pending_players[0]) != HUMAN:
            raise HTTPException(409, "行選択待ちではない")
        self.env.step_rows(np.array([row], dtype=np.intp))
        self.events = [self._sim_place(HUMAN, self.pending_card, row, forced=True)]
        del self.pending_card
        self._k += 1
        self._resolve()
        return self.events

    def next_deal(self):
        if not self.deal_over or self.match_over:
            raise HTTPException(409, "ディールは終了していない")
        self._new_deal()
        if hasattr(self, "deal_taken_check"):
            del self.deal_taken_check

    # ---------- 状態 ----------

    def state(self) -> dict:
        env = self.env
        hand = [{"slot": i, "card": int(c), "bulls": int(BULL[c])}
                for i, c in enumerate(env.hands[0, HUMAN]) if c > 0]
        rows = [[{"card": int(c), "bulls": int(BULL[c])}
                 for c in env.rows[0, r, : env.row_len[0, r]]] for r in range(N_ROWS)]
        waiting_row = self.played is not None and hasattr(self, "pending_card")
        return {
            "match_id": self.match_id,
            "players": self.players, "deal_no": self.deal_no,
            "turn": int(env.turn), "rows": rows, "hand": hand,
            "deal_taken": env.taken[0].astype(int).tolist(),
            "total": self.total.tolist(), "target": TARGET,
            "row_points": [self._row_points(r) for r in range(N_ROWS)],
            "waiting_row": waiting_row,
            "pending_card": int(self.pending_card) if waiting_row else None,
            "deal_over": self.deal_over, "match_over": self.match_over,
        }


_match: Match | None = None


class NewGame(BaseModel):
    players: int


class Play(BaseModel):
    slot: int
    match_id: str


class Row(BaseModel):
    row: int
    match_id: str


class NextDeal(BaseModel):
    match_id: str


def _m(match_id: str | None = None) -> Match:
    if _match is None:
        raise HTTPException(404, "マッチが始まっていない")
    if match_id is not None and match_id != _match.match_id:
        # 別タブ・古いタブからの操作を進行中のマッチに混ぜない
        raise HTTPException(409, "このタブのマッチは終了している。リロードして")
    return _match


@app.post("/api/game")
def new_game(req: NewGame):
    if req.players not in (2, 4):
        raise HTTPException(400, "players は 2 か 4")
    global _match
    _match = Match(req.players)
    return _match.state()


@app.get("/api/state")
def state():
    return _m().state()


@app.post("/api/play")
def play(req: Play):
    m = _m(req.match_id)
    events = m.play_card(req.slot)
    return {"events": events, "state": m.state()}


@app.post("/api/row")
def row(req: Row):
    m = _m(req.match_id)
    events = m.choose_row(req.row)
    return {"events": events, "state": m.state()}


@app.post("/api/next_deal")
def next_deal(req: NextDeal):
    m = _m(req.match_id)
    m.next_deal()
    return m.state()


@app.get("/")
def index():
    return FileResponse(ROOT / "webapp" / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "webapp" / "static"), name="static")
