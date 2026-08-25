"""素直な逐次実装。ベクトル化実装の正しさを検証する基準として使う。速度は問わない。"""

from dataclasses import dataclass
from typing import Callable, Sequence

from .cards import BULL, HAND_SIZE, N_ROWS, ROW_CAP


@dataclass
class RefGame:
    rows: list[list[int]]
    hands: list[list[int]]
    taken: list[int]

    @property
    def n_players(self) -> int:
        return len(self.hands)


def new_game(deck: Sequence[int], n_players: int) -> RefGame:
    hands = [sorted(deck[p * HAND_SIZE : (p + 1) * HAND_SIZE]) for p in range(n_players)]
    rows = [[int(deck[n_players * HAND_SIZE + r])] for r in range(N_ROWS)]
    return RefGame(rows=rows, hands=hands, taken=[0] * n_players)


def target_row(rows: list[list[int]], card: int) -> int | None:
    """行末が card より小さい行のうち、行末が最大の行の index。無ければ None。"""
    best, best_end = None, -1
    for i, row in enumerate(rows):
        end = row[-1]
        if end < card and end > best_end:
            best, best_end = i, end
    return best


def row_points(row: list[int]) -> int:
    return int(sum(int(BULL[c]) for c in row))


def play_turn(
    g: RefGame,
    plays: Sequence[int],
    row_policy: Callable[[RefGame, int, int], int],
) -> list[int]:
    """全員が同時に出した plays を昇順に解決する。各プレイヤーの今ターンの失点を返す。"""
    n = g.n_players
    for p in range(n):
        g.hands[p].remove(plays[p])

    gained = [0] * n
    order = sorted(range(n), key=lambda p: plays[p])
    for p in order:
        card = int(plays[p])
        r = target_row(g.rows, card)
        if r is None:
            r = int(row_policy(g, p, card))
            assert 0 <= r < N_ROWS
            pts = row_points(g.rows[r])
            g.rows[r] = [card]
        elif len(g.rows[r]) == ROW_CAP:
            pts = row_points(g.rows[r])
            g.rows[r] = [card]
        else:
            pts = 0
            g.rows[r].append(card)
        gained[p] += pts
        g.taken[p] += pts
    return gained


def play_deal(
    deck: Sequence[int],
    n_players: int,
    card_policy: Callable[[RefGame, int], int],
    row_policy: Callable[[RefGame, int, int], int],
) -> list[int]:
    g = new_game(deck, n_players)
    for _ in range(HAND_SIZE):
        plays = [int(card_policy(g, p)) for p in range(n_players)]
        play_turn(g, plays, row_policy)
    return list(g.taken)
