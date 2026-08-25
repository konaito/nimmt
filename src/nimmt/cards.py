"""カードと牛頭（ペナルティ点）の定義。他のどのモジュールにも依存しない。"""

import numpy as np

N_CARDS = 104
N_ROWS = 4
ROW_CAP = 5
HAND_SIZE = 10
TOTAL_BULLHEADS = 171


def bullheads(card: int) -> int:
    """カード番号 1..104 の牛頭数を返す。判定順序が重要（55 は 5 と 11 の両方の倍数）。"""
    if card == 55:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 5 == 0:
        return 2
    return 1


BULL = np.zeros(N_CARDS + 1, dtype=np.int16)
for _c in range(1, N_CARDS + 1):
    BULL[_c] = bullheads(_c)
BULL.flags.writeable = False
