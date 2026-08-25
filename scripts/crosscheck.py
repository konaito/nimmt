"""参照実装とベクトル化実装を大量のディールで突き合わせる。
Usage: uv run python scripts/crosscheck.py [n_deals] [n_players]"""

import sys

import numpy as np

from nimmt.cards import BULL
from nimmt.reference import play_deal
from nimmt.vec import PHASE_ROW, VecNimmt, new_decks


def main() -> int:
    n_deals = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    n_players = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    rng = np.random.default_rng(20260824)
    decks = new_decks(n_deals, rng)

    env = VecNimmt(n_deals, n_players, seed=0)
    env.reset(decks)
    while not env.done():
        h = np.where(env.hands > 0, env.hands, np.int16(32767))
        env.step_cards(h.argmin(axis=2))
        while env.phase == PHASE_ROW:
            env.step_rows(env.row_points_all(env.pending_games).argmin(axis=1))
    vec_taken = env.taken.copy()

    def ref_card(g, p):
        return min(g.hands[p])

    def ref_row(g, p, card):
        return min((sum(int(BULL[c]) for c in row), i) for i, row in enumerate(g.rows))[1]

    bad = 0
    for i in range(n_deals):
        ref = play_deal(decks[i].tolist(), n_players, ref_card, ref_row)
        if ref != vec_taken[i].tolist():
            bad += 1
            if bad <= 3:
                print(f"MISMATCH deal={i} ref={ref} vec={vec_taken[i].tolist()}")
    print(f"deals={n_deals} players={n_players} mismatches={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
