"""対人対戦CLI。
Usage: uv run python scripts/play.py --checkpoint runs/exp1/latest.pt --players 4"""

import argparse

import numpy as np

from nimmt.cards import BULL, N_ROWS
from nimmt.bots.simple import HeuristicBot
from nimmt.vec import PHASE_ROW, VecNimmt


def render(env: VecNimmt, seat: int) -> str:
    lines = []
    for r in range(N_ROWS):
        cards = env.rows[0, r, : env.row_len[0, r]]
        cells = " ".join(f"{int(c):3d}({int(BULL[c])})" for c in cards)
        slots = "・" * (5 - int(env.row_len[0, r]))
        lines.append(f"行{r + 1}: {cells} {slots}")
    hand = [int(c) for c in env.hands[0, seat] if c > 0]
    lines.append("")
    lines.append("手札: " + "  ".join(f"[{i + 1}] {c}({int(BULL[c])})" for i, c in enumerate(hand)))
    scores = "  ".join(
        f"{'あなた' if p == seat else f'AI{p}'}={int(env.taken[0, p])}" for p in range(env.P)
    )
    lines.append(f"失点: {scores}   ターン {env.turn + 1}/10")
    return "\n".join(lines)


def _hand_slots(env, seat) -> list[int]:
    return [i for i in range(10) if env.hands[0, seat, i] > 0]


def _ask(prompt: str, n: int) -> int:
    """1〜n の入力を 0-origin にして返す。入力が尽きたら -1 を返す。

    EOFError を ValueError と一緒に握りつぶして continue すると、標準入力が尽きた状態で
    `input()` が即座に EOFError を再送出し続け、ブロックしないタイトループになる
    （実測: 10秒で約185万行を出力して終わらない）。対話ターミナルで Ctrl-D を
    押しただけでこれが起きるので、EOF は必ず区別して抜ける。
    """
    while True:
        try:
            raw = input(prompt)
        except EOFError:
            print("\n入力が終了したので中断する。")
            return -1
        try:
            v = int(raw.strip())
        except ValueError:
            print("数字を入れて")
            continue
        if 1 <= v <= n:
            return v - 1
        print(f"1〜{n} で入れて")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.checkpoint:
        from nimmt.bots.neural import NeuralBot
        from nimmt.rl.model import load

        ai = NeuralBot(load(args.checkpoint), temperature=0.0)
    else:
        ai = HeuristicBot()

    env = VecNimmt(1, args.players, seed=args.seed)
    seat = args.seat
    g0 = np.array([0], dtype=np.intp)

    while not env.done():
        print("\n" + "=" * 60)
        print(render(env, seat))
        slots_avail = _hand_slots(env, seat)
        pick = _ask(f"どれを出す？ [1-{len(slots_avail)}]: ", len(slots_avail))
        if pick < 0:
            return 1

        slots = np.zeros((1, env.P), dtype=np.intp)
        slots[0, seat] = slots_avail[pick]
        for s in range(env.P):
            if s == seat:
                continue
            slots[0, s] = ai.pick_cards(env, g0, np.array([s], dtype=np.intp))[0]
        played = env.hands[0, np.arange(env.P), slots[0]].copy()
        env.step_cards(slots)
        print("公開: " + "  ".join(
            f"{'あなた' if p == seat else f'AI{p}'}={int(played[p])}" for p in range(env.P)
        ))

        while env.phase == PHASE_ROW:
            choices = np.zeros(env.pending_games.size, dtype=np.intp)
            mine = np.flatnonzero(env.pending_players == seat)
            other = np.flatnonzero(env.pending_players != seat)
            if other.size:
                choices[other] = ai.pick_rows(env, other)
            if mine.size:
                print("\n" + render(env, seat))
                print(f"あなたのカード {int(env.pending_cards[mine[0]])} はどの行末よりも小さい。")
                bulls = env.row_points_all(env.pending_games[mine])[0]
                print("引き取る牛頭: " + "  ".join(
                    f"[{r + 1}] {int(bulls[r])}" for r in range(N_ROWS)))
                choice = _ask("どの行を引き取る？ [1-4]: ", N_ROWS)
                if choice < 0:
                    return 1
                choices[mine] = choice
            env.step_rows(choices)

        pts = env.last_turn_points()[0]
        if pts.any():
            print("引き取り: " + "  ".join(
                f"{'あなた' if p == seat else f'AI{p}'}+{int(pts[p])}"
                for p in range(env.P) if pts[p]))

    print("\n" + "=" * 60)
    final = env.taken[0]
    print("最終結果: " + "  ".join(
        f"{'あなた' if p == seat else f'AI{p}'}={int(final[p])}" for p in range(env.P)))
    best = int(final.min())
    winners = [("あなた" if p == seat else f"AI{p}") for p in range(env.P) if final[p] == best]
    print(f"勝者: {', '.join(winners)}（{best}牛頭）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
