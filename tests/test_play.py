import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_render_shows_rows_hand_and_scores():
    from nimmt.vec import VecNimmt
    from play import render

    env = VecNimmt(1, 4, seed=0)
    out = render(env, seat=0)
    assert "行1" in out and "手札" in out and "失点" in out
    for c in env.hands[0, 0]:
        assert str(int(c)) in out


def test_eof_input_terminates_instead_of_looping(monkeypatch):
    """入力が尽きたら無限ループせずに中断すること。

    EOFError を握りつぶして continue すると、input() が即座に再送出してタイトループになる
    （実測: 10秒で約185万行）。対話ターミナルで Ctrl-D を押すだけで起きる経路。
    """
    from play import main

    def eof(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["--players", "4", "--seed", "3"]) == 1


def test_full_game_with_scripted_input(monkeypatch, capsys):
    """常に手札の先頭を選ぶ入力で1ディール完走する。"""
    from play import main

    answers = iter(["1"] * 60)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    rc = main(["--players", "4", "--seed", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "最終結果" in out
