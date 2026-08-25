import numpy as np

from nimmt.features import C_DIM, G_DIM, R_DIM, card_obs, global_features, row_obs
from nimmt.vec import PHASE_ROW, VecNimmt


def make_env():
    env = VecNimmt(1, 2, seed=0)
    env.rows[:] = 0
    env.rows[0, 0, :2] = [10, 11]      # 牛頭 3+5 = 8、行末 11
    env.rows[0, 1, 0] = 20             # 牛頭 3
    env.rows[0, 2, 0] = 30             # 牛頭 3
    env.rows[0, 3, :] = [40, 41, 42, 43, 44]   # 5枚、牛頭 3+1+1+1+5 = 11、行末 44（44は11の倍数なので牛頭5）
    env.row_len[0] = [2, 1, 1, 5]
    env.seen[0] = False
    env.seen[0, [10, 11, 20, 30, 40, 41, 42, 43, 44]] = True
    env.hands[0] = 0
    env.hands[0, 0, :4] = [5, 12, 45, 104]
    env.hands[0, 1, :2] = [50, 60]
    env.taken[0] = [7, 3]
    env.turn = 2
    return env


def test_shapes_and_dtypes():
    env = VecNimmt(8, 4, seed=1)
    g, c, m = card_obs(env)
    assert g.shape == (8, 4, G_DIM) and g.dtype == np.float32
    assert c.shape == (8, 4, 10, C_DIM) and c.dtype == np.float32
    assert m.shape == (8, 4, 10) and m.dtype == np.bool_
    assert m.all()  # 開始時は全スロット有効
    assert np.isfinite(g).all() and np.isfinite(c).all()
    assert G_DIM == 266 and C_DIM == 12 and R_DIM == 9


def test_global_layout():
    env = make_env()
    g = global_features(env, np.array([0]), np.array([0]))[0]
    assert g.shape == (G_DIM,)
    # 行0: 行末11, 枚数2, 牛頭8
    assert g[0] == np.float32(11 / 104)
    assert g[1] == np.float32(2 / 5)
    assert g[2] == np.float32(8 / 14)
    assert g[3:8].tolist() == [0, 1, 0, 0, 0]        # 枚数 2 の one-hot
    # 行3: 5枚
    assert g[24 + 3 : 24 + 8].tolist() == [0, 0, 0, 0, 1]
    # 手札 multi-hot
    for card in (5, 12, 45, 104):
        assert g[32 + card - 1] == 1.0
    assert g[32 + 50 - 1] == 0.0                      # 相手の手札は見えない
    assert g[32:136].sum() == 4
    # 既知カード
    assert g[136 + 10 - 1] == 1.0 and g[136:240].sum() == 9
    # 失点: 自分7, 相手3（P=2なので残り2枠は0）
    # float32 に量子化した値同士で比較する（tolist() は float64 に昇格するため
    # 生の 7/20 と直接比較すると丸め誤差で不一致になる）
    assert g[240:244].tolist() == [np.float32(7 / 20), np.float32(3 / 20), 0.0, 0.0]
    # ターン one-hot
    assert g[244 + 2] == 1.0 and g[244:254].sum() == 1
    # 人数
    assert g[254] == np.float32(2 / 10)
    assert g[255] == 1.0 and g[255:264].sum() == 1
    # カード相では末尾2次元は0
    assert g[264] == 0.0 and g[265] == 0.0


def test_card_features_placement_logic():
    env = make_env()
    _, c, m = card_obs(env)
    f = c[0, 0]
    assert m[0, 0].tolist() == [True] * 4 + [False] * 6
    assert (f[4:] == 0).all()                          # 無効スロットは全0

    # カード5: 全行末(11,20,30,44)より小さい → 強制取得。最小牛頭の行は行1か行2の3
    assert f[0][6] == 1.0
    assert f[0][4] == 0.0
    assert f[0][8] == np.float32(3 / 14)
    # 行き先が無いので、行き先依存の特徴(2,3,4,7)は0。行0の状態が漏れていないこと
    assert f[0][2] == 0.0 and f[0][3] == 0.0 and f[0][7] == 0.0

    # カード12: 行0(行末11)へ。枚数2なので6枚目ではない
    assert f[1][6] == 0.0
    assert f[1][2] == np.float32(2 / 5)
    assert f[1][3] == np.float32(8 / 14)
    assert f[1][4] == np.float32(1 / 104)
    assert f[1][5] == 0.0
    assert f[1][8] == 0.0
    assert f[1][7] == np.float32(3 / 5)

    # カード45: 行3(行末44, 5枚)へ → 6枚目、牛頭11を引き取る（44は11の倍数なので牛頭5、行合計3+1+1+1+5=11）
    assert f[2][5] == 1.0
    assert f[2][8] == np.float32(11 / 14)

    # カード104: 行3へ。行末44 との差
    assert f[3][4] == np.float32((104 - 44) / 104)


def test_unseen_counts():
    env = make_env()
    _, c, _ = card_obs(env)
    f = c[0, 0]
    # 未見 = 104 - 既知9 - 自分の手札4 = 91
    # カード12 の行き先は行0（行末11）。(11,12) に入る未見カードは無い
    assert f[1][10] == 0.0
    # カード12より小さい未見カード: 1..11 のうち既知(10,11)と手札(5)を除く = 8枚
    assert abs(f[1][9] - 8 / 91) < 1e-6
    # カード104 の行き先は行3（行末44）。(44,104) の未見カード数を数える
    lower = [x for x in range(45, 104) if x not in (5, 12, 45, 104) and not env.seen[0, x]]
    assert abs(f[3][10] - len(lower) / 91) < 1e-6


def test_row_obs():
    env = make_env()
    env.hands[0] = 0
    env.hands[0, 0, 0] = 5
    env.hands[0, 1, 0] = 6
    env.step_cards(np.zeros((1, 2), dtype=np.intp))
    assert env.phase == PHASE_ROW
    g, r = row_obs(env)
    assert g.shape == (1, G_DIM) and r.shape == (1, 4, R_DIM)
    # 行相では末尾2次元に出したカードが入る（カード5、牛頭2）
    assert g[0, 264] == np.float32(5 / 104)
    assert g[0, 265] == np.float32(2 / 7)
    # 行1（牛頭3）と行2（牛頭3）が最小 → is_min フラグが立つ
    assert r[0, :, 8].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert r[0, 0, 0] == np.float32(2 / 5)
    assert r[0, 0, 1] == np.float32(8 / 14)
    assert r[0, 0, 2] == np.float32(11 / 104)
