"""webapp/server.py のAPIフローのテスト。

サーバー内部に「自前シミュレーションと env の照合ガード」があり、ズレると 500 になる。
ここで人間役を機械的に動かして複数ディール完走させることが、そのガードの回帰テストを兼ねる。
"""

import pytest
from fastapi.testclient import TestClient

from webapp.server import TARGET, app

client = TestClient(app, raise_server_exceptions=True)


def _play_until_match_over(players: int, max_actions: int = 3000) -> dict:
    state = client.post("/api/game", json={"players": players}).json()
    assert state["players"] == players
    for _ in range(max_actions):
        if state["match_over"]:
            return state
        mid = state["match_id"]
        if state["deal_over"]:
            state = client.post("/api/next_deal", json={"match_id": mid}).json()
            continue
        if state["waiting_row"]:
            r = client.post("/api/row", json={"row": 0, "match_id": mid})
            assert r.status_code == 200, r.text
            state = r.json()["state"]
            continue
        slot = state["hand"][0]["slot"]
        r = client.post("/api/play", json={"slot": slot, "match_id": mid})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["events"][0]["type"] == "reveal"
        state = body["state"]
        for row in state["rows"]:
            assert 1 <= len(row) <= 5
    pytest.fail("マッチが終わらない")


@pytest.mark.parametrize("players", [2, 4])
def test_full_match(players):
    state = _play_until_match_over(players)
    assert max(state["total"]) >= TARGET
    assert state["match_over"]


def test_rejects_bad_inputs():
    mid = client.post("/api/game", json={"players": 2}).json()["match_id"]
    assert client.post("/api/game", json={"players": 3}).status_code == 400
    assert client.post("/api/play", json={"slot": 99, "match_id": mid}).status_code == 400
    assert client.post("/api/play", json={"slot": 0, "match_id": "stale"}).status_code == 409
    assert client.post("/api/next_deal", json={"match_id": mid}).status_code == 409
    state = client.get("/api/state").json()
    assert state["turn"] == 0 and not state["deal_over"]
