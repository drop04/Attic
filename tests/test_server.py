import pytest
from starlette.testclient import TestClient

from attic.db import Store
from attic.server import build_app
from tests.samples import INJECTED, RECEIPT

TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ATTIC_BEDROCK_MODEL", raising=False)
    app = build_app(Store(tmp_path / "a.db"), TOKEN, tmp_path)
    with TestClient(app) as c:
        yield c


def test_auth_required_except_public_paths(client):
    assert client.get("/healthz").text == "ok"
    assert "Attic" in client.get("/upload").text
    assert client.post("/api/ingest", data={"text": RECEIPT}).status_code == 401
    assert client.post("/api/ingest", data={"text": RECEIPT}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/mcp", json={}).status_code == 401


def test_ingest_then_confirm_flow(client):
    r = client.post("/api/ingest", data={"text": RECEIPT}, headers=AUTH).json()
    assert r["fields"]["expires_on"] == "2026-03-14" and r["used_cloud"] is False
    fields = {**r["fields"], "vendor": "Sharma Electronics (edited)"}
    saved = client.post("/api/confirm", json={"draft_id": r["draft_id"], "fields": fields}, headers=AUTH).json()
    assert saved["fields"]["vendor"] == "Sharma Electronics (edited)"
    # a draft can only be confirmed once
    assert client.post("/api/confirm", json={"draft_id": r["draft_id"]}, headers=AUTH).status_code == 404


def test_ingest_rejects_bad_input(client):
    assert client.post("/api/ingest", data={"text": " "}, headers=AUTH).status_code == 422
    r = client.post("/api/ingest", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, headers=AUTH)
    assert r.status_code == 415
    assert client.post("/api/confirm", content=b"not json", headers=AUTH).status_code == 400
