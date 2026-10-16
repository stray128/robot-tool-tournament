"""HTTP API of the Grasp Engineer: uploads, passcode, sessions over SSE, runs, print packs (no network)."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import trimesh
from fastapi.testclient import TestClient

from rtt.agent.backends import PartNotFoundError
from rtt.agent.caps import AgentCaps, PasscodeGate, PasscodeRequiredError, passcode_digest
from rtt.agent.fakes import FakePartBackend, ScriptedAgentTransport
from rtt.api.app import create_app, sse_format
from rtt.api.service import AgentService, ApiSettings, check_solid, check_stl, clean_name

PASSCODE = "correct horse"


def stl_bytes(extents: tuple[float, float, float] = (40.0, 30.0, 60.0)) -> bytes:
    return trimesh.creation.box(extents=extents).export(file_type="stl")


@pytest.fixture
def service(tmp_path: Path) -> Iterator[AgentService]:
    settings = ApiSettings(root=tmp_path / "agent", max_upload_bytes=200_000, caps=AgentCaps(poll_wait_s=5.0))
    svc = AgentService(
        settings,
        backend=FakePartBackend(tmp_path / "agent" / "parts", delay_s=0.1),
        gate=PasscodeGate(passcode_digest(PASSCODE)),
        transport_factory=ScriptedAgentTransport,
    )
    yield svc
    svc.shutdown()


@pytest.fixture
def client(service: AgentService) -> Iterator[TestClient]:
    with TestClient(create_app(service=service)) as c:
        yield c


def upload(client: TestClient, data: bytes | None = None, **params: Any) -> Any:
    return client.post(
        "/parts",
        content=data if data is not None else stl_bytes(),
        params=params,
        headers={"content-type": "application/octet-stream"},
    )


def read_sse(client: TestClient, url: str, **kwargs: Any) -> list[tuple[str, dict[str, Any]]]:
    frames: list[tuple[str, dict[str, Any]]] = []
    event = None
    with client.stream("GET", url, **kwargs) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event is not None:
                frames.append((event, json.loads(line.removeprefix("data: "))))
                event = None
    return frames


# --------------------------------------------------------------------------- uploads


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["ok"] and body["mode"] == "live" and body["passcode_configured"]
    assert body["label"].startswith("Product mode") and body["targets"] == ["local"]


def test_upload_part(client: TestClient) -> None:
    response = upload(client, name="My <bracket>/../x", units="mm")
    assert response.status_code == 201, response.text
    part = response.json()["part"]
    assert part["part_id"].startswith("part-") and part["name"] == "My bracket..x"
    assert client.get(f"/parts/{part['part_id']}").json()["part"]["part_id"] == part["part_id"]
    assert client.get("/parts/part-000000000000").status_code == 404
    assert client.get("/parts/..%2F..%2Fetc").status_code == 404


def test_upload_limits(client: TestClient) -> None:
    assert upload(client, b"\0" * 200_001).status_code == 413
    assert upload(client, b"hello world" * 10).status_code == 422
    assert upload(client, units="furlong").status_code == 422
    ascii_stl = trimesh.creation.box(extents=(10, 20, 30)).export(file_type="stl_ascii").encode()
    assert upload(client, ascii_stl).status_code == 201


def test_check_stl_and_clean_name() -> None:
    check_stl(stl_bytes())
    with pytest.raises(ValueError):
        check_stl(b"x" * 84)
    assert clean_name("  ../../etc/passwd  ", "fallback") == "etcpasswd"
    assert clean_name("$$$", "fallback") == "fallback"


# --------------------------------------------------------------------------- sessions


def test_session_needs_the_passcode(client: TestClient) -> None:
    part_id = upload(client).json()["part"]["part_id"]
    assert client.post("/sessions", json={"part_id": part_id}).status_code == 403
    assert client.post("/sessions", json={"part_id": part_id, "passcode": "nope"}).status_code == 403
    assert (
        client.post("/sessions", json={"part_id": "part-000000000000", "passcode": PASSCODE}).status_code
        == 404
    )
    assert client.post("/sessions", json={"part_id": "bad"}).status_code == 422
    assert (
        client.post("/sessions", json={"part_id": part_id, "passcode": PASSCODE, "x": 1}).status_code == 422
    )


def test_passcode_lockout(client: TestClient) -> None:
    part_id = upload(client).json()["part"]["part_id"]
    for _ in range(5):
        assert client.post("/sessions", json={"part_id": part_id, "passcode": "guess"}).status_code == 403
    assert client.post("/sessions", json={"part_id": part_id, "passcode": PASSCODE}).status_code == 429


def test_session_streams_events_and_serves_outputs(client: TestClient, service: AgentService) -> None:
    part_id = upload(client).json()["part"]["part_id"]
    started = client.post(
        "/sessions", json={"part_id": part_id, "goal": "hold it"}, headers={"X-RTT-Passcode": PASSCODE}
    )
    assert started.status_code == 201, started.text
    session_id = started.json()["session_id"]
    frames = read_sse(client, f"/sessions/{session_id}/events")
    kinds = [k for k, _ in frames]
    assert kinds[0] == "session.started" and kinds[-1] == "end"
    assert "session.finished" in kinds and kinds.count("tool.call") == 8
    finished = next(d for k, d in frames if k == "session.finished")
    assert finished["status"] == "completed"
    seqs = [d["seq"] for k, d in frames if k != "end"]
    assert seqs == sorted(seqs) and seqs[0] == 1

    # resume after an event id: only later events come back
    later = read_sse(client, f"/sessions/{session_id}/events", headers={"Last-Event-ID": str(seqs[-3])})
    assert [d["seq"] for k, d in later if k != "end"] == seqs[-2:]

    doc = client.get(f"/sessions/{session_id}").json()
    assert doc["status"] == "completed" and doc["goal"] == "hold it" and doc["offline"] is False
    fetched = next(d for k, d in frames if k == "tool.result" and d["name"] == "fetch_results")
    run = client.get(f"/runs/{fetched['result']['run_id']}").json()
    assert run["best_design_id"] == fetched["result"]["best_design_id"] and run["designs"][0]["episodes"]
    pack = next(d for k, d in frames if k == "tool.result" and d["name"] == "export_print_pack")["result"]
    response = client.get(pack["download"])
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert any(n.endswith("finger_left.stl") for n in zf.namelist())
    assert client.get("/runs/run-000000000000").status_code == 404
    assert client.get("/printpack/pack-000000000000").status_code == 404
    assert client.get("/printpack/..%2Fdaily_spend.json").status_code == 404
    assert client.get("/sessions/sess-000000000000").status_code == 404
    assert client.post(f"/sessions/{session_id}/stop").json() == {"stopping": False}


def test_one_session_at_a_time(client: TestClient, service: AgentService) -> None:
    part_id = upload(client).json()["part"]["part_id"]
    service.backend.gate.clear()  # hold the first session's job
    first = client.post("/sessions", json={"part_id": part_id, "passcode": PASSCODE})
    assert first.status_code == 201
    second = client.post("/sessions", json={"part_id": part_id, "passcode": PASSCODE})
    assert second.status_code == 429
    session_id = first.json()["session_id"]
    assert client.post(f"/sessions/{session_id}/stop").json() == {"stopping": True}
    service.backend.gate.set()
    frames = read_sse(client, f"/sessions/{session_id}/events")
    assert next(d for k, d in frames if k == "session.finished")["status"] == "stopped"


def test_cors_allows_local_dev_only(client: TestClient) -> None:
    headers = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}
    ok = client.options("/sessions", headers=headers)
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    bad = client.options("/sessions", headers={**headers, "Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_offline_mode_needs_no_passcode(tmp_path: Path) -> None:
    settings = ApiSettings(
        root=tmp_path / "agent", offline=True, fake_physics=True, caps=AgentCaps(poll_wait_s=5.0)
    )
    svc = AgentService(settings, gate=PasscodeGate(None))
    try:
        with TestClient(create_app(service=svc)) as client:
            part_id = upload(client).json()["part"]["part_id"]
            started = client.post("/sessions", json={"part_id": part_id})
            assert started.status_code == 201 and started.json()["offline"] is True
            frames = read_sse(client, f"/sessions/{started.json()['session_id']}/events")
            assert frames[0][1]["models"]["agent"] == "scripted-offline"
            assert not (tmp_path / "agent" / "daily_spend.json").exists()  # nothing billed, nothing booked
    finally:
        svc.shutdown()


def test_settings_from_env() -> None:
    settings = ApiSettings.from_env(
        {"RTT_AGENT_OFFLINE": "1", "RTT_AGENT_SESSION_USD": "0.2", "RTT_AGENT_CORS": "http://a:1, http://b:2"}
    )
    assert settings.offline and not settings.enable_nebius
    assert settings.caps.session_usd == 0.2 and settings.cors_origins == ("http://a:1", "http://b:2")


def test_sse_format() -> None:
    frame = sse_format({"seq": 3, "type": "tool.call", "name": "x"})
    assert frame.startswith("id: 3\nevent: tool.call\ndata: {") and frame.endswith("\n\n")


# --------------------------------------------------------------------------- hardening (verifier)


def test_upload_needs_a_binary_content_type(client: TestClient) -> None:
    # a cross-site page can send text/plain or a form without a CORS preflight; refuse those
    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        response = client.post("/parts", content=stl_bytes(), headers={"content-type": content_type})
        assert response.status_code == 415
    assert (
        client.post("/parts", content=stl_bytes(), headers={"content-type": "model/stl"}).status_code == 201
    )


def test_unknown_host_headers_are_refused(client: TestClient) -> None:
    # DNS rebinding: a page on rebind.example resolving to 127.0.0.1 sends its own Host
    assert client.get("/health", headers={"host": "rebind.example:8000"}).status_code == 400
    assert client.get("/health", headers={"host": "127.0.0.1:8000"}).status_code == 200
    assert client.get("/health", headers={"host": "localhost:8000"}).status_code == 200


def test_concurrent_starts_cannot_exceed_the_session_limit(client: TestClient, service: AgentService) -> None:
    import threading

    from rtt.api.service import BusyError

    part_id = upload(client).json()["part"]["part_id"]
    service.backend.gate.clear()  # every session that starts stays alive
    barrier = threading.Barrier(8)
    started: list[str] = []
    busy: list[BusyError] = []

    def start() -> None:
        barrier.wait()
        try:
            started.append(service.start_session(part_id, passcode=PASSCODE))
        except BusyError as exc:
            busy.append(exc)

    threads = [threading.Thread(target=start) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)
    assert len(started) == 1 and len(busy) == 7
    service.stop_session(started[0])
    service.backend.gate.set()
    service.wait_session(started[0], 20.0)


def test_offline_sessions_never_get_a_paid_target(tmp_path: Path) -> None:
    settings = ApiSettings(root=tmp_path / "agent", offline=True, fake_physics=True, enable_nebius=True)
    svc = AgentService(settings, gate=PasscodeGate(None), nebius_runner=object())  # type: ignore[arg-type]
    try:
        assert sorted(svc.runners()) == ["local"]
        with TestClient(create_app(service=svc)) as c:
            assert c.get("/health").json()["targets"] == ["local"]
    finally:
        svc.shutdown()


def test_failed_intake_leaves_no_part_folder(tmp_path: Path) -> None:
    class Broken(FakePartBackend):
        def intake(self, stl_path: Path, **kwargs: Any) -> Any:
            raise ValueError(f"cannot read {tmp_path}/deep/folder/{Path(stl_path).name}")

    settings = ApiSettings(root=tmp_path / "agent", offline=True)
    svc = AgentService(settings, backend=Broken(tmp_path / "agent" / "parts"), gate=PasscodeGate(None))
    try:
        with TestClient(create_app(service=svc)) as c:
            response = upload(c)
            assert response.status_code == 422
            assert (
                "source.stl" in response.json()["detail"] and str(tmp_path) not in response.json()["detail"]
            )
        assert not any((tmp_path / "agent" / "parts").glob("part-*"))
    finally:
        svc.shutdown()


def test_offline_sessions_record_no_spend(tmp_path: Path) -> None:
    settings = ApiSettings(
        root=tmp_path / "agent", offline=True, fake_physics=True, caps=AgentCaps(poll_wait_s=5.0)
    )
    svc = AgentService(settings, gate=PasscodeGate(None))
    try:
        part = svc.upload_part(stl_bytes())
        session_id = svc.start_session(part.part_id)
        result = svc.wait_session(session_id, 60.0)
        assert result is not None and result.status == "completed"
        doc = svc.session(session_id).to_json()
        calls = [e for e in doc["events"] if e["type"] == "llm.call"]
        assert calls and all(e["cost_usd"] == 0.0 for e in calls)  # no model was called, nothing billed
        assert doc["spend"]["spent_usd"] == 0.0 and doc["spend"]["by_kind_usd"] == {}
    finally:
        svc.shutdown()


def test_day_book_does_not_follow_the_root() -> None:
    # a different --root must not start a fresh day: the day cap is shared
    from rtt.api.service import DEFAULT_ROOT

    settings = ApiSettings.from_env({"RTT_AGENT_ROOT": "/tmp/elsewhere"})
    assert settings.day_book == DEFAULT_ROOT / "daily_spend.json"
    assert ApiSettings.from_env({"RTT_AGENT_DAY_BOOK": "/tmp/book.json"}).day_book == Path("/tmp/book.json")
    assert ApiSettings().day_book is None  # direct construction (tests): <root>/daily_spend.json


# --------------------------------------------------------------------------- hardening (verifier, round 2)


@pytest.fixture
def public_service(tmp_path: Path) -> Iterator[AgentService]:
    """A server reachable beyond loopback: uploads and offline sessions need the passcode too."""
    settings = ApiSettings(
        root=tmp_path / "agent",
        offline=True,
        fake_physics=True,
        require_passcode=True,
        caps=AgentCaps(poll_wait_s=5.0),
    )
    svc = AgentService(settings, gate=PasscodeGate(passcode_digest(PASSCODE)))
    yield svc
    svc.shutdown()


def test_public_server_needs_the_passcode_for_uploads(public_service: AgentService) -> None:
    with pytest.raises(PasscodeRequiredError):  # the service checks too, not only the HTTP layer
        public_service.upload_part(stl_bytes())
    with TestClient(create_app(service=public_service)) as c:
        assert upload(c).status_code == 403
        assert (
            c.post(
                "/parts", content=stl_bytes(), headers={"content-type": "model/stl", "X-RTT-Passcode": "nope"}
            ).status_code
            == 403
        )
        ok = c.post(
            "/parts", content=stl_bytes(), headers={"content-type": "model/stl", "X-RTT-Passcode": PASSCODE}
        )
        assert ok.status_code == 201, ok.text
        part_id = ok.json()["part"]["part_id"]
        # offline sessions cost nothing, but on a public server they still need the passcode
        assert c.post("/sessions", json={"part_id": part_id}).status_code == 403
        started = c.post("/sessions", json={"part_id": part_id, "passcode": PASSCODE})
        assert started.status_code == 201, started.text
        public_service.wait_session(started.json()["session_id"], 60.0)
        assert c.get("/health").json()["passcode_for_uploads"] is True
        wrong = {"content-type": "model/stl", "X-RTT-Passcode": "guess"}
        for _ in range(4):  # one wrong guess above + 4 more: locked, even for the right passcode
            assert c.post("/parts", content=stl_bytes(), headers=wrong).status_code == 403
        right = {"content-type": "model/stl", "X-RTT-Passcode": PASSCODE}
        assert c.post("/parts", content=stl_bytes(), headers=right).status_code == 429


def test_passcode_is_checked_before_the_part_is_looked_up(service: AgentService) -> None:
    looked_up: list[str] = []
    real_info = service.backend.info

    def info(part_id: str) -> Any:
        looked_up.append(part_id)
        return real_info(part_id)

    service.backend.info = info  # type: ignore[method-assign]
    with pytest.raises(PasscodeRequiredError):
        service.start_session("part-0123456789ab", passcode=None)
    assert looked_up == []  # an unauthenticated request cannot trigger a part reload (a full intake)
    with pytest.raises(PartNotFoundError):
        service.start_session("not-a-part", passcode=PASSCODE)


def test_unusable_nebius_target_is_a_503_and_leaves_no_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rtt.agent.jobs import NebiusJobRunner

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(f"no price for cpu-d3 in {tmp_path}/deep/config/nebius.env")

    monkeypatch.setattr(NebiusJobRunner, "from_config", broken)
    settings = ApiSettings(root=tmp_path / "agent", enable_nebius=True, caps=AgentCaps(poll_wait_s=5.0))
    svc = AgentService(
        settings,
        backend=FakePartBackend(tmp_path / "agent" / "parts"),
        gate=PasscodeGate(passcode_digest(PASSCODE)),
        transport_factory=ScriptedAgentTransport,
    )
    try:
        with TestClient(create_app(service=svc)) as c:
            part_id = upload(c).json()["part"]["part_id"]
            response = c.post("/sessions", json={"part_id": part_id, "passcode": PASSCODE})
            assert response.status_code == 503, response.text  # not "no such part"
            assert str(tmp_path) not in response.json()["detail"]
        assert not list((tmp_path / "agent").glob("sessions/*.json"))  # no session left "running"
    finally:
        svc.shutdown()


def test_is_loopback() -> None:
    from rtt.api.__main__ import is_loopback

    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert is_loopback("127.0.0.2") and is_loopback("[::1]")
    assert not is_loopback("0.0.0.0") and not is_loopback("::") and not is_loopback("192.168.1.5")
    assert not is_loopback("myhost.local")


def test_settings_from_env_require_passcode() -> None:
    assert ApiSettings.from_env({"RTT_AGENT_REQUIRE_PASSCODE": "1"}).require_passcode
    assert not ApiSettings.from_env({}).require_passcode
    with pytest.raises(ValueError):
        ApiSettings.from_env({"RTT_AGENT_SESSION_USD": "nan"})


def _flat_box(tilt_rad: float = 0.0) -> bytes:
    import numpy as np

    box = trimesh.creation.box(extents=(40.0, 30.0, 60.0))
    flat = trimesh.Trimesh(box.vertices * np.array([1.0, 1.0, 0.0]), box.faces, process=False)
    flat.apply_transform(trimesh.transformations.rotation_matrix(tilt_rad, [1.0, 0.0, 0.0]))
    flat.apply_translation([5.0, 7.0, 9.0])
    return flat.export(file_type="stl")


def test_flat_uploads_are_refused_before_intake(client: TestClient, service: AgentService) -> None:
    # A flat (or nearly flat) mesh makes trimesh's stable-pose search loop forever, so it
    # must never reach the intake: one 684-byte upload would pin a CPU core for good.
    called: list[Path] = []
    real_intake = service.backend.intake

    def intake(stl_path: Path, **kwargs: Any) -> Any:
        called.append(stl_path)
        return real_intake(stl_path, **kwargs)

    service.backend.intake = intake  # type: ignore[method-assign]
    thin = trimesh.creation.box(extents=(40.0, 30.0, 1e-6)).export(file_type="stl")
    for data in (_flat_box(), _flat_box(0.5), thin):
        response = upload(client, data)
        assert response.status_code == 422 and "flat" in response.json()["detail"], response.text
    assert called == [] and not any((service.store.parts_dir).glob("part-*"))
    plate = trimesh.creation.box(extents=(40.0, 30.0, 0.5)).export(file_type="stl")
    assert upload(client, plate).status_code == 201  # thin but solid is fine
    assert check_solid(stl_bytes((0.04, 0.03, 0.06)), units="m") == pytest.approx(13.333, rel=1e-3)
