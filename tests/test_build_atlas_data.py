"""The atlas exporter's offline parts: telling names from words, the screen, and the assembly.

The rewrite itself calls the Claude API and is not run here; these tests use small made-up records.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
WORDS = {
    "hospital",
    "robot",
    "suction",
    "cup",
    "tool",
    "origin",
    "construction",
    "robotics",
    "the",
    "of",
    "frame",
}


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_atlas_data", REPO / "scripts" / "build_atlas_data.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_atlas_data"] = mod
    spec.loader.exec_module(mod)
    return mod


atlas = _module()


def test_fragments_split_names_and_drop_citation_details() -> None:
    frags = atlas._fragments("Hospital robot (Acmebot X2, now Zeta Robotics)")
    assert "Acmebot X2" in frags and "Zeta Robotics" in frags
    assert atlas._fragments("Patent tool (patent US1234567)") == ["Patent tool"]
    assert atlas._fragments("Foo Systems (US)") == ["Foo Systems"]


def test_names_are_told_from_descriptions() -> None:
    got = atlas.classify_names(
        [
            "Hospital robot",
            "Suction cup tool",
            "Acmebot X2",
            "Origin",
            "Construction Robotics",
            "a robot",
            "1234",
        ],
        WORDS,
    )
    assert got["ci"] == ["acmebot x2"]  # matched in any case
    assert got["cs"] == ["Construction Robotics", "Origin"]  # ordinary words, matched as written
    screen = atlas.Screen(got)
    assert screen.hits("an ACMEBOT X2 gripper") == ["acmebot x2"]
    assert screen.hits("the origin of the frame") == []
    assert screen.hits("Origin builds robots") == ["Origin"]
    assert screen.hits("a Robotiq gripper") == ["robotiq"]  # the library's list applies too


def _record(**kw: object) -> object:
    base = {
        "id": "T1",
        "domain": "construction",
        "setting": "construction",
        "title": "Lay bricks",
        "workflow": "",
        "subtasks": [{"verb": "pick", "object": "brick", "notes": ""}],
        "kind": "single",
        "tools": ["gripper"],
        "why": "x",
        "feasibility": "rigid",
        "demo_score": 4,
        "sources": ["https://example.org/a"],
    }
    base.update(kw)
    return atlas.Record(**base)


def test_assembly_keeps_urls_as_targets_and_the_screen_fails_on_a_brand() -> None:
    rec = _record()
    out = {
        "T1": {
            "id": "T1",
            "title": "Lay bricks",
            "subtasks": [{"verb": "Pick", "object": "brick", "note": "From the pallet — one at a time."}],
            "tools": ["Parallel gripper"],
            "why": "One gripper lays every brick.",
        }
    }
    doc = atlas.assemble([rec], [], out, {}, "2026-10-26")
    task = doc["tasks"][0]
    assert task["subtasks"][0] == {
        "verb": "pick",
        "object": "brick",
        "note": "From the pallet, one at a time.",
    }
    assert task["sources"] == [{"url": "https://example.org/a"}]
    assert task["sim"] == {"feasibility": "rigid", "hero": None, "match": None}
    screen = atlas.Screen({"ci": ["acmebot"], "cs": []})
    assert atlas.screen_atlas(doc, screen) == []
    task["why"] = "Like the Acmebot cell, see https://acme.example"
    problems = atlas.screen_atlas(doc, screen)
    assert any("names acmebot" in p for p in problems)
    assert any("URL outside a url field" in p for p in problems)


def test_a_shape_problem_is_refused() -> None:
    rec = _record()
    bad = {"T1": {"id": "T1", "title": "Lay bricks", "subtasks": [], "tools": [], "why": "x"}}
    with pytest.raises(atlas.AtlasError, match="unusable"):
        atlas.assemble([rec], [], bad, {}, "2026-10-26")


def test_the_heroes_name_six_records() -> None:
    assert sorted(atlas.HERO_RECORDS) == [
        "h1_tote",
        "h2_coupling",
        "h4_cnc",
        "h5_solar",
        "h6_tile",
        "h7_hotel",
    ]


def test_the_published_atlas_passes_the_committed_screen() -> None:
    data = json.loads((REPO / "web" / "public" / "data" / "atlas.json").read_text(encoding="utf-8"))
    tokens = json.loads((REPO / "tests" / "data" / "atlas_brand_tokens.json").read_text(encoding="utf-8"))
    assert len(data["tasks"]) == 95
    assert atlas.screen_atlas(data, atlas.Screen(tokens)) == []


# --------------------------------------------------------------------------- the Claude client, stubbed


class _Usage:
    def __init__(self, inp: int, out: int) -> None:
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Msg:
    def __init__(self, data: object, usage: _Usage, stop: str = "end_turn") -> None:
        self.model = "claude-opus-5-5"
        self.usage = usage
        self.stop_reason = stop
        self.content = [_Text(json.dumps(data))]


class StubClient:
    """Stands in for anthropic.Anthropic: answers by the requested schema, records every call."""

    def __init__(self, answer, usage: _Usage | None = None) -> None:
        self.calls: list[dict] = []
        self._answer = answer
        self._usage = usage or _Usage(1000, 500)
        outer = self

        class _Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                return _Msg(outer._answer(kw), outer._usage)

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def _kind(kw: dict) -> str:
    return next(iter(kw["output_config"]["format"]["schema"]["properties"]))


def _items(kw: dict) -> list[dict]:
    text = kw["messages"][0]["content"]
    return json.loads(text[text.index("[") : text.rindex("]") + 1])


def _rewriter(tmp_path: Path, client: StubClient | None, cap: float = 4.5, offline: bool = False):
    rw = atlas.Rewriter(tmp_path / "cache", None, cap, offline)
    rw._client = client
    return rw


def _clean_answer(kw: dict) -> dict:
    kind = _kind(kw)
    items = _items(kw)
    if kind == "records":
        return {
            "records": [
                {
                    "id": it["id"],
                    "title": it["title"],
                    "subtasks": [{"verb": "pick", "object": "brick", "note": ""} for _ in it["subtasks"]],
                    "tools": ["Parallel gripper"],
                    "why": "One gripper lays every brick.",
                }
                for it in items
            ]
        }
    if kind == "rules":
        return {
            "rules": [{"id": it["id"], "title": "One tool first", "text": "Use one tool."} for it in items]
        }
    return {"names": ["Acmebot X2", "Zeta Robotics", "Origin"]}


def test_the_rewriter_caches_answers_and_replays_them_offline(tmp_path: Path) -> None:
    stub = StubClient(_clean_answer)
    rw = _rewriter(tmp_path, stub)
    rec = _record()
    first = rw.ask("records", atlas._RECORD_SCHEMA, [rec.prompt_view()], ())
    again = rw.ask("records", atlas._RECORD_SCHEMA, [rec.prompt_view()], ())
    assert first == again and len(stub.calls) == 1  # the second ask is served from the cache
    assert rw.calls == 1 and rw.spent == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    kw = stub.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["effort"] == "low"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert (tmp_path / "cache" / "spend.jsonl").read_text().count("\n") == 1
    # a new run on the same cache, offline, needs no client
    offline = _rewriter(tmp_path, None, offline=True)
    assert offline.ask("records", atlas._RECORD_SCHEMA, [rec.prompt_view()], ()) == first
    with pytest.raises(atlas.AtlasError, match="offline"):
        offline.ask("records", atlas._RECORD_SCHEMA, [_record(id="T2").prompt_view()], ())


def test_the_spend_cap_stops_before_a_call_could_pass_it(tmp_path: Path) -> None:
    # A cap below the assumed worst call: nothing is sent.
    stub = StubClient(_clean_answer)
    with pytest.raises(atlas.AtlasError, match="spend cap"):
        _rewriter(tmp_path / "a", stub, cap=0.5).ask(
            "records", atlas._RECORD_SCHEMA, [_record().prompt_view()], ()
        )
    assert stub.calls == []
    # An expensive answer ($20.004 of output) raises the worst-call estimate; the next call is refused.
    big = StubClient(_clean_answer, _Usage(1000, 1_000_000))
    rw = _rewriter(tmp_path / "b", big, cap=30.0)
    rw.ask("records", atlas._RECORD_SCHEMA, [_record().prompt_view()], ())
    assert rw.spent == pytest.approx(20.004)
    with pytest.raises(atlas.AtlasError, match="spend cap"):
        rw.ask("records", atlas._RECORD_SCHEMA, [_record(id="T2").prompt_view()], ())
    assert len(big.calls) == 1


def test_the_second_pass_names_leftover_brands_and_fixes_shapes(tmp_path: Path) -> None:
    def answer(kw: dict) -> dict:
        out = _clean_answer(kw)
        prompt = kw["messages"][0]["content"]
        retry = "An earlier rewrite still named these" in prompt
        if _kind(kw) == "records" and not retry:
            out["records"][0]["why"] = "Like the Acmebot X2 cell, one gripper lays every brick."
            out["records"][1]["subtasks"] = []  # a shape problem
        return out

    stub = StubClient(answer)
    rw = _rewriter(tmp_path, stub)
    screen = atlas.Screen({"ci": ["acmebot x2"], "cs": []})
    records = [_record(id="T1"), _record(id="T2")]
    rules = [atlas.Rule(id="R01_x", condition="c", recommendation="r", rationale="why", sources=[])]
    rec_out, rule_out = atlas.rewrite_all(records, rules, rw, screen)
    retries = [
        kw for kw in stub.calls if "An earlier rewrite still named these" in kw["messages"][0]["content"]
    ]
    assert len(retries) == 2  # one per broken record, each alone
    assert "acmebot x2" in retries[0]["messages"][0]["content"]
    assert [_items(kw)[0]["id"] for kw in retries] == ["T1", "T2"]
    assert all(not screen.hits(atlas._strings(e)[1]) for e in rec_out.values())
    assert rec_out["T2"]["subtasks"] and rule_out["R01_x"]["text"] == "Use one tool."
    doc = atlas.assemble(records, rules, rec_out, rule_out, "2026-10-26")
    assert atlas.screen_atlas(doc, screen) == []


def test_extract_brand_names_joins_the_rules_and_the_model(tmp_path: Path) -> None:
    rec = _record(
        products=[{"name": "Hospital robot (Acmebot X2)", "maker": "Zeta Robotics (US)"}],
        prose=["Uses Origin's frame."],
    )
    stub = StubClient(_clean_answer)
    rw = _rewriter(tmp_path, stub)
    names = atlas.extract_brand_names([rec], [], WORDS, rw)
    assert {_kind(kw) for kw in stub.calls} == {"names"}
    assert len(stub.calls) == 2  # the product strings, then the prose
    assert "Hospital robot (Acmebot X2)" in stub.calls[0]["messages"][0]["content"]
    assert "Origin's frame" in stub.calls[1]["messages"][0]["content"]
    assert "acmebot x2" in names["ci"] and "zeta robotics" in names["ci"]
    assert "Origin" in names["cs"]
    assert "hospital robot" not in names["ci"]  # a description, not a name
    # without a client: the rule-based names alone
    assert atlas.extract_brand_names([rec], [], WORDS, None)["ci"] == ["acmebot x2", "zeta robotics"]
