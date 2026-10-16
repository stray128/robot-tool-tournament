"""No held-out data in any prompt.

Runs both LLM arms with a recording model while a held-out seed sits in a ``secrets``
folder, then checks every request payload (and the response cache on disk) for the
seed, its commitment, held-out vocabulary and held-out conditions. Every condition a
prompt names must belong to the training suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rtt.llm.client import TokenBucket
from rtt.search.fakes import FakeEvaluator, ScriptedProposer
from rtt.search.nemotron import make_proposer_client
from rtt.search.nemotron_v2 import FORBIDDEN_PROMPT_TEXT_V2
from rtt.search.runner import run_search
from rtt.store.vault import HeldoutVault, seed_sha256

SEED = "5f" * 32  # 64 hex digits, like secrets.token_hex(32)
HELDOUT_WORDS = re.compile(r"held[\s_-]?out|heldout_seed|secret", re.IGNORECASE)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("noheldout")
    vault = HeldoutVault.create(root / "secrets" / "heldout_seed.txt", SEED)
    payloads: list[dict] = []
    for arm in ("nemotron", "zeroshot"):
        model = ScriptedProposer(5, invalid_rate=0.3, similar_rate=0.3, garbage_rate=0.1)
        client = make_proposer_client(root / arm / "cache", transport=model, limiter=TokenBucket(1e12))
        run_search(
            arm, 2, 40, [1, 2], FakeEvaluator(0), root / arm, client, protected_paths=[root / "secrets"]
        )
        payloads.extend(model.payloads)
    return {"root": root, "vault": vault, "payloads": payloads}


def _texts(payloads: list[dict]) -> list[str]:
    return [json.dumps(p, ensure_ascii=False) for p in payloads]


def test_prompts_carry_no_seed_commitment_or_held_out_words(recorded) -> None:
    texts = _texts(recorded["payloads"])
    assert len(texts) > 20
    commitment = seed_sha256(SEED)
    for text in texts:
        assert SEED not in text and SEED[:16] not in text
        assert commitment not in text and commitment[:16] not in text
        assert not HELDOUT_WORDS.search(text), HELDOUT_WORDS.search(text)


def test_cache_files_carry_no_held_out_material(recorded) -> None:
    root: Path = recorded["root"]
    files = list(root.glob("*/cache/??/*.json"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert SEED not in text and seed_sha256(SEED) not in text
        assert not HELDOUT_WORDS.search(text)


def test_prompts_name_only_training_conditions(recorded) -> None:
    training = {c.condition_id for c in FakeEvaluator(0).describe().conditions}
    pattern = re.compile(r"\b(?:box|cylinder|rod)_[a-z]_[a-z]\b")
    named: set[str] = set()
    for payload in recorded["payloads"]:
        for message in payload["messages"]:
            named |= set(pattern.findall(message["content"]))
    assert named and named <= training


def test_simulator_held_out_conditions_never_appear(recorded) -> None:
    conditions = pytest.importorskip("rtt.sim.conditions")
    try:
        suite = conditions.heldout_suite(SEED, commitment=seed_sha256(SEED))
    except (TypeError, ValueError, OSError) as exc:  # the simulator's held-out API is optional here
        pytest.skip(f"held-out suite unavailable: {exc}")
    texts = _texts(recorded["payloads"])
    for cond in suite:
        for text in texts:
            assert cond.condition_id not in text


# --------------------------------------------------------------------------- freeform-v2: parts, families, dimensions

UNSEEN_FAMILIES = re.compile(r"\belbows?\b|\bspur\b|\bgears?\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def recorded_v2(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    from rtt.search.fakes_v2 import FakeEvaluatorV2, ScriptedProposerV2, stub_entrants

    root = tmp_path_factory.mktemp("noheldout_v2")
    HeldoutVault.create(root / "secrets" / "heldout_seed.txt", SEED)
    payloads: list[dict] = []
    for arm in ("nemotron", "zeroshot"):
        model = ScriptedProposerV2(5, invalid_rate=0.3, similar_rate=0.3, garbage_rate=0.1)
        client = make_proposer_client(root / arm / "cache", transport=model, limiter=TokenBucket(1e12))
        run_search(
            arm,
            2,
            40,
            [1, 2],
            FakeEvaluatorV2(0),
            root / arm,
            client,
            protected_paths=[root / "secrets"],
            entrants=stub_entrants(),
        )
        payloads.extend(model.payloads)
    return {"root": root, "payloads": payloads}


def test_v2_prompts_carry_no_seed_held_out_words_or_unseen_families(recorded_v2) -> None:
    texts = _texts(recorded_v2["payloads"])
    assert len(texts) > 20
    commitment = seed_sha256(SEED)
    for text in texts:
        assert SEED[:16] not in text and commitment[:16] not in text
        assert not HELDOUT_WORDS.search(text), HELDOUT_WORDS.search(text)
        assert not UNSEEN_FAMILIES.search(text), UNSEEN_FAMILIES.search(text)
    for path in recorded_v2["root"].glob("*/cache/??/*.json"):
        text = path.read_text(encoding="utf-8")
        assert not HELDOUT_WORDS.search(text) and not UNSEEN_FAMILIES.search(text)


def test_v2_prompts_name_only_training_parts_and_their_dimensions(recorded_v2) -> None:
    from rtt.search.fakes_v2 import DEFAULT_FAKE_PARTS_V2

    training = {p.part_id for p in DEFAULT_FAKE_PARTS_V2}
    sizes = {" / ".join(f"{v:g}" for v in p.size_mm) for p in DEFAULT_FAKE_PARTS_V2}
    table_row = re.compile(r"^\| ([a-z_]+) \| [^|]* \| ([0-9. /]+) \| ", re.MULTILINE)
    named: set[str] = set()
    for payload in recorded_v2["payloads"]:
        for message in payload["messages"]:
            text = message["content"]
            if "# Training parts" not in text:
                continue
            table = text.split("# Training parts")[1].split("# Placements")[0]
            for part, size in table_row.findall(table):
                named.add(part)
                assert size.strip() in sizes, size
            for label in re.findall(r"\b([a-z_]+)_c[0-9]\b", text):  # condition ids: <part>_<placement>
                assert label in training, label
    assert named == training


def test_v2_held_out_parts_from_track_o_never_appear(recorded_v2) -> None:
    """Track O's held-out procedural parts (drawn from the secret): no name, id or size reaches a prompt."""
    heldout = pytest.importorskip("rtt.parts.heldout")
    make = getattr(heldout, "heldout_procedural_suite", None)
    if make is None:
        pytest.skip("rtt.parts.heldout has no heldout_procedural_suite(secret) yet")
    suite = make(SEED, n=40)
    texts = _texts(recorded_v2["payloads"])
    assert texts
    from rtt.search.fakes_v2 import DEFAULT_FAKE_PARTS_V2

    training_sizes = {" / ".join(f"{v:g}" for v in p.size_mm) for p in DEFAULT_FAKE_PARTS_V2}
    for cond in suite:
        part = cond.part
        size = " / ".join(f"{round(1000.0 * float(v), 1):g}" for v in part.size)
        for text in texts:
            assert str(part.part_id) not in text and cond.condition_id not in text
            if size not in training_sizes:
                assert size not in text, (part.part_id, size)
        assert FORBIDDEN_PROMPT_TEXT_V2.search(str(part.part_id))  # the guard would refuse the id itself


def test_v2_guard_refuses_a_misconfigured_training_suite(tmp_path) -> None:
    """If an unseen held-out family ever reached the training parts, no prompt would be sent."""
    from dataclasses import replace

    from rtt.search.fakes_v2 import DEFAULT_FAKE_PARTS_V2, FakeEvaluatorV2, ScriptedProposerV2, stub_entrants
    from rtt.search.nemotron import PromptLeakError

    leaked = (*DEFAULT_FAKE_PARTS_V2[:5], replace(DEFAULT_FAKE_PARTS_V2[5], description="90 deg pipe elbow"))
    model = ScriptedProposerV2(1)
    client = make_proposer_client(None, transport=model, limiter=TokenBucket(1e12))
    with pytest.raises(PromptLeakError):
        run_search(
            "nemotron",
            1,
            16,
            [1],
            FakeEvaluatorV2(0, parts=leaked),
            tmp_path / "x",
            client,
            entrants=stub_entrants(),
        )
    assert model.calls == 0
