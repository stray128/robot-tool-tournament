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
