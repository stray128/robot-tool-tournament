"""Track S4's gate: the gate descriptions through the Task Engineer, and the manifest of what came back.

:func:`run_gate` plans every :data:`~rtt.studio.engineer.examples.GATE_CASES` description (or a
subset) with one client, in order, and :func:`gate_manifest` summarises the results:

* **plans**: the three hero use cases and the five other realistic ones must give a valid
  ``TaskSpec`` (on the first call, after the one repair call, or after the deterministic layout
  fix that follows it), in at least 90 % of the cases;
* **refusals**: the deliberately unsupported ones must be refused, with a reason and at least
  one nearest primitive.

The gate is judged on **one** full live run: a case that fails stays recorded as failed (track
S4e); the manifest's summary counts each case once, under its ``result``. A case may be run again
live once its checks change or its description is made clearer (:func:`merge_rerun`): the attempt
it replaces stays in the manifest under ``reruns``, in the cache and in the ledger.

Live (``--live``) the run calls Token Factory under a hard cap on everything recorded in its
ledger; without it the run replays the recorded cache with zero API calls, and the test suite
checks that the replay reproduces the recorded manifest.

**Two recordings.** :data:`RECORDED_GATE` is Studio v2's (the six hero jobs, recorded with the
prompts that teach the job templates); it is the default replay cache of the command line, the
API and the agent's tools, and a replay rebuilds the very same requests. A re-run may answer a
later prompt than the first run (v2.1 after v2, both with rule set v2): each case row names the
prompt it answered (``prompt_version``), the manifest lists every prompt used (``prompts``), and
its top-level ``prompt_version``, ``tool`` and ``template_hashes`` stay the first run's. :data:`RECORDED_GATE_V1`
is the v1 run whose hero plans the pre-registered use-case benchmark v1 froze. Its requests were
built by the v1 prompt and tool, which the current code no longer renders, so
:func:`replay_recorded` replays it from what was recorded: each case's text and the model's
recorded responses, in order, judged by the current checks (its manifest names the v1 prompt's
template hashes and tool sha256, and every cache entry still hashes to its key). The test suite
checks that this replay reproduces the v1 manifest and its frozen task ids.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.llm.cache import CacheEntry, CacheMiss, ResponseCache, wire_json
from rtt.llm.client import LLMClient
from rtt.llm.ledger import read_ledger
from rtt.studio.engineer import prompts
from rtt.studio.engineer.checks import RULES_REVISION
from rtt.studio.engineer.engineer import (
    COMPACT_TOOL,
    ENGINEER_MODEL,
    ENGINEER_SAMPLING,
    PlanResult,
    engineer_tool,
    make_engineer_client,
    plan_task,
    tool_sha256,
)
from rtt.studio.engineer.examples import GATE_CASES, UseCase
from rtt.studio.engineer.uploads import UploadInfo
from rtt.studio.types import CONTRACT_VERSION

__all__ = [
    "GATE_VALID_RATE",
    "RECORDED_GATE",
    "RECORDED_GATE_V1",
    "case_row",
    "gate_manifest",
    "merge_rerun",
    "recorded_cases",
    "recorded_version",
    "recorded_versions",
    "replay_plan",
    "replay_recorded",
    "responses_client",
    "run_gate",
    "superseded_cache",
]

#: Share of the plan cases that must give a valid TaskSpec after at most one repair.
GATE_VALID_RATE: Final = 0.9
#: The recorded live gate run of Studio v2 (scrubbed cache, ledger, manifest): the six hero jobs.
RECORDED_GATE: Final = Path(__file__).with_name("recorded") / "gate-2026-10-27-v2c"
#: The recorded live gate run of v1, frozen by the use-case benchmark v1 (module docstring).
RECORDED_GATE_V1: Final = Path(__file__).with_name("recorded") / "gate-2026-10-18-s4e"


def run_gate(
    client: LLMClient,
    cases: Sequence[UseCase] = GATE_CASES,
    *,
    compact: bool = COMPACT_TOOL,
    version: str = prompts.CURRENT,
    versions: Mapping[str, str] | None = None,
) -> list[tuple[UseCase, PlanResult]]:
    """Plan every case in order with ``client`` (one repair each), on prompt ``version`` and its
    rule set, or the prompt ``versions`` names for a case (a recorded run's, :func:`recorded_versions`)."""
    return [
        (
            case,
            plan_task(
                case.text,
                client=client,
                compact=compact,
                tag=f"gate/{case.case_id}",
                version=(versions or {}).get(case.case_id, version),
            ),
        )
        for case in cases
    ]


def replay_plan(
    text: str, cache_dir: Path, uploads: Sequence[UploadInfo] = (), *, compact: bool = COMPACT_TOOL
) -> PlanResult:
    """``text`` planned from a recorded run's replay cache (:class:`CacheMiss` when it is not recorded):
    a description of the run beside ``cache_dir`` is replayed with the prompt its row answered (a
    re-run may have answered a later prompt than the rest), any other with the current prompt."""
    path = cache_dir.parent / "manifest.json"
    versions: list[str] = []
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        versions = [_row_version(manifest, r) for r in manifest.get("cases") or [] if r.get("text") == text]
    miss: CacheMiss | None = None
    for version in dict.fromkeys([*versions, prompts.CURRENT]):
        try:
            client = make_engineer_client(cache_dir, cache_mode="replay")
            return plan_task(text, client=client, uploads=uploads, compact=compact, version=version)
        except CacheMiss as exc:
            miss = exc
    assert miss is not None
    raise miss


def recorded_versions(recorded: Path) -> dict[str, str]:
    """The prompt each case of a recorded run answered (case id to version)."""
    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    return {r["case_id"]: _row_version(manifest, r) for r in manifest["cases"]}


def recorded_version(recorded: Path) -> str:
    """The prompt a recorded run's first run used (``prompt_version``; v1 recordings predate it)."""
    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest.get("prompt_version", "v1"))


def _row_version(manifest: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """The prompt a case row answered (a re-run may answer a later one than the first run)."""
    return str(row.get("prompt_version", manifest.get("prompt_version", "v1")))


def _prompt_of(manifest: Mapping[str, Any], version: str) -> dict[str, Any]:
    """The tool and template hashes of prompt ``version`` as a recorded manifest names them."""
    used = manifest.get("prompts") or {}
    if version in used:
        return dict(used[version])
    return {"tool": manifest["tool"], "template_hashes": manifest["template_hashes"]}


class _RecordedResponses(ResponseCache):
    """A read-only cache that answers the calls of one recorded case with its recorded entries, in
    the order they were made, whatever the request now hashes to; an entry that no longer hashes
    to its own key, or one call too many, is an error (:func:`replay_recorded`)."""

    def __init__(self, root: Path, hashes: Sequence[str]) -> None:
        super().__init__(root, mode="replay")
        self._hashes = list(hashes)

    def get(self, key: str) -> CacheEntry | None:
        if not self._hashes:
            raise CacheMiss(key)
        return super().get(self._hashes.pop(0))

    def put(self, entry: CacheEntry) -> Path | None:
        return None


def responses_client(cache_dir: Path, hashes: Sequence[str]) -> LLMClient:
    """A Task Engineer client that answers its calls with the recorded entries ``hashes`` of
    ``cache_dir``, in order (:class:`_RecordedResponses`): a recorded plan replayed from what the
    model returned, whatever prompt the current code renders."""
    client = make_engineer_client(None, cache_mode="replay")
    client.cache = _RecordedResponses(cache_dir, hashes)
    return client


def recorded_cases(recorded: Path) -> tuple[UseCase, ...]:
    """The cases of a recorded gate run, from its manifest, in the order they ran."""
    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    return tuple(
        UseCase(r["case_id"], r["setting"], r["title"], r["text"], r["expect"]) for r in manifest["cases"]
    )


def replay_recorded(recorded: Path) -> list[tuple[UseCase, PlanResult]]:
    """Every case of a recorded gate run, planned again from its recorded responses (module
    docstring): zero calls, the current checks, and the recorded run's template hashes and tool
    sha256 on each result (the prompt those responses answered)."""
    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for case, row in zip(recorded_cases(recorded), manifest["cases"], strict=True):
        version = _row_version(manifest, row)
        prompt = _prompt_of(manifest, version)
        client = responses_client(recorded / "cache", [a["request_hash"] for a in row["attempts"]])
        result = plan_task(case.text, client=client, tag=f"gate/{case.case_id}", version=version)
        result = replace(
            result,
            template_hashes=dict(prompt["template_hashes"]),
            tool_sha256=prompt["tool"]["sha256"],
        )
        out.append((case, result))
    return out


def _recorded_prompt(recorded: Path | None) -> dict[str, Any]:
    if recorded is None:
        return {}
    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    out = {"tool": manifest["tool"], "template_hashes": manifest["template_hashes"]}
    if "prompts" in manifest:
        out["prompts"] = manifest["prompts"]
    return out


def _truncated(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The calls of ``rows`` cut off at ``max_tokens`` (finish_reason ``length``), and their cases:
    a truncated call is never a valid plan (rule set v2 does not judge it; one that does not parse
    is a parse error), and the manifest says so, whatever the case's verdict came from."""
    hits = [
        (r["case_id"], a["stage"]) for r in rows for a in r["attempts"] if a.get("finish_reason") == "length"
    ]
    return {"calls": len(hits), "cases": list(dict.fromkeys(c for c, _ in hits))}


def _row_truncated(row: Mapping[str, Any]) -> list[str]:
    return [a["stage"] for a in row["attempts"] if a.get("finish_reason") == "length"]


def case_row(case: UseCase, result: PlanResult) -> dict[str, Any]:
    """One case of the manifest: the verdict, the steps produced and how it got there."""
    spec = result.spec
    steps = [
        {"id": s.id, "primitive": s.primitive.value, "object": s.object, "target": s.target}
        for s in (spec.steps if spec is not None else ())
    ]
    nearest = [u.nearest_primitive.value for u in spec.unsupported if u.nearest_primitive] if spec else []
    if case.expect == "plan":
        passed = result.status == "valid"
    else:
        passed = result.status == "refused" and spec is not None and bool(spec.refusal) and bool(nearest)
    return {
        "case_id": case.case_id,
        "setting": case.setting,
        "title": case.title,
        "expect": case.expect,
        "text": case.text,
        "status": result.status,
        "passed": passed,
        "first_try": result.first_try,
        "repaired": result.repaired,
        "layout_fixed": result.layout_fixed,
        "result": result.result,
        "layout_moves": list(result.layout_moves),
        "calls": len(result.attempts),
        "steps": steps,
        "n_objects": len(spec.objects) if spec is not None else 0,
        "refusal": spec.refusal if spec is not None else None,
        "nearest_primitives": nearest,
        "task_id": spec.task_id if spec is not None and spec.steps else None,
        "problems": list(result.problems),
        "first_problems": list(result.attempts[0].problems) if result.attempts else [],
        "warnings": list(result.warnings),
        "notes": list(result.notes),
        "cost_usd": round(result.cost_usd, 6),
        "truncated": list(result.truncated),
        "prompt_version": result.version,
        "attempts": [a.to_dict() for a in result.attempts],
        "plan": result.to_dict(),
    }


def _summary(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """The manifest's summary of its case rows, and whether the gate passed."""
    plans = [r for r in rows if r["expect"] == "plan"]
    refusals = [r for r in rows if r["expect"] == "refusal"]
    valid = [r for r in plans if r["passed"]]
    rate = len(valid) / len(plans) if plans else math.nan

    def count(rs: Sequence[dict[str, Any]], result: str) -> int:
        return sum(1 for r in rs if r["result"] == result)

    summary = {
        "plans": {
            "n": len(plans),
            "valid": len(valid),
            "valid_first_call": count(valid, "valid_first_call"),
            "valid_after_repair": count(valid, "valid_after_repair"),
            "valid_after_layout_fix": count(valid, "valid_after_layout_fix"),
            "failed": len(plans) - len(valid),
            "valid_rate": round(rate, 4) if plans else None,
        },
        "refusals": {
            "n": len(refusals),
            "refused_with_nearest": sum(1 for r in refusals if r["passed"]),
            "failed": sum(1 for r in refusals if not r["passed"]),
        },
        "all_ok": sum(1 for r in rows if r["passed"]),
        "n": len(rows),
    }
    passed = (not plans or rate >= GATE_VALID_RATE) and all(r["passed"] for r in refusals)
    return summary, passed


def _llm(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The calls sent (a call the run's spend cap refused is an attempt, never sent: ``refused_calls``)."""
    sent = [a for a in attempts if a["outcome"] != "spend_cap"]
    return {
        "calls": len(sent),
        "live_calls": sum(1 for a in sent if not a["cached"]),
        "refused_calls": len(attempts) - len(sent),
        "cost_usd": round(sum(a["cost_usd"] for a in attempts), 6),
        "prompt_tokens": sum(a["prompt_tokens"] for a in attempts),
        "completion_tokens": sum(a["completion_tokens"] for a in attempts),
    }


def superseded_cache(recorded: Path, rerun: int) -> Path:
    """Where re-run number ``rerun`` of a recorded run moved the cache entries of the attempts it
    superseded and sent again (the same request); the others stay in ``cache/`` (:func:`merge_rerun`)."""
    return recorded / "superseded" / f"rerun-{rerun}" / "cache"


def merge_rerun(recorded: Path, rerun: Path, reasons: Mapping[str, str]) -> dict[str, Any]:
    """Fold a live re-run of some cases (``rerun``: a gate run of those cases alone, with the same
    rule set and the same prompt or a later one) into the recorded run ``recorded``, and return the
    new manifest (written).

    Each re-run case replaces its row; the row it replaces stays in the manifest under ``reruns``
    with ``reasons[case_id]`` (why it was run again) and the re-run's number, so nothing recorded is
    dropped: the first run's cache entries and ledger lines stay, the re-run's are added after them,
    and ``llm`` counts every call in the ledger (``superseded_calls`` and ``superseded_cost_usd`` the
    replaced ones). Every row names the prompt it answered, ``prompts`` lists them all, and the
    budget is the runs' caps added up."""
    base = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    new = json.loads((rerun / "manifest.json").read_text(encoding="utf-8"))
    if new["mode"] != "live" or base["mode"] != "live":
        raise ValueError("only live runs are merged")
    if new.get("rules_version", "v1") != base.get("rules_version", "v1"):
        raise ValueError("the re-run used another rule set than the recorded run")
    used = dict(base.get("prompts") or {_row_version(base, {}): _prompt_of(base, _row_version(base, {}))})
    version = _row_version(new, {})
    prompt = {"tool": new["tool"], "template_hashes": new["template_hashes"]}
    if used.setdefault(version, prompt) != prompt:
        raise ValueError(f"the re-run used another prompt or tool than the recorded run's {version}")
    rows_new = {r["case_id"]: r for r in new["cases"]}
    ids = [r["case_id"] for r in base["cases"]]
    if set(rows_new) != set(reasons) or not set(rows_new) <= set(ids):
        raise ValueError("give a reason for every re-run case, each a case of the recorded run")
    number = 1 + max((x.get("rerun", 1) for x in base.get("reruns", [])), default=0)
    # a re-run case may send a request its superseded attempt sent (the same description and prompt):
    # that recorded entry moves to superseded/rerun-<n>/cache/ (n: this re-run's number), so every
    # answer stays on record
    replaced = {a["request_hash"] for r in base["cases"] if r["case_id"] in rows_new for a in r["attempts"]}
    for src in sorted((rerun / "cache").glob("??/*.json")):
        dst = recorded / "cache" / src.parent.name / src.name
        if dst.exists():
            if src.stem not in replaced:
                raise ValueError(f"the re-run repeats a recorded request ({src.stem[:12]}) of another case")
            keep = superseded_cache(recorded, number) / src.parent.name / src.name
            keep.parent.mkdir(parents=True, exist_ok=True)
            dst.replace(keep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    with (recorded / "ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write((rerun / "ledger.jsonl").read_text(encoding="utf-8"))

    def named(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
        return {**row, "prompt_version": _row_version(manifest, row), "truncated": _row_truncated(row)}

    reruns = [
        {**x, "rerun": x.get("rerun", 1), "superseded": named(x["superseded"], base)}
        for x in base.get("reruns", [])
    ]
    rows = []
    for r in base["cases"]:
        if r["case_id"] in rows_new:
            reruns.append(
                {
                    "case_id": r["case_id"],
                    "rerun": number,
                    "reason": reasons[r["case_id"]],
                    "superseded": named(r, base),
                }
            )
            rows.append(named(rows_new[r["case_id"]], new))
        else:
            rows.append(named(r, base))
    summary, passed = _summary(rows)
    old = [a for x in reruns for a in x["superseded"]["attempts"]]
    llm = _llm([*[a for r in rows for a in r["attempts"]], *old])
    llm |= {
        "superseded_calls": _llm(old)["calls"],
        "superseded_cost_usd": round(sum(a["cost_usd"] for a in old), 6),
    }
    budgets = [b for b in (base.get("budget_usd"), new.get("budget_usd")) if b is not None]
    manifest = {
        **base,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "budget_usd": round(sum(budgets), 6) if budgets else None,
        "prompts": used,
        "rules_revision": RULES_REVISION[str(base.get("rules_version", "v1"))],
        "gate": {"valid_rate_needed": GATE_VALID_RATE, "passed": passed},
        "summary": summary,
        "truncated": _truncated(rows),
        "llm": llm,
        "cases": rows,
        "reruns": reruns,
    }
    if "drafts" in base:  # every call's spend: the ledger and the draft recordings' kept lines
        ledgers = (recorded / "ledger.jsonl", recorded / base["drafts"]["ledger"])
        manifest["llm_total_usd"] = round(sum(e.cost_usd for path in ledgers for e in read_ledger(path)), 6)
    (recorded / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def gate_manifest(
    results: Sequence[tuple[UseCase, PlanResult]],
    *,
    mode: str,
    budget_usd: float | None = None,
    compact: bool = COMPACT_TOOL,
    recorded: Path | None = None,
    version: str = prompts.CURRENT,
) -> dict[str, Any]:
    """The gate's manifest (module docstring); ``mode`` is ``live`` or ``replay``. A replay of a
    recorded run from its responses (``recorded``, :func:`replay_recorded`) names that run's prompt
    and tool, not the current ones."""
    rows = [case_row(c, r) for c, r in results]
    attempts = [a for r in rows for a in r["attempts"]]
    tool = engineer_tool(compact=compact, version=version)
    summary, passed = _summary(rows)
    return {
        "format": "rtt-studio-engineer-gate",
        "version": 1,
        "label": PRODUCT_MODE_LABEL,
        "contract": CONTRACT_VERSION,
        "model": ENGINEER_MODEL,
        "sampling": dict(ENGINEER_SAMPLING),
        "mode": mode,
        "budget_usd": budget_usd,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "prompt_version": version,
        "rules_version": prompts.rules_of(version),
        "rules_revision": RULES_REVISION[prompts.rules_of(version)],
        "tool": {
            "compact": compact,
            "sha256": tool_sha256(compact=compact, version=version),
            "bytes": len(wire_json(tool)),
        },
        "template_hashes": prompts.template_hashes(version),
        **_recorded_prompt(recorded),
        "gate": {"valid_rate_needed": GATE_VALID_RATE, "passed": passed},
        "summary": summary,
        "truncated": _truncated(rows),
        "llm": _llm(attempts),
        "cases": rows,
    }
