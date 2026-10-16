"""Deterministic stand-ins for the agent's model and physics, for tests and offline dry runs.

* :class:`FakePartBackend` implements :class:`rtt.agent.backends.PartBackend` without
  MuJoCo: the part is measured with ``trimesh`` and every design gets a hash-seeded
  synthetic score. Jobs can be held open with :attr:`FakePartBackend.gate` to test
  polling and the concurrency cap.
* :class:`ScriptedAgentTransport` implements the LLM transport protocol. It plays a
  fixed, sensible Grasp Engineer policy (inspect, propose, simulate, poll, fetch,
  stress-test, export, ledger, answer) from the conversation it is sent, answers the
  proposer's forced ``submit_designs`` call with valid designs, and answers Lightning
  summary requests with a templated sentence. Usage is reported, so the spend guard and
  the ledger are exercised with real prices.

Nothing produced with these is a result: sessions recorded with the scripted transport
name ``scripted-offline`` as their model.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from rtt.agent.backends import OPEN_MARGIN_MM, PartInfo, PartNotFoundError, Units
from rtt.design.schema import FingerDesign, GripperTask
from rtt.llm.providers import NEMOTRON_LIGHTNING, TransportResponse
from rtt.search.base import DesignScore, EpisodeOutcome

__all__ = ["SCRIPTED_DESIGNS", "FakePartBackend", "ScriptedAgentTransport"]


# --------------------------------------------------------------------------- physics


class FakePartBackend:
    """Synthetic physics with the real backend's interface (see the module docstring).

    Args:
        root: Per-part folder root (only ``print_pack`` writes files).
        delay_s: Seconds every ``evaluate`` sleeps (to exercise polling).
        settings: Grasp heights per design; ``seeds``: episodes per height.
    """

    name = "fake"

    def __init__(self, root: str | Path, *, delay_s: float = 0.0, settings: int = 3, seeds: int = 3) -> None:
        self.root = Path(root)
        self.delay_s = delay_s
        self.settings = settings
        self.seeds = seeds
        self.gate = threading.Event()
        self.gate.set()
        self.parts: dict[str, PartInfo] = {}
        self.evaluated: list[tuple[str, tuple[str, ...]]] = []
        self.certified: list[str] = []

    def intake(
        self, stl_path: Path, *, part_id: str, name: str, units: Units = "mm", density: float | None = None
    ) -> PartInfo:
        import trimesh

        mesh = trimesh.load(Path(stl_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("no triangle mesh found")
        scale = 1.0 if units == "mm" else 1000.0
        ext = sorted((float(v) * scale for v in mesh.extents[:2]), reverse=True)
        size = (round(ext[0], 2), round(ext[1], 2), round(float(mesh.extents[2]) * scale, 2))
        density = density or 1200.0
        volume_cm3 = abs(float(mesh.volume)) * scale**3 / 1000.0
        info = PartInfo(
            part_id=part_id,
            name=name,
            source_sha256=hashlib.sha256(Path(stl_path).read_bytes()).hexdigest(),
            size_mm=size,
            grip_width_mm=size[1],
            height_mm=size[2],
            volume_cm3=round(volume_cm3, 3),
            mass_g=round(volume_cm3 * density / 1000.0, 2),
            density_kg_m3=density,
            max_opening_mm=84.0,
            fits_gripper=size[1] + OPEN_MARGIN_MM <= 84.0,
            n_convex_pieces=1,
            stable_pose_probability=1.0,
            conditions=("fake: part centred",),
            episodes_per_design=self.settings * self.seeds,
            backend=self.name,
        )
        self.parts[part_id] = info
        return info

    def info(self, part_id: str) -> PartInfo:
        if part_id not in self.parts:
            raise PartNotFoundError(part_id)
        return self.parts[part_id]

    def gripper_task(self, part_id: str) -> GripperTask:
        info = self.info(part_id)
        return GripperTask(widest_training_object_mm=max(info.grip_width_mm, 1.0))

    def _p(self, design: FingerDesign, salt: str) -> float:
        digest = hashlib.sha256(f"{design.design_id()}:{salt}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 2**32

    def evaluate(self, part_id: str, designs: Sequence[FingerDesign]) -> list[DesignScore]:
        self.info(part_id)
        self.gate.wait()
        if self.delay_s:
            time.sleep(self.delay_s)
        self.evaluated.append((part_id, tuple(d.design_id() for d in designs)))
        scores = []
        for design in designs:
            base = 0.35 + 0.6 * self._p(design, "base")
            episodes = []
            for k in range(self.settings):
                for s in range(self.seeds):
                    ok = self._p(design, f"{k}:{s}") < base - 0.1 * abs(k - 1)
                    episodes.append(
                        EpisodeOutcome(
                            condition_id=f"{part_id}/centred",
                            family="mesh",
                            controller_setting=k,
                            seed=s,
                            success=ok,
                            failure_mode=None if ok else ("slip" if s % 2 else "no_lift"),
                            max_lift_m=0.1 if ok else 0.02,
                            hold_frac=1.0 if ok else 0.1,
                        )
                    )
            scores.append(DesignScore.from_episodes(design.design_id(), 20.0, episodes, self.settings))
        return scores

    def certify(
        self, part_id: str, design: FingerDesign, *, score: DesignScore | None = None
    ) -> dict[str, Any]:
        self.info(part_id)
        self.certified.append(design.design_id())
        names = ["nominal"] + [f"offset_{a}_{s}{e}mm" for a in "xy" for e in (1, 2, 3) for s in "+-"]
        names += ["shake_x", "shake_y", "shake_z", "push_x", "push_y"]
        p = self._p(design, "cert")
        tests = []
        for i, name in enumerate(names):
            ok = name == "nominal" or self._p(design, name) < 0.5 + 0.5 * p - 0.02 * i
            category = "placement" if name == "nominal" or name.startswith("offset") else name.split("_")[0]
            tests.append(
                {"name": name, "category": category, "success": ok, "failure_mode": None if ok else "slip"}
            )
        cats: dict[str, list[bool]] = {}
        for t in tests:
            cats.setdefault(t["category"], []).append(t["success"])
        scores = {k: sum(v) / len(v) for k, v in cats.items()}
        return {
            "design_id": design.design_id(),
            "object": part_id,
            "score": sum(scores.values()) / len(scores),
            "category_scores": scores,
            "tolerance_mm": {"x": 2.0, "y": 1.0},
            "passed": f"{sum(t['success'] for t in tests)}/{len(tests)}",
            "tests": tests,
        }

    def print_pack(
        self,
        part_id: str,
        design: FingerDesign,
        out_dir: Path,
        *,
        results: Mapping[str, Any] | None = None,
        certificate: Mapping[str, Any] | None = None,
    ) -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for side in ("left", "right"):
            path = out / f"finger_{side}.stl"
            path.write_text(f"solid fake_{side}\nendsolid fake_{side}\n", encoding="ascii")
            files.append(path)
        design_path = out / "design.json"
        design_path.write_text(
            json.dumps({"design_id": design.design_id(), **design.model_dump()}), encoding="utf-8"
        )
        files.append(design_path)
        if certificate is not None:
            cert_path = out / "certificate.json"
            cert_path.write_text(json.dumps(dict(certificate)), encoding="utf-8")
            files.append(cert_path)
        return files


# --------------------------------------------------------------------------- model

#: Valid, mutually distinct designs the scripted proposer hands out (in order, cycling).
SCRIPTED_DESIGNS: Final[tuple[dict[str, Any], ...]] = (
    {
        "finger_length_mm": 70.0,
        "finger_thickness_mm": 10.0,
        "finger_width_mm": 22.0,
        "bend_angle_deg": 0.0,
        "bend_start_frac": 0.5,
        "inner_face": "flat",
        "inner_face_depth_mm": 0.0,
        "tip_profile": "rounded",
        "tip_size_mm": 3.0,
        "taper_ratio": 0.8,
    },
    {
        "finger_length_mm": 75.0,
        "finger_thickness_mm": 12.0,
        "finger_width_mm": 24.0,
        "bend_angle_deg": 0.0,
        "bend_start_frac": 0.5,
        "inner_face": "v_groove",
        "inner_face_depth_mm": 4.0,
        "tip_profile": "chamfer",
        "tip_size_mm": 3.0,
        "taper_ratio": 0.9,
    },
    {
        "finger_length_mm": 60.0,
        "finger_thickness_mm": 10.0,
        "finger_width_mm": 26.0,
        "bend_angle_deg": 0.0,
        "bend_start_frac": 0.6,
        "inner_face": "concave_arc",
        "inner_face_depth_mm": 3.0,
        "tip_profile": "rounded",
        "tip_size_mm": 2.0,
        "taper_ratio": 1.0,
    },
    {
        "finger_length_mm": 85.0,
        "finger_thickness_mm": 11.0,
        "finger_width_mm": 18.0,
        "bend_angle_deg": 0.0,
        "bend_start_frac": 0.5,
        "inner_face": "v_groove",
        "inner_face_depth_mm": 3.0,
        "tip_profile": "hook",
        "tip_size_mm": 4.0,
        "taper_ratio": 1.0,
    },
)


def _response(
    message: dict[str, Any], model: str, prompt_tokens: int, completion_tokens: int
) -> TransportResponse:
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    return TransportResponse(
        status_code=200,
        headers={},
        body={
            "id": "scripted",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )


class ScriptedAgentTransport:
    """A :class:`rtt.llm.providers.Transport` that plays the Grasp Engineer offline.

    Args:
        proposals: Designs handed out by the scripted proposer (default
            :data:`SCRIPTED_DESIGNS`).
        n_designs: Designs the scripted agent asks for per proposal.
        target: Simulation target the scripted agent launches on.
        bad_first_call: Make the agent's first tool call carry invalid arguments.
    """

    def __init__(
        self,
        *,
        proposals: Sequence[Mapping[str, Any]] = SCRIPTED_DESIGNS,
        n_designs: int = 3,
        target: str = "local",
        bad_first_call: bool = False,
    ) -> None:
        self.proposals = [dict(p) for p in proposals]
        self.n_designs = n_designs
        self.target = target
        self.bad_first_call = bad_first_call
        self.payloads: list[dict[str, Any]] = []
        self._next = 0
        self._calls = 0
        self._lock = threading.Lock()

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        with self._lock:
            self.payloads.append(dict(payload))
            self._calls += 1
            call_n = self._calls
        prompt_tokens = len(json.dumps(payload.get("messages", []))) // 4 + 200
        model = str(payload.get("model"))
        if model == NEMOTRON_LIGHTNING or "tools" not in payload:
            text = "Scripted summary: the simulator's numbers are in the step above."
            return _response({"role": "assistant", "content": text}, model, prompt_tokens, 30)
        choice = payload.get("tool_choice")
        if isinstance(choice, Mapping):  # the proposer's forced submit_designs call
            return _response(self._propose(payload), model, prompt_tokens, 400)
        message = self._agent_turn(payload, call_n)
        return _response(message, model, prompt_tokens, 120)

    # -- proposer

    def _propose(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        tool = payload["tools"][0]["function"]
        slots = [k for k in tool["parameters"]["properties"] if k != "analysis"]
        args: dict[str, Any] = {"analysis": "Scripted analysis: vary the contact face and the tip."}
        with self._lock:
            for slot in slots:
                design = dict(self.proposals[self._next % len(self.proposals)])
                self._next += 1
                args[slot] = {"hypothesis": f"Scripted idea {self._next}.", **design}
        call = {
            "id": "call_propose",
            "type": "function",
            "function": {"name": "submit_designs", "arguments": json.dumps(args)},
        }
        return {"role": "assistant", "content": None, "tool_calls": [call]}

    # -- agent policy

    def _agent_turn(self, payload: Mapping[str, Any], call_n: int) -> dict[str, Any]:
        messages = list(payload.get("messages", []))
        part_id = ""
        for m in messages:
            if (
                m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].startswith("Part: ")
            ):
                part_id = m["content"].split("\n", 1)[0].removeprefix("Part: ").strip()
        names: dict[str, str] = {}
        results: list[tuple[str, dict[str, Any]]] = []
        for m in messages:
            if m.get("role") == "assistant":
                for c in m.get("tool_calls") or []:
                    names[c["id"]] = c["function"]["name"]
            elif m.get("role") == "tool":
                results.append((names.get(m.get("tool_call_id", ""), "?"), json.loads(m["content"])))
        if payload.get("tool_choice") == "none":
            return {"role": "assistant", "content": self._final(results)}
        if not results:
            if self.bad_first_call:
                return self._call("inspect_part", {"part": part_id}, call_n)
            return self._call("inspect_part", {"part_id": part_id}, call_n)
        last_name, last = results[-1]
        if not last.get("ok"):
            if last_name == "inspect_part" and last.get("code") == "invalid_args":
                return self._call("inspect_part", {"part_id": part_id}, call_n)
            return {"role": "assistant", "content": f"Stopping: {last_name} failed ({last.get('error')})."}
        result = last["result"]
        if last_name == "inspect_part":
            return self._call("propose_designs", {"part_id": part_id, "n": self.n_designs}, call_n)
        if last_name == "propose_designs":
            designs = [d["design_id"] for d in result["designs"]]
            for name, res in results:
                if name == "inspect_part" and res.get("ok"):
                    valid = [b["design_id"] for b in res["result"]["baselines"] if b["valid_for_part"]]
                    designs += valid[:1]
            return self._call(
                "launch_sim_job", {"part_id": part_id, "designs": designs, "target": self.target}, call_n
            )
        if last_name in ("launch_sim_job", "poll_job"):
            if result["state"] == "COMPLETED":
                return self._call("fetch_results", {"job_id": result["job_id"]}, call_n)
            if result["state"] in ("FAILED", "CANCELLED"):
                return {"role": "assistant", "content": f"The simulation job ended {result['state']}."}
            return self._call("poll_job", {"job_id": result["job_id"]}, call_n)
        if last_name == "fetch_results":
            return self._call(
                "stress_test", {"design_id": result["best_design_id"], "part_id": part_id}, call_n
            )
        if last_name == "stress_test":
            run_id = next(r["result"]["run_id"] for n, r in reversed(results) if n == "fetch_results")
            return self._call("export_print_pack", {"run_id": run_id}, call_n)
        if last_name == "export_print_pack":
            return self._call("cost_ledger", {}, call_n)
        return {"role": "assistant", "content": self._final(results)}

    @staticmethod
    def _call(name: str, args: Mapping[str, Any], n: int) -> dict[str, Any]:
        call = {
            "id": f"call_{n}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        return {"role": "assistant", "content": "", "tool_calls": [call]}

    @staticmethod
    def _final(results: Sequence[tuple[str, dict[str, Any]]]) -> str:
        fetched = next(
            (r["result"] for n, r in reversed(results) if n == "fetch_results" and r.get("ok")), None
        )
        pack = next(
            (r["result"] for n, r in reversed(results) if n == "export_print_pack" and r.get("ok")), None
        )
        if fetched is None:
            return "No design was simulated in this session."
        best = next(d for d in fetched["designs"] if d["design_id"] == fetched["best_design_id"])
        text = f"Recommended design {best['design_id']}: {best['success_pct']}% simulated success ({best['verdict']})."
        if pack is not None:
            text += f" Print pack {pack['pack_id']}."
        return text
