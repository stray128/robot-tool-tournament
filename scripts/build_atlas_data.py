#!/usr/bin/env python
"""Build the use-case atlas of the site (``web/public/data/atlas.json``) from the research catalogue.

::

    python scripts/build_atlas_data.py --research DIR [--keys FILE] [--out web/public/data/atlas.json]
        [--tokens tests/data/atlas_brand_tokens.json] [--cache runs/atlas] [--spend-cap 4.5]
        [--offline] [--tokens-only]

``DIR`` holds the research catalogue: ``usecases_construction.json``, ``usecases_manufacturing.json``
(factories, warehouses and retail), ``usecases_lab_food_service.json`` (labs, food and services) and
``effector_rules.json`` (the integrator rules). The catalogue is not part of this repository and is
never copied into it: its prose names companies and products, and the site shows none.

What the script does, in order (nothing is written when a step fails):

1. **Names to keep off screen.** Every company and product name in the catalogue's
   ``existing_products`` (names, makers and the trade names in brackets) goes to ``--tokens``, a
   committed list the site's tests screen with too. Ordinary English words among them (a robot
   called by a common noun) are matched only with their capital letter; the rest in any case.
2. **Brand-free rewrite.** Each record's title, step list (verb, object, a short note), tool names
   and "why" are rewritten in plain, brand-free words by Claude (``claude-opus-5-5``, low effort,
   JSON output); so are the integrator rules. Answers are cached under ``--cache`` (git-ignored
   ``runs/``) by a hash of the prompt and the record, so a rerun spends nothing. The API key is read
   at run time from ``--keys`` (key ``claude``) and never printed or written. Spend is counted from
   each response's usage and the run stops before it could pass ``--spend-cap`` dollars.
3. **Screen.** Every string the atlas shows is checked with
   :func:`rtt.effectors.types.find_brand_tokens`, :func:`rtt.library.brands.find_library_brands`
   and the names of step 1. A record whose rewrite still names one is rewritten once more with the
   names it must drop; any remaining hit fails the build. URLs are kept only as link targets
   (fields named ``url``), never as text.

Six records are the Studio's worked examples (:data:`HERO_RECORDS`); the site links each one to its
story page when the Studio's data lists that hero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rtt.effectors.types import find_brand_tokens
from rtt.library.brands import find_library_brands

SCHEMA: Final = "rtt-web-atlas"
TOKENS_SCHEMA: Final = "rtt-atlas-brand-tokens"
MODEL: Final = "claude-opus-5-5"
#: Bump when the prompt or the output shape changes (invalidates the cache).
PROMPT_VERSION: Final = "atlas-rewrite/3"
BATCH: Final = 6
DOMAINS: Final[dict[str, tuple[str, str]]] = {
    # file stem -> (domain id, title)
    "usecases_construction": ("construction", "Construction"),
    "usecases_manufacturing": ("manufacturing", "Factories, warehouses and retail"),
    "usecases_lab_food_service": ("lab_food_service", "Labs, food and services"),
}
KINDS: Final = ("single", "tool_change", "two_tool_head")
FEASIBILITY: Final = ("rigid", "approximation", "out_of_scope")
#: The Studio's worked examples (hero case id) and the research record each one simulates. "same":
#: the record is the hero's job; "related": the nearest researched job (the hero is a variant).
HERO_RECORDS: Final[dict[str, tuple[str, str]]] = {
    "h1_tote": ("WH-07", "related"),
    "h2_coupling": ("con_pipe_conduit_install", "related"),
    "h4_cnc": ("MF-02", "same"),
    "h5_solar": ("con_solar_module_install", "same"),
    "h6_tile": ("con_floor_tile_setting", "same"),
    "h7_hotel": ("hospitality-room-delivery", "same"),
}
#: USD per million tokens: input, output, cache write, cache read. Unknown models are priced at the
#: highest row (a server-side fallback may answer with another model).
PRICES: Final[dict[str, tuple[float, float, float, float]]] = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 6.25, 0.50),
}
_URL: Final = re.compile(r"https?://|www\.", re.IGNORECASE)
_EM_DASH: Final = re.compile("[—–]")


class AtlasError(RuntimeError):
    """A check failed; nothing was written."""


# --------------------------------------------------------------------------- research input


@dataclass(frozen=True)
class Record:
    id: str
    domain: str
    setting: str
    title: str
    workflow: str
    subtasks: list[Any]
    kind: str
    tools: list[str]
    why: str
    feasibility: str
    demo_score: int
    sources: list[str]
    products: list[Mapping[str, Any]] = field(default_factory=list)
    #: Every other prose field (notes, alternatives, value and simulation notes), for the name search.
    prose: list[str] = field(default_factory=list)

    def prompt_view(self) -> dict[str, Any]:
        """What the rewrite sees: the job, its steps and the setup with its reasons."""
        return {
            "id": self.id,
            "setting": self.setting,
            "title": self.title,
            "human_workflow": self.workflow,
            "subtasks": self.subtasks,
            "recommended_setup": {"kind": self.kind, "tools": self.tools, "why": self.why},
        }


@dataclass(frozen=True)
class Rule:
    id: str
    condition: str
    recommendation: str
    rationale: str
    sources: list[str]

    def prompt_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "condition": self.condition,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


def _source_urls(raw: Any) -> list[str]:
    out: list[str] = []
    for s in raw or []:
        url = s if isinstance(s, str) else (s.get("url") if isinstance(s, Mapping) else None)
        if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in out:
            out.append(url)
    return out


def _prose(r: Mapping[str, Any]) -> list[str]:
    """Every string of a record outside its sources and products (URLs removed)."""
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            text = re.sub(r"https?://\S+", " ", v).strip()
            if text:
                out.append(text)
        elif isinstance(v, Mapping):
            for k, x in v.items():
                if k not in ("sources", "value_sources", "existing_products", "url", "source"):
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk({k: v for k, v in r.items() if k not in ("id", "setting", "scores", "rank_factors")})
    return out


def _records_of(stem: str, data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping) and isinstance(data.get("records"), list):
        return data["records"]
    raise AtlasError(f"{stem}.json: expected a list of records or an object with 'records'")


def load_research(research: Path) -> tuple[list[Record], list[Rule]]:
    records: list[Record] = []
    for stem, (domain, _title) in DOMAINS.items():
        file = research / f"{stem}.json"
        if not file.is_file():
            raise AtlasError(f"missing {file.name} in the research directory")
        for r in _records_of(stem, json.loads(file.read_text(encoding="utf-8"))):
            setup = r.get("recommended_setup") or {}
            kind = setup.get("kind")
            feas = r.get("sim_feasibility")
            if kind not in KINDS or feas not in FEASIBILITY:
                raise AtlasError(
                    f"{stem}.json {r.get('id')}: unknown setup kind {kind!r} or feasibility {feas!r}"
                )
            score = r.get("demo_score")
            if not isinstance(score, int) or not 1 <= score <= 5:
                raise AtlasError(f"{stem}.json {r.get('id')}: demo_score must be 1 to 5")
            records.append(
                Record(
                    id=str(r["id"]),
                    domain=domain,
                    setting=str(r["setting"]),
                    title=str(r["title"]),
                    workflow=str(r.get("human_workflow") or ""),
                    subtasks=list(r.get("subtasks") or []),
                    kind=kind,
                    tools=[str(t) for t in setup.get("tools") or []],
                    why=str(setup.get("why") or ""),
                    feasibility=feas,
                    demo_score=score,
                    sources=_source_urls(r.get("sources")),
                    products=[p for p in r.get("existing_products") or [] if isinstance(p, Mapping)],
                    prose=_prose(r),
                )
            )
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise AtlasError("record ids repeat across the catalogue")
    for case, (rid, _match) in HERO_RECORDS.items():
        if rid not in ids:
            raise AtlasError(f"hero {case} names record {rid}, which the catalogue does not have")
    rules_file = research / "effector_rules.json"
    if not rules_file.is_file():
        raise AtlasError("missing effector_rules.json in the research directory")
    rules = [
        Rule(
            id=str(r["id"]),
            condition=str(r.get("condition") or ""),
            recommendation=str(r.get("recommendation") or ""),
            rationale=str(r.get("rationale") or ""),
            sources=_source_urls(r.get("sources")),
        )
        for r in json.loads(rules_file.read_text(encoding="utf-8"))
    ]
    return records, rules


# --------------------------------------------------------------------------- names to keep off screen

#: A bracketed or listed fragment that is a citation detail, not a name.
_NOT_A_NAME: Final = re.compile(
    r"^(patent( application)?( [a-z]{2}[0-9]+)?|us ?[0-9][0-9,]*|us patent [0-9]+|chinese patent family|"
    r"formerly|via|with|now|per|and|e\.g\.,?|-|[0-9 ,.]+|[0-9]{4}|.* et al\.?( [0-9]{4})?|"
    r"(us|uk|eu|japan|china|singapore|switzerland|france|netherlands|denmark|czechia|australia|us/israel|"
    r"germany|italy|korea|canada)( [0-9]{4})?)$",
    re.IGNORECASE,
)
_COUNTRY_TAIL: Final = re.compile(
    r"\s*\((?:US|UK|EU|Japan|China|Singapore|Switzerland|France|Netherlands|Denmark|Czechia|Australia|"
    r"US/Israel|Germany|Italy|Korea|Canada)[^)]*\)\s*$"
)
#: Trade words the word list lacks; a phrase made only of these and ordinary words describes a thing.
TRADE_WORDS: Final = frozenset(
    [
        "gripper",
        "grippers",
        "benchtop",
        "decapper",
        "recapper",
        "palletizer",
        "palletizers",
        "palletiser",
        "palletisers",
        "depalletizing",
        "depalletising",
        "cobot",
        "cobots",
        "scara",
        "amr",
        "amrs",
        "agv",
        "agvs",
        "transplanter",
        "transplanters",
        "dishwashing",
        "cartoner",
        "eoat",
        "eoats",
        "multipack",
        "mobile",
        "cnc",
        "pcb",
        "tht",
        "mep",
        "cmu",
        "aac",
        "atex",
        "cssd",
        "iv",
        "fdm",
        "ai",
    ]
)
#: Fragments the rules above keep that are descriptions, not names, on review (lower case).
NOT_NAMES: Final = frozenset(
    {
        "a pharmacy informaticist",
        "two cobots in a hospital lab",
        "2026 study",
        "humanoid",
        "transplanter",
        "transplanters",
        "actuators",
        "chemo compounding robot",
        "centralised robotic dishwashing",
        "cdpr for panelised envelope retrofits",
        "robot cafe kiosk",
        "pharma pick",
        "ppe",
        "quick changer",
        "shoebox picker",
        "etics",
        # this project's own names and open tools, and plain words a source used as a name
        "effector lab",
        "studio",
        "mujoco",
        "apache-2.0",
        "arxiv",
        "pmc",
        "factory",
        "sander",
        "screwdriver",
        "shadow",
        "chef",
        "soft gripper",
    }
)


def _lowercase_words(path: Path) -> set[str]:
    """Ordinary words: the word list's entries written in lower case (proper nouns excluded)."""
    if not path.is_file():
        raise AtlasError(f"no word list at {path} (needed to tell names from ordinary words)")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return {w.strip() for w in lines if w.strip() and w.strip() == w.strip().lower()}


def _ordinary(token: str, words: set[str]) -> bool:
    t = token.lower().strip("'")
    for cand in (t, t.removesuffix("s"), t.removesuffix("es"), t.removesuffix("'s")):
        if cand in words or cand in TRADE_WORDS:
            return True
    return False


def _tokens(name: str) -> list[str]:
    return re.findall(r"[^\s/-]+", name)


def _all_ordinary(name: str, words: set[str]) -> bool:
    toks = _tokens(name)
    if not toks or any(re.search(r"[0-9]", t) for t in toks):
        return False
    letters = [re.sub(r"[^A-Za-z']", "", t) for t in toks]
    return all(t and _ordinary(t, words) for t in letters)


def _title_case(name: str) -> bool:
    small = {"of", "and", "for", "the", "a", "an", "to", "in", "on", "with"}
    return all(t[0].isupper() or not t[0].isalpha() or t.lower() in small for t in _tokens(name))


def _fragments(text: str) -> list[str]:
    """Candidate names in a product or maker string: the outer name, the bracketed names and the
    parts of lists joined by ',', '/', ';', ' and ' or ' with '."""
    out: list[str] = []
    text = _COUNTRY_TAIL.sub("", text.strip())
    inner = re.findall(r"\(([^()]*)\)", text)
    outer = re.sub(r"\([^()]*\)", " ", text)
    for chunk in [outer, *inner]:
        chunk = re.sub(r"\b(formerly|now|via|per|e\.g\.,?|including)\b", ",", chunk, flags=re.IGNORECASE)
        for part in re.split(r",|/|;| and | with | plus | on a | on |\+(?=\s)", chunk):
            part = re.sub(r"\s+", " ", part).strip(" .:-'\"")
            if part and not _NOT_A_NAME.match(part):
                out.append(part)
    return out


def classify_names(candidates: Iterable[str], words: set[str]) -> dict[str, list[str]]:
    """Split candidate names into ``ci`` (match in any case) and ``cs`` (match as written).

    Dropped: fragments without letters, ones that start with a lower-case word, and descriptions
    (ordinary words not in title case: "Hospital robot", "Suction cup tool"). A name made only of
    ordinary words in title case ("Origin", "Construction Robotics") is matched as written, so the
    same words in a sentence do not count; every other name ("Hadrian X", "Boston Dynamics") is
    matched in any case.
    """
    ci: set[str] = set()
    cs: set[str] = set()
    for raw in candidates:
        n = re.sub(r"\s+", " ", raw).strip(" .,:;'\"")
        if len(n) < 2 or not re.search(r"[A-Za-z]", n) or n.lower() in NOT_NAMES or _NOT_A_NAME.match(n):
            continue
        if n[0].islower():
            continue
        if _all_ordinary(n, words):
            if _title_case(n) and len(_tokens(n)) <= 5:
                cs.add(n)
            continue
        ci.add(n.lower())
    cs -= {c for c in cs if c.lower() in ci}
    return {"ci": sorted(ci), "cs": sorted(cs)}


def product_strings(records: Sequence[Record]) -> list[str]:
    """Every name and maker string of the records' ``existing_products``."""
    out: list[str] = []
    for rec in records:
        for p in rec.products:
            for key in ("name", "maker"):
                value = p.get(key)
                if isinstance(value, str) and value.strip() and value.strip() not in out:
                    out.append(value.strip())
    return out


def rule_based_names(records: Sequence[Record]) -> list[str]:
    return [frag for s in product_strings(records) for frag in _fragments(s)]


NAMES_SYSTEM: Final = """You list proper names in short catalogue entries about robots. For each \
entry return every company, brand, product, robot, model, trade name, project, programme, research \
institute, university or person it names, each as it is written (keep capitals). Split combined names: \
"Acme Robotics FooBot 3" gives "Acme Robotics", "FooBot", "FooBot 3". Return nothing for generic \
descriptions ("hospital robot", "suction cup tool"). Do not invent names."""

_NAMES_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    "required": ["names"],
    "additionalProperties": False,
}


def extract_brand_names(
    records: Sequence[Record], rules: Sequence[Rule], words: set[str], rw: Rewriter | None
) -> dict[str, list[str]]:
    """Names to keep off screen: the product and maker strings split by rule, plus (with ``rw``)
    the names Claude finds in those strings and in every other prose field of the catalogue."""
    candidates = rule_based_names(records)
    if rw is not None:
        strings = product_strings(records)
        for i in range(0, len(strings), 120):
            chunk = strings[i : i + 120]
            data = rw.ask(
                "names",
                _NAMES_SCHEMA,
                [{"id": str(i + j), "text": s} for j, s in enumerate(chunk)],
                (),
                system=NAMES_SYSTEM,
            )
            candidates += [n for n in data.get("names", []) if isinstance(n, str)]
        prose = [{"id": r.id, "text": " | ".join(r.prose)} for r in records] + [
            {"id": r.id, "text": f"{r.condition} | {r.recommendation} | {r.rationale}"} for r in rules
        ]
        for i in range(0, len(prose), 12):
            data = rw.ask("names", _NAMES_SCHEMA, prose[i : i + 12], (), system=NAMES_SYSTEM)
            candidates += [n for n in data.get("names", []) if isinstance(n, str)]
    return classify_names(candidates, words)


class Screen:
    """Every brand rule the atlas must pass, as one ``hits(text)`` call."""

    def __init__(self, tokens: Mapping[str, Sequence[str]]) -> None:
        def alt(items: Iterable[str]) -> str:
            return "|".join(re.escape(t) for t in sorted(set(items), key=len, reverse=True))

        ci = list(tokens.get("ci", []))
        cs = list(tokens.get("cs", []))
        self._ci = re.compile(rf"(?<![A-Za-z0-9])({alt(ci)})(?![A-Za-z0-9])", re.IGNORECASE) if ci else None
        self._cs = re.compile(rf"(?<![A-Za-z0-9])({alt(cs)})(?![A-Za-z0-9])") if cs else None

    def hits(self, text: str) -> list[str]:
        found = [*find_brand_tokens(text), *find_library_brands(text)]
        if self._ci:
            found += [m.group(1).lower() for m in self._ci.finditer(text)]
        if self._cs:
            found += [m.group(1) for m in self._cs.finditer(text)]
        return sorted(set(found))


# --------------------------------------------------------------------------- the rewrite

SYSTEM: Final = """You rewrite entries of a robotics use-case catalogue for a public website that must \
name no brand. The website helps automation engineers choose end effectors (grippers, suction, process \
tools) for a robot arm.

Rules for every string you write:
- Name no company, product, robot model, trade name, project, programme, university, lab, person, \
patent or place. Describe them generically instead ("one deployed system", "a research prototype", \
"production machines").
- Keep the engineering content: what the job is, what each step needs, why the setup fits.
- Plain words, active voice, short sentences, British spelling. No em dashes or en dashes; use commas \
or full stops. No marketing words. No URLs.
- Add nothing that the input does not say. Never invent numbers. In "why" you may keep at most one \
figure from the input when it is a measured or published result, attributed generically.

Fields:
- title: the job in a few words, sentence case, at most 70 characters, same meaning as the input title.
- subtasks: the input's steps in the same order (split a plain-text step into verb and object). \
verb: one or two lower-case words. object: a short noun phrase, lower case unless a unit or acronym. \
note: one short sentence from the input's note, or an empty string.
- tools: one short generic name per tool of the recommended setup, sentence case, at most 60 characters \
(for example "Vacuum frame, 4 to 6 cups"). Translate snake_case vocabulary keys into words.
- why: one or two sentences, at most 320 characters, saying why an experienced integrator would pick \
this setup for this job.
- For rules: title is a short name of the rule (at most 60 characters); text is one or two sentences \
(at most 320 characters) stating when it applies and what to pick, brand-free."""

_RECORD_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "verb": {"type": "string"},
                                "object": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["verb", "object", "note"],
                            "additionalProperties": False,
                        },
                    },
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "why": {"type": "string"},
                },
                "required": ["id", "title", "subtasks", "tools", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}
_RULE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["id", "title", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rules"],
    "additionalProperties": False,
}


def _key(kind: str, view: Mapping[str, Any], avoid: Sequence[str] = ()) -> str:
    blob = json.dumps(
        {"v": PROMPT_VERSION, "kind": kind, "view": view, "avoid": sorted(avoid)}, sort_keys=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Rewriter:
    """Claude calls with a disk cache and a dollar cap; the key never leaves this object."""

    def __init__(self, cache_dir: Path, keys: Path | None, spend_cap: float, offline: bool) -> None:
        self.cache_file = cache_dir / "rewrites.json"
        self.ledger = cache_dir / "spend.jsonl"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, Any] = (
            json.loads(self.cache_file.read_text(encoding="utf-8")) if self.cache_file.is_file() else {}
        )
        self.keys = keys
        self.cap = spend_cap
        self.offline = offline
        self.spent = 0.0
        self.calls = 0
        self.usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        self._client: Any = None
        self._worst_call = 0.6  # dollars; raised to the most expensive call seen

    def _client_or_fail(self) -> Any:
        if self._client is not None:
            return self._client
        if self.offline:
            raise AtlasError("--offline: a rewrite is missing from the cache")
        if self.keys is None or not self.keys.is_file():
            raise AtlasError("no API key file (pass --keys)")
        try:
            import anthropic
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise AtlasError("the anthropic package is not installed (pip install anthropic)") from err
        key = json.loads(self.keys.read_text(encoding="utf-8")).get("claude")
        if not isinstance(key, str) or not key:
            raise AtlasError("the key file has no 'claude' entry")
        self._client = anthropic.Anthropic(api_key=key, max_retries=4, timeout=600.0)
        return self._client

    def _price(self, model: str, u: Any) -> float:
        pin, pout, pw, pr = PRICES.get(model, max(PRICES.values()))
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        return (u.input_tokens * pin + u.output_tokens * pout + cw * pw + cr * pr) / 1e6

    def ask(
        self,
        kind: str,
        schema: Mapping[str, Any],
        items: Sequence[Mapping[str, Any]],
        avoid: Sequence[str],
        *,
        system: str | None = None,
    ) -> Any:
        system = system or SYSTEM
        key = _key(kind, {"items": list(items), "schema": schema, "system": system}, avoid)
        if key in self.cache:
            return self.cache[key]
        client = self._client_or_fail()
        if self.spent + self._worst_call > self.cap:
            raise AtlasError(f"spend cap: ${self.spent:.2f} spent, the next call could pass ${self.cap:.2f}")
        what = {"rules": "rules", "records": "catalogue records", "names": "entries"}[kind]
        verb = "List the names in" if kind == "names" else "Rewrite"
        prompt = (
            f"{verb} these {len(items)} {what} following the rules."
            + ("" if kind == "names" else " Return one entry per input, same ids, same order.")
            + "\n\n"
            + json.dumps(list(items), ensure_ascii=False, indent=1)
        )
        if avoid:
            prompt += (
                "\n\nAn earlier rewrite still named these; none of them may appear in any field, in any "
                "case or spelling: " + ", ".join(sorted(avoid))
            )
        msg = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": dict(schema)}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        cost = self._price(msg.model, msg.usage)
        self.spent += cost
        self.calls += 1
        self._worst_call = max(self._worst_call, cost * 1.5)
        self.usage["input"] += msg.usage.input_tokens
        self.usage["output"] += msg.usage.output_tokens
        self.usage["cache_write"] += getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
        self.usage["cache_read"] += getattr(msg.usage, "cache_read_input_tokens", 0) or 0
        with self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "model": msg.model,
                        "kind": kind,
                        "n": len(items),
                        "input_tokens": msg.usage.input_tokens,
                        "output_tokens": msg.usage.output_tokens,
                        "usd": round(cost, 5),
                    }
                )
                + "\n"
            )
        if msg.stop_reason == "refusal":
            raise AtlasError(f"the model declined a {kind} batch ({[i['id'] for i in items]})")
        if msg.stop_reason == "max_tokens":
            raise AtlasError(f"a {kind} batch hit max_tokens ({[i['id'] for i in items]})")
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise AtlasError(f"no text in the answer to a {kind} batch")
        data = json.loads(text)
        self.cache[key] = data
        self.cache_file.write_text(
            json.dumps(self.cache, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return data


# --------------------------------------------------------------------------- assembly


def _clean(text: str) -> str:
    text = _EM_DASH.sub(", ", text)
    text = re.sub(r"\s+,", ",", text)
    return re.sub(r"\s+", " ", text).strip()


def _check_record(src: Record, out: Mapping[str, Any]) -> list[str]:
    """Shape problems of one rewritten record (empty when it is usable)."""
    problems: list[str] = []
    if out.get("id") != src.id:
        problems.append(f"id {out.get('id')!r} != {src.id!r}")
    if not 0 < len(out.get("title", "")) <= 90:
        problems.append("title length")
    subs = out.get("subtasks") or []
    if len(subs) != len(src.subtasks):
        problems.append(f"{len(subs)} subtasks for {len(src.subtasks)}")
    if any(not s.get("verb") or not s.get("object") for s in subs):
        problems.append("empty verb or object")
    if len(out.get("tools") or []) == 0:
        problems.append("no tools")
    if not 0 < len(out.get("why", "")) <= 420:
        problems.append("why length")
    return problems


def _strings(entry: Mapping[str, Any]) -> list[str]:
    out = [entry.get("title", ""), entry.get("why", ""), entry.get("text", "")]
    for s in entry.get("subtasks") or []:
        out += [s.get("verb", ""), s.get("object", ""), s.get("note", "")]
    out += list(entry.get("tools") or [])
    return [s for s in out if s]


def rewrite_all(
    records: Sequence[Record], rules: Sequence[Rule], rw: Rewriter, screen: Screen
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewritten records and rules by id; a second pass names the leftover hits to the model."""
    rec_out: dict[str, Any] = {}
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        data = rw.ask("records", _RECORD_SCHEMA, [r.prompt_view() for r in batch], ())
        by_id = {e.get("id"): e for e in data.get("records", [])}
        for r in batch:
            rec_out[r.id] = by_id.get(r.id, {})
    rule_out: dict[str, Any] = {}
    for i in range(0, len(rules), 12):
        batch = list(rules[i : i + 12])
        data = rw.ask("rules", _RULE_SCHEMA, [r.prompt_view() for r in batch], ())
        by_id = {e.get("id"): e for e in data.get("rules", [])}
        for r in batch:
            rule_out[r.id] = by_id.get(r.id, {})
    # Second pass, one record at a time: shape problems or brand hits.
    for r in records:
        e = rec_out[r.id]
        hits = sorted({h for s in _strings(e) for h in screen.hits(s)})
        if _check_record(r, e) or hits:
            data = rw.ask("records", _RECORD_SCHEMA, [r.prompt_view()], hits or ["(shape)"])
            rec_out[r.id] = (data.get("records") or [{}])[0]
    for r in rules:
        e = rule_out[r.id]
        hits = sorted({h for s in _strings(e) for h in screen.hits(s)})
        if hits or e.get("id") != r.id or not e.get("text"):
            data = rw.ask("rules", _RULE_SCHEMA, [r.prompt_view()], hits or ["(shape)"])
            rule_out[r.id] = (data.get("rules") or [{}])[0]
    return rec_out, rule_out


def assemble(
    records: Sequence[Record],
    rules: Sequence[Rule],
    rec_out: Mapping[str, Any],
    rule_out: Mapping[str, Any],
    generated: str,
) -> dict[str, Any]:
    hero_of = {rid: (case, match) for case, (rid, match) in HERO_RECORDS.items()}
    tasks = []
    for r in records:
        e = rec_out[r.id]
        problems = _check_record(r, e)
        if problems:
            raise AtlasError(f"{r.id}: the rewrite is unusable ({'; '.join(problems)})")
        case, match = hero_of.get(r.id, (None, None))
        tasks.append(
            {
                "id": r.id,
                "domain": r.domain,
                "setting": r.setting,
                "title": _clean(e["title"]),
                "subtasks": [
                    {
                        "verb": _clean(s["verb"]).lower(),
                        "object": _clean(s["object"]),
                        "note": _clean(s.get("note", "")),
                    }
                    for s in e["subtasks"]
                ],
                "setup": {"kind": r.kind, "tools": [_clean(t) for t in e["tools"]]},
                "why": _clean(e["why"]),
                "sim": {"feasibility": r.feasibility, "hero": case, "match": match},
                "demo_score": r.demo_score,
                "sources": [{"url": u} for u in r.sources],
            }
        )
    rule_rows = []
    for r in rules:
        e = rule_out[r.id]
        if e.get("id") != r.id or not e.get("title") or not e.get("text"):
            raise AtlasError(f"rule {r.id}: the rewrite is unusable")
        rule_rows.append(
            {
                "id": r.id,
                "title": _clean(e["title"]),
                "text": _clean(e["text"]),
                "sources": [{"url": u} for u in r.sources],
            }
        )
    return {
        "schema": SCHEMA,
        "version": 1,
        "generated": generated,
        "note": (
            "Researched jobs and the setup an experienced integrator would expect for each. Research "
            "estimates, not simulation results, except where a task links to a worked example."
        ),
        "domains": [{"id": d, "title": t} for d, t in DOMAINS.values()],
        "tasks": tasks,
        "rules": rule_rows,
    }


def research_date(research: Path) -> str:
    """The catalogue's own date (its newest ``generated`` field), so a rerun writes the same file."""
    dates: list[str] = []
    for stem in DOMAINS:
        data = json.loads((research / f"{stem}.json").read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            for v in (data.get("generated"), (data.get("meta") or {}).get("generated")):
                if isinstance(v, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", v):
                    dates.append(v)
    return max(dates) if dates else time.strftime("%Y-%m-%d", time.gmtime())


def _walk(value: Any, where: str) -> Iterable[tuple[str, str, str]]:
    """(path, key, string) for every string in ``value``."""
    if isinstance(value, Mapping):
        for k, v in value.items():
            if isinstance(v, str):
                yield f"{where}.{k}", str(k), v
            else:
                yield from _walk(v, f"{where}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if isinstance(v, str):
                yield f"{where}[{i}]", "", v
            else:
                yield from _walk(v, f"{where}[{i}]")


def screen_atlas(atlas: Mapping[str, Any], screen: Screen) -> list[str]:
    """Every problem with the atlas text: brand hits, URLs outside ``url`` fields, dashes."""
    problems: list[str] = []
    for path, key, text in _walk(atlas, "$"):
        if key == "url":
            if not text.startswith(("https://", "http://")):
                problems.append(f"{path}: not a URL")
            continue
        if _URL.search(text):
            problems.append(f"{path}: a URL outside a url field")
        hits = screen.hits(text)
        if hits:
            problems.append(f"{path}: names {', '.join(hits)}")
        if _EM_DASH.search(text):
            problems.append(f"{path}: an em or en dash")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--research", type=Path, required=True, help="the research catalogue directory")
    ap.add_argument("--keys", type=Path, default=None, help="JSON file with the API key under 'claude'")
    ap.add_argument("--out", type=Path, default=REPO / "web" / "public" / "data" / "atlas.json")
    ap.add_argument("--tokens", type=Path, default=REPO / "tests" / "data" / "atlas_brand_tokens.json")
    ap.add_argument("--cache", type=Path, default=REPO / "runs" / "atlas")
    ap.add_argument("--words", type=Path, default=Path("/usr/share/dict/words"))
    ap.add_argument(
        "--spend-cap", type=float, default=4.5, help="stop before spending more dollars than this"
    )
    ap.add_argument("--offline", action="store_true", help="use cached rewrites only")
    ap.add_argument("--tokens-only", action="store_true", help="write the name list and stop")
    args = ap.parse_args(argv)

    try:
        records, rules = load_research(args.research)
        rw = Rewriter(args.cache, args.keys, args.spend_cap, args.offline)
        try:
            tokens = extract_brand_names(records, rules, _lowercase_words(args.words), rw)
            doc = {
                "schema": TOKENS_SCHEMA,
                "version": 1,
                "note": (
                    "Company and product names from the use-case research (existing_products: names, makers "
                    "and bracketed trade names). The atlas shows none of them. 'ci' match in any case; 'cs' "
                    "are made of ordinary words and match only as written."
                ),
                **tokens,
            }
            args.tokens.parent.mkdir(parents=True, exist_ok=True)
            args.tokens.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"names: {len(tokens['ci'])} any-case, {len(tokens['cs'])} as written -> {args.tokens}")
            if args.tokens_only:
                return 0
            screen = Screen(tokens)
            rec_out, rule_out = rewrite_all(records, rules, rw, screen)
        finally:
            print(f"claude: {rw.calls} calls, ${rw.spent:.3f} this run; tokens {rw.usage}")
        atlas = assemble(records, rules, rec_out, rule_out, research_date(args.research))
        problems = screen_atlas(atlas, screen)
        if problems:
            raise AtlasError("the atlas fails its screen:\n  " + "\n  ".join(problems[:40]))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(atlas, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        n = len(atlas["tasks"])
        print(f"atlas: {n} tasks, {len(atlas['rules'])} rules -> {args.out}")
    except AtlasError as err:
        print(f"build_atlas_data: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
