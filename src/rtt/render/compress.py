"""Meshopt-compressed GLBs for the site, made by the web build's own compressor.

The asset store (:mod:`rtt.render.glb`) writes deterministic, quantised but uncompressed
GLBs: the offline renderers (MuJoCo studio, Blender) read those. The site serves the same
meshes meshopt-compressed (``EXT_meshopt_compression`` over ``KHR_mesh_quantization``),
which three.js decodes with its bundled decoder. :func:`compress_assets` runs
``web/scripts/compress_glb.mjs`` once over a batch of raw GLBs and returns, for each raw
asset id, the served file and its manifest entry (``docs/ASSET_MANIFEST.md``). The served
id is the hash of the compressed file, so a replay header that draws an asset must be
remapped to the served ids (:func:`remap_header_assets`).

The compressor keeps each GLB's credit (``asset.copyright``) and its licence, role and
changes (``asset.extras``); :func:`compress_assets` checks that it did. Scans of
third-party objects are simplified to half their triangles, as the asset rules say; our own
CAD and jaw geometry is never simplified. The output is deterministic for a given input and
the pinned compressor versions (``web/package.json``).

Compression needs Node and the web app's dev dependencies (``pnpm install`` in ``web/``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from rtt.render.glb import GSO_ATTRIBUTION, check_entry, read_glb_json

__all__ = [
    "COMPRESSOR",
    "SCAN_SIMPLIFY",
    "WEB_DIR",
    "CompressionUnavailable",
    "compress_assets",
    "compression_available",
    "remap_header_assets",
]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
WEB_DIR: Final = REPO_ROOT / "web"
COMPRESSOR: Final = Path("scripts") / "compress_glb.mjs"
SCAN_SIMPLIFY: Final = 0.5  # scans are served at half their triangles (docs/ATTRIBUTION.md)
_MODULES: Final = (
    "@gltf-transform/core",
    "@gltf-transform/extensions",
    "@gltf-transform/functions",
    "meshoptimizer",
)
_ENTRY_FIELDS: Final = ("role", "license", "attribution", "changes", "label", "source", "brand_visible")


class CompressionUnavailable(RuntimeError):
    """Node or the compressor's packages are missing."""


def _node(node: str | None) -> str:
    found = shutil.which(node or "node")
    if found is None:
        raise CompressionUnavailable("GLB compression needs Node.js (node on PATH)")
    return found


def compression_available(web_dir: str | Path = WEB_DIR) -> bool:
    """Whether :func:`compress_assets` can run here (Node and the web dev dependencies)."""
    web = Path(web_dir)
    if shutil.which("node") is None or not (web / COMPRESSOR).is_file():
        return False
    return all((web / "node_modules" / m / "package.json").is_file() for m in _MODULES)


def _is_scan(entry: Mapping[str, Any]) -> bool:
    """A third-party scan drawn from its own visual mesh (not our clay pieces of it)."""
    source = entry.get("source") or {}
    return entry.get("attribution") == GSO_ATTRIBUTION and source.get("kind") == "upstream"


def compress_assets(
    assets: Mapping[str, tuple[bytes, Mapping[str, Any]]],
    *,
    web_dir: str | Path = WEB_DIR,
    node: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    """Meshopt-compress raw GLBs in one compressor run.

    ``assets`` maps each raw asset id to its GLB bytes and manifest entry (as
    :class:`rtt.render.glb.AssetStore` keeps them). Returns ``{raw id: (served bytes, served
    entry)}``; each served entry is content-addressed by the compressed file and passes
    :func:`rtt.render.glb.check_entry`. Raises :class:`CompressionUnavailable` when Node or the
    packages are missing, and ``RuntimeError`` when the compressor fails or drops a credit.
    """
    if not assets:
        return {}
    web = Path(web_dir)
    script = web / COMPRESSOR
    if not script.is_file():
        raise CompressionUnavailable(f"the compressor {COMPRESSOR} is missing from {web.name}/")
    missing = [m for m in _MODULES if not (web / "node_modules" / m / "package.json").is_file()]
    if missing:
        raise CompressionUnavailable(
            f"GLB compression needs {', '.join(missing)}: run `pnpm install --frozen-lockfile` in web/"
        )
    exe = _node(node)
    with tempfile.TemporaryDirectory(prefix="rtt-compress-") as tmp:
        root = Path(tmp)
        (root / "raw").mkdir()
        inputs = []
        for aid, (data, entry) in sorted(assets.items()):
            if hashlib.sha256(data).hexdigest()[:16] != aid:
                raise ValueError(f"asset {aid}: the bytes given do not hash to its id")
            (root / "raw" / f"{aid}.glb").write_bytes(data)
            item: dict[str, Any] = {"key": aid, "file": f"raw/{aid}.glb"}
            item.update({k: entry[k] for k in _ENTRY_FIELDS if entry.get(k) is not None})
            if _is_scan(entry):
                item["simplify"] = SCAN_SIMPLIFY
            inputs.append(item)
        spec = root / "spec.json"
        spec.write_text(json.dumps({"inputs": inputs}, indent=1), encoding="utf-8")
        out = root / "out"
        cmd = [
            exe,
            str(script),
            str(spec),
            "--out-dir",
            str(out),
            "--manifest",
            str(out / "manifest.json"),
            "--map",
            str(root / "map.json"),
        ]
        proc = subprocess.run(cmd, cwd=web, capture_output=True, text=True, timeout=timeout_s, check=False)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            raise RuntimeError("GLB compression failed: " + " | ".join(tail))
        mapping = json.loads((root / "map.json").read_text(encoding="utf-8"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["assets"]
        result: dict[str, tuple[bytes, dict[str, Any]]] = {}
        for aid, (_, entry) in assets.items():
            new_id = mapping[aid]
            served = (out / f"{new_id}.glb").read_bytes()
            new_entry = dict(manifest[new_id])
            errs = check_entry(new_id, new_entry)
            if hashlib.sha256(served).hexdigest() != new_entry["sha256"] or len(served) != new_entry["bytes"]:
                errs.append(f"{new_id}: the served file does not match its entry")
            doc = read_glb_json(served)
            if doc.get("asset", {}).get("copyright") != entry["attribution"]:
                errs.append(f"{new_id}: the compressor dropped the credit line from asset.copyright")
            extras = doc.get("asset", {}).get("extras") or {}
            if extras.get("license") != entry["license"] or extras.get("role") != entry["role"]:
                errs.append(f"{new_id}: the compressor dropped the licence or role from asset.extras")
            if errs:
                raise RuntimeError("; ".join(errs))
            result[aid] = (served, new_entry)
    return result


def remap_header_assets(
    header: dict[str, Any], served: Mapping[str, tuple[bytes, Mapping[str, Any]]]
) -> None:
    """Point a replay v2 header at served (compressed) assets, in place.

    Every visual's ``asset`` and the header's ``assets`` map move from the raw ids to the
    served ids, with the served entries copied verbatim (the site checks they equal the
    manifest's). Raises ``KeyError`` for an asset that was not compressed.
    """
    ids = {aid: str(served[aid][1]["sha256"])[:16] for aid in header["assets"]}
    for body in header["bodies"]:
        for v in body.get("visuals", []):
            v["asset"] = ids[v["asset"]]
    for v in header.get("static_visuals", []):
        v["asset"] = ids[v["asset"]]
    header["assets"] = {ids[aid]: json.loads(json.dumps(served[aid][1])) for aid in sorted(header["assets"])}
    header["assets"] = dict(sorted(header["assets"].items()))
