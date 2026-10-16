"""Offscreen MP4 rendering of recorded episodes.

MuJoCo picks its OpenGL backend from ``MUJOCO_GL`` when ``mujoco`` is first
imported, so a process that already imported it cannot switch backends. If
``MUJOCO_GL`` is already ``egl`` or ``osmesa`` the video is rendered in-process;
otherwise each attempt runs in a fresh Python subprocess with
``MUJOCO_GL=egl`` first and ``MUJOCO_GL=osmesa`` as the fallback.

Rendering replays the recorded ``qpos`` trajectory (``EpisodeResult.qpos``),
so it never re-simulates and cannot change an outcome.

:func:`call_with_gl` applies the same backend rule to any rendering function (the v2
studio renderer in :mod:`rtt.render.playback` uses it): it calls ``module:function`` in
this process when a headless backend is already selected, otherwise in a fresh subprocess
per backend, passing arguments and the result by pickle.
"""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rtt.design.types import FingerGeometry
from rtt.sim.objects import AnyObject
from rtt.sim.task import TaskConfig

BACKENDS: tuple[str, ...] = ("egl", "osmesa")


@dataclass(frozen=True)
class CameraView:
    """Free-camera framing (MuJoCo ``mjvCamera`` conventions, degrees)."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.16)
    distance: float = 0.6
    azimuth: float = 160.0
    elevation: float = -12.0


@dataclass(frozen=True)
class RenderJob:
    finger: FingerGeometry
    obj: AnyObject
    task: TaskConfig
    qpos: np.ndarray  # (n_frames, nq)
    path: str
    fps: int = 30
    width: int = 640
    height: int = 480
    camera: CameraView | None = None


def default_camera(qpos: np.ndarray, object_qpos: int) -> CameraView:
    """Frame the object's starting position, looking across the closing axis."""
    x, y = (float(v) for v in qpos[0, object_qpos : object_qpos + 2])
    return CameraView(lookat=(x, y, CameraView.lookat[2]))


def _render_in_process(job: RenderJob) -> Path:
    import imageio.v2 as imageio
    import mujoco

    from rtt.sim.scene import build_model

    model, ids = build_model(job.finger, job.obj, job.task)
    data = mujoco.MjData(model)
    view = job.camera or default_camera(job.qpos, ids.object_qpos)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = view.lookat
    cam.distance = view.distance
    cam.azimuth = view.azimuth
    cam.elevation = view.elevation

    path = Path(job.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=job.height, width=job.width)
    try:
        with imageio.get_writer(
            path,
            format="FFMPEG",  # type: ignore[arg-type]
            mode="I",
            fps=job.fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=6,
            macro_block_size=16,
        ) as writer:
            for q in job.qpos:
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
    finally:
        renderer.close()
    return path


def _render_in_subprocess(job: RenderJob, backend: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="rtt-render-") as tmp:
        job_file = Path(tmp) / "job.pkl"
        job_file.write_bytes(pickle.dumps(job))
        env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        proc = subprocess.run(
            [sys.executable, "-m", "rtt.sim.render", str(job_file)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    return proc.returncode == 0, (proc.stderr or proc.stdout)[-2000:]


def render_mp4(
    finger: FingerGeometry,
    obj: AnyObject,
    task: TaskConfig,
    qpos: np.ndarray,
    path: str | Path,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera: CameraView | None = None,
) -> Path:
    """Render a recorded ``qpos`` trajectory to an H.264 MP4 and return its path."""
    job = RenderJob(
        finger=finger,
        obj=obj,
        task=task,
        qpos=np.asarray(qpos, dtype=np.float64),
        path=str(path),
        fps=fps,
        width=width,
        height=height,
        camera=camera,
    )
    if os.environ.get("MUJOCO_GL", "").lower() in BACKENDS:
        return _render_in_process(job)
    errors = []
    for backend in BACKENDS:
        ok, log = _render_in_subprocess(job, backend)
        if ok:
            return Path(path)
        errors.append(f"[{backend}] {log.strip()}")
    raise RuntimeError("offscreen rendering failed with every backend:\n" + "\n".join(errors))


def _resolve(target: str) -> Any:
    module, _, name = target.partition(":")
    if not module.startswith("rtt.") or not name:
        raise ValueError(f"call_with_gl targets an rtt function as 'module:function', got {target!r}")
    return getattr(importlib.import_module(module), name)


def call_with_gl(target: str, *args: Any, **kwargs: Any) -> Any:
    """Call ``target`` (``"rtt.module:function"``) where MuJoCo can render offscreen.

    In-process if ``MUJOCO_GL`` is ``egl`` or ``osmesa``; otherwise in a subprocess with
    ``MUJOCO_GL=egl``, then ``osmesa``. Arguments and the return value must pickle.
    """
    fn = _resolve(target)
    if os.environ.get("MUJOCO_GL", "").lower() in BACKENDS:
        return fn(*args, **kwargs)
    errors = []
    with tempfile.TemporaryDirectory(prefix="rtt-gl-") as tmp:
        job_file, out_file = Path(tmp) / "call.pkl", Path(tmp) / "result.pkl"
        job_file.write_bytes(pickle.dumps((target, args, kwargs)))
        for backend in BACKENDS:
            env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
            proc = subprocess.run(
                [sys.executable, "-m", "rtt.sim.render", "--call", str(job_file), str(out_file)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and out_file.exists():
                return pickle.loads(out_file.read_bytes())  # written by our own subprocess
            errors.append(f"[{backend}] {(proc.stderr or proc.stdout)[-2000:].strip()}")
    raise RuntimeError(f"{target}: offscreen rendering failed with every backend:\n" + "\n".join(errors))


def _main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "--call":
        target, args, kwargs = pickle.loads(Path(argv[1]).read_bytes())  # written by call_with_gl
        Path(argv[2]).write_bytes(pickle.dumps(_resolve(target)(*args, **kwargs)))
        return 0
    if len(argv) != 1:
        print("usage: python -m rtt.sim.render JOB.pkl | --call CALL.pkl RESULT.pkl", file=sys.stderr)
        return 2
    job = pickle.loads(Path(argv[0]).read_bytes())  # written by render_mp4 in this package
    _render_in_process(job)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
