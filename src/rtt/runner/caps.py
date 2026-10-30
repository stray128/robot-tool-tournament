"""The runner's hard caps as store operations, plus the kill switch and the auto-off date.

Caps are checked in the same transaction that takes them (:meth:`rtt.runner.store.Store.commit`),
before any money is spent:

* free-tier plans: ``free_plans_per_day`` site-wide and ``free_plans_per_ip_per_day`` per IP;
* plans on the visitor's own key: ``keyed_plans_per_ip_per_day`` per IP;
* runs: ``runs_per_ip_per_day`` per IP, and each run reserves its worst-case Nebius cost against
  ``spend_per_day_usd`` and ``spend_total_usd``; it settles to its estimated cost when it ends.

Days are UTC days; a daily cap resets at the next UTC midnight, the total cap never does.

IP addresses are never stored or logged. :func:`ip_hash` keys an HMAC with a server secret and the
day, so a hash cannot be reversed by trying every address and two days' hashes of one address do
not match.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

from rtt.cloud.cost import hourly_usd
from rtt.runner.config import RunnerConfig, to_micros
from rtt.runner.store import Bump

__all__ = [
    "CAP_NAMES",
    "CapRefused",
    "KillSwitch",
    "ServiceState",
    "cap_refusal",
    "client_key",
    "day_of",
    "hourly_job_usd",
    "ip_hash",
    "ip_salt",
    "job_cost_micros",
    "plan_bumps",
    "reservation_micros",
    "run_bumps",
    "settle_bumps",
    "spend_keys",
]

SALT_CONTEXT: Final = b"effector-lab/runner/ip-hash-salt/v1"
SPEND_TOTAL_KEY: Final = "spend#nebius#total"

#: cap name -> what the 429 says
CAP_NAMES: Final = {
    "free_plans_per_day": "the free planning tier's daily limit for the whole site",
    "free_plans_per_ip_per_day": "the free planning tier's daily limit per visitor",
    "keyed_plans_per_day": "the daily limit on plans with your own key for the whole site",
    "keyed_plans_per_ip_per_day": "the daily limit on plans with your own key per visitor",
    "runs_per_ip_per_day": "the daily limit on runs per visitor",
    "spend_per_day_usd": "the runner's daily compute budget",
    "spend_total_usd": "the runner's total compute budget",
}


def day_of(now: datetime) -> str:
    return now.astimezone(UTC).date().isoformat()


def next_reset(now: datetime) -> str:
    tomorrow = now.astimezone(UTC).date() + timedelta(days=1)
    return (
        datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    )


def _epoch(d: date, days: int) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) + days * 86400


def ip_salt(server_secret: str) -> bytes:
    """A key for :func:`ip_hash`, derived from a server secret (domain-separated HMAC)."""
    return hmac.new(server_secret.encode(), SALT_CONTEXT, hashlib.sha256).digest()


def client_key(ip: str) -> str:
    """What a per-visitor cap counts: the IPv4 address, or the /64 prefix of an IPv6 address (one
    subscriber usually holds a whole /64, so counting single v6 addresses would be no cap at all).
    An IPv4-mapped IPv6 address counts as its IPv4 address; anything unparsable as itself."""
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


def ip_hash(salt: bytes, day: str, ip: str) -> str:
    """HMAC of the day and :func:`client_key` of ``ip`` (32 hex characters)."""
    return hmac.new(salt, f"{day}|{client_key(ip)}".encode(), hashlib.sha256).hexdigest()[:32]


def hourly_job_usd(cfg: RunnerConfig) -> float:
    """List price per hour of the job's platform and preset (``cpu-d3`` 32 vCPU: $1.056/h)."""
    return hourly_usd(cfg.nebius.platform, cfg.nebius.preset)


def job_cost_micros(cfg: RunnerConfig, wall_s: float) -> int:
    """Estimated cost of a job: wall time (creation to finish) times the hourly price [assumed]."""
    return to_micros(hourly_job_usd(cfg) * max(0.0, wall_s) / 3600.0)


def reservation_micros(cfg: RunnerConfig) -> int:
    """What a run reserves before it is submitted: the wall-time limit plus the cancel grace."""
    return job_cost_micros(cfg, cfg.caps.job_wall_limit_s + cfg.caps.cancel_grace_s)


def spend_keys(day: str) -> tuple[str, str]:
    return f"spend#nebius#day#{day}", SPEND_TOTAL_KEY


def plan_bumps(cfg: RunnerConfig, provider: str, ip: str, now: datetime) -> list[Bump]:
    day = day_of(now)
    exp = _epoch(now.astimezone(UTC).date(), 2)
    caps = cfg.caps
    if provider == "free":
        return [
            Bump(f"count#free-plans#{day}", "n", 1, exp, caps.free_plans_per_day, "free_plans_per_day"),
            Bump(
                f"count#free-plans#{ip}#{day}",
                "n",
                1,
                exp,
                caps.free_plans_per_ip_per_day,
                "free_plans_per_ip_per_day",
            ),
        ]
    return [
        Bump(f"count#keyed-plans#{day}", "n", 1, exp, caps.keyed_plans_per_day, "keyed_plans_per_day"),
        Bump(
            f"count#keyed-plans#{ip}#{day}",
            "n",
            1,
            exp,
            caps.keyed_plans_per_ip_per_day,
            "keyed_plans_per_ip_per_day",
        ),
    ]


def _spend_expiry(cfg: RunnerConfig, now: datetime) -> tuple[int, int]:
    """Day items live 40 days; the total lives 180 days past the auto-off date."""
    return _epoch(now.astimezone(UTC).date(), 40), _epoch(cfg.auto_off, 180)


def run_bumps(cfg: RunnerConfig, ip: str, reserve: int, now: datetime) -> list[Bump]:
    day = day_of(now)
    day_key, total_key = spend_keys(day)
    day_exp, total_exp = _spend_expiry(cfg, now)
    caps = cfg.caps
    return [
        Bump(
            f"count#runs#{ip}#{day}",
            "n",
            1,
            _epoch(now.astimezone(UTC).date(), 2),
            caps.runs_per_ip_per_day,
            "runs_per_ip_per_day",
        ),
        Bump(day_key, "committed", reserve, day_exp, to_micros(caps.spend_per_day_usd), "spend_per_day_usd"),
        Bump(total_key, "committed", reserve, total_exp, to_micros(caps.spend_total_usd), "spend_total_usd"),
    ]


def settle_bumps(
    cfg: RunnerConfig, day: str, reserved: int, actual: int, now: datetime, *, release_ip: str | None = None
) -> list[Bump]:
    """Replace a run's reservation with its estimated cost (on the day it was reserved).
    ``release_ip``: also give the IP its run back (the job never started through our fault)."""
    day_key, total_key = spend_keys(day)
    day_exp, total_exp = _spend_expiry(cfg, now)
    delta = actual - reserved
    out = [
        Bump(day_key, "committed", delta, day_exp),
        Bump(day_key, "settled", actual, day_exp),
        Bump(total_key, "committed", delta, total_exp),
        Bump(total_key, "settled", actual, total_exp),
    ]
    if release_ip is not None:
        out.append(Bump(f"count#runs#{release_ip}#{day}", "n", -1, _epoch(date.fromisoformat(day), 2)))
    return [b for b in out if b.amount != 0]


class CapRefused(Exception):
    """A cap refused the request: which one, its limit, and when it resets (``None``: never)."""

    def __init__(self, cap: str, limit: float, resets_at: str | None) -> None:
        super().__init__(f"cap reached: {cap}")
        self.cap = cap
        self.limit = limit
        self.resets_at = resets_at

    def body(self) -> dict[str, Any]:
        what = CAP_NAMES.get(self.cap, self.cap)
        when = f"it resets at {self.resets_at}" if self.resets_at else "it does not reset"
        return {
            "error": f"cap reached: {what}; {when}",
            "cap": self.cap,
            "limit": self.limit,
            "resets_at": self.resets_at,
        }


def cap_refusal(cfg: RunnerConfig, bump: Bump, now: datetime) -> CapRefused:
    limit: float = getattr(cfg.caps, bump.cap, bump.limit or 0)
    resets = None if bump.cap == "spend_total_usd" else next_reset(now)
    return CapRefused(bump.cap, limit, resets)


# --------------------------------------------------------------------------- kill switch


@dataclass(frozen=True)
class ServiceState:
    kill_switch: bool
    auto_off: bool

    @property
    def open(self) -> bool:
        return not (self.kill_switch or self.auto_off)


class KillSwitch:
    """The kill switch: the ``kill_switch`` setting, or "on" in the SSM parameter.

    The parameter is read at most every ``kill_switch_ttl_s``. When it cannot be read the switch
    counts as on (fail closed: no new spend while the runner cannot tell).
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        reader: Callable[[str], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._reader = reader
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: tuple[float, bool] | None = None

    def _read(self, name: str) -> str:
        if self._reader is not None:
            return self._reader(name)
        import boto3

        resp = boto3.client("ssm").get_parameter(Name=name)
        return str(resp["Parameter"]["Value"])

    def on(self) -> bool:
        if self.cfg.kill_switch:
            return True
        name = self.cfg.kill_switch_param
        if not name:
            return False
        with self._lock:
            now = self._clock()
            if self._cached is not None and now - self._cached[0] < self.cfg.kill_switch_ttl_s:
                return self._cached[1]
            try:
                value = self._read(name).strip().lower() not in ("off", "false", "0", "")
            except Exception:  # noqa: BLE001 - any failure to read counts as on
                value = True
            self._cached = (now, value)
            return value

    def state(self, now: datetime) -> ServiceState:
        return ServiceState(self.on(), now.astimezone(UTC).date() > self.cfg.auto_off)
