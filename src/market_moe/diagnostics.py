"""Offline-safe installation, policy and local-state diagnostics."""

from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import asdict, dataclass

from market_moe.data.universe import discover_universes, load_universe
from market_moe.settings import Settings


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    name: str
    ok: bool
    detail: str
    critical: bool = True


def run_diagnostics(settings: Settings) -> list[DiagnosticCheck]:
    checks = [
        DiagnosticCheck(
            "python", platform.python_version_tuple() >= ("3", "11", "0"), platform.python_version()
        ),
        DiagnosticCheck(
            "localhost_bind", settings.host in {"127.0.0.1", "localhost"}, settings.host
        ),
        DiagnosticCheck(
            "cors_no_wildcard",
            "*" not in settings.allowed_origins,
            ", ".join(settings.allowed_origins),
        ),
        DiagnosticCheck("no_paid_api_key", True, "no key is defined in Settings"),
    ]
    for package in ("torch", "pandas", "fastapi", "ccxt", "yfinance", "duckdb"):
        try:
            version = importlib.metadata.version(package)
            checks.append(DiagnosticCheck(f"dependency:{package}", True, version))
        except importlib.metadata.PackageNotFoundError:
            checks.append(DiagnosticCheck(f"dependency:{package}", False, "not installed"))
    try:
        settings.ensure_local_directories()
        checks.append(DiagnosticCheck("local_directories", True, str(settings.data_dir)))
    except OSError as exc:
        checks.append(DiagnosticCheck("local_directories", False, str(exc)))
    try:
        discovered = discover_universes(settings.config_dir)
        instruments = sum(len(load_universe(path).instruments) for path in discovered.values())
        checks.append(
            DiagnosticCheck(
                "universes",
                bool(discovered),
                f"{len(discovered)} universes, {instruments} instruments",
            )
        )
    except Exception as exc:
        checks.append(DiagnosticCheck("universes", False, str(exc)))
    return checks


def diagnostics_payload(settings: Settings) -> dict[str, object]:
    checks = run_diagnostics(settings)
    return {
        "healthy": all(check.ok for check in checks if check.critical),
        "checks": [asdict(check) for check in checks],
    }
