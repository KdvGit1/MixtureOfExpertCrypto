from __future__ import annotations

from pathlib import Path

from market_moe.diagnostics import diagnostics_payload
from market_moe.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_package_has_no_order_or_secret_configuration() -> None:
    source = "\n".join(
        path.read_text("utf-8") for path in (PROJECT_ROOT / "src" / "market_moe").rglob("*.py")
    )
    forbidden = ("create_order(", "place_order(", "api_secret:", "broker_password")
    assert all(token not in source for token in forbidden)


def test_default_settings_are_local_and_keyless(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        config_dir=PROJECT_ROOT / "configs",
    )
    payload = diagnostics_payload(settings)
    assert payload["healthy"]
    assert settings.host == "127.0.0.1"
    assert not any(
        "key" in name.lower() or "secret" in name.lower() for name in Settings.model_fields
    )
