"""Local-only application settings and filesystem layout."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MARKET_MOE_", env_file=".env", extra="ignore")

    project_root: Path = PROJECT_ROOT
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    config_dir: Path = Field(default=PROJECT_ROOT / "configs")
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: str = "INFO"
    default_equity_provider: str = "yfinance"
    default_crypto_provider: str = "ccxt"
    default_base_currency: str = "USD"
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    cache_freshness_minutes: int = Field(default=60, ge=0)
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    )

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_data_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def feature_data_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def backtest_dir(self) -> Path:
        return self.artifacts_dir / "backtests"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.duckdb"

    def ensure_local_directories(self) -> None:
        for path in (
            self.raw_data_dir,
            self.normalized_data_dir,
            self.feature_data_dir,
            self.model_dir,
            self.backtest_dir,
            self.artifacts_dir / "reports",
            self.artifacts_dir / "experiments",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_local_directories()
    return settings
