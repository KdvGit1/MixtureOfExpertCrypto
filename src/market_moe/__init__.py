"""MarketMoE local-first market research platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("market-moe")
except PackageNotFoundError:  # pragma: no cover - editable source tree fallback
    __version__ = "0.1.0"

__all__ = ["__version__"]
