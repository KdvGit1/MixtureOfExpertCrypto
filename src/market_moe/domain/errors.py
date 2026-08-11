"""Typed errors surfaced by services and the web API."""


class MarketMoEError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(MarketMoEError):
    """Raised when a local configuration is invalid."""


class DataProviderError(MarketMoEError):
    """Raised when a free public data provider cannot satisfy a request."""


class DataQualityError(MarketMoEError):
    """Raised when market data violates canonical invariants."""


class FeatureSchemaError(MarketMoEError):
    """Raised when model and feature schemas are incompatible."""


class ModelBundleError(MarketMoEError):
    """Raised when a local model bundle is incomplete or untrusted."""


class ModelCompatibilityError(ModelBundleError):
    """Raised when a bundle cannot consume the requested feature contract."""


class BacktestError(MarketMoEError):
    """Raised when a backtest request is invalid."""
