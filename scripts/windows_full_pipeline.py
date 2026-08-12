"""One-command Windows CUDA pipeline: free data -> train -> locked backtests."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import html
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_moe.data.protocols import timeframe_delta
from market_moe.data.universe import discover_universes, load_universe
from market_moe.domain.instruments import AssetClass, Instrument
from market_moe.models.bundle import ModelBundle
from market_moe.services.data_service import DataService
from market_moe.settings import Settings
from market_moe.training.pipeline import PooledTrainingSpec, train_pooled_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONTRACT_VERSION = 2


class JobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    enabled: bool = True
    model_id: str
    asset_class: str
    universes: list[str]
    timeframe: str
    history_days: int = Field(ge=30)
    minimum_usable_rows: int = Field(ge=100)
    require_complete_universe: bool = True
    window: int = Field(default=120, ge=2)
    horizon_bars: int = Field(default=1, ge=1)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    embed_dim: int = Field(default=96, ge=8)
    router_hidden_dim: int = Field(default=32, ge=4)
    dropout: float = Field(default=0.20, ge=0, lt=1)
    branch_dropout_probability: float = Field(default=0.20, ge=0, lt=1)
    batch_size: int = Field(default=256, ge=1)
    minimum_batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=30, ge=1)
    patience: int = Field(default=7, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    seed: int = 20260811
    branch_dropout_start_epoch: int = Field(default=4, ge=0)
    probability_threshold: float = Field(default=0.58, gt=0.5, lt=1)
    allow_short: bool = False
    periods_per_year: int = Field(default=252, ge=1)
    initial_cash: float = Field(default=100_000.0, gt=0)
    maximum_position_fraction: float = Field(default=1.0, gt=0, le=1)
    maximum_drawdown: float = Field(default=0.50, gt=0, le=1)
    maximum_gap_fraction: float = Field(default=0.25, gt=0)
    commission_bps: float = Field(default=5.0, ge=0)
    spread_bps: float = Field(default=5.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    funding_bps_per_bar: float = Field(default=0.0, ge=0)
    borrow_bps_per_bar: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_job(self) -> JobConfig:
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train + validation fractions must leave a test fold")
        if self.minimum_batch_size > self.batch_size:
            raise ValueError("minimum_batch_size cannot exceed batch_size")
        if self.minimum_usable_rows < self.window + 202:
            raise ValueError("minimum_usable_rows must cover EMA-200 warmup plus model window")
        return self


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_id: str = "windows_cuda_full"
    require_cuda: bool = True
    minimum_vram_gb: float = Field(default=10.0, gt=0)
    minimum_free_disk_gb: float = Field(default=20.0, gt=0)
    download_retries: int = Field(default=3, ge=1, le=10)
    cache_latest_tolerance_days: int = Field(default=7, ge=0)
    resume: bool = True
    auto_promote: bool = False
    jobs: list[JobConfig]

    @model_validator(mode="after")
    def validate_pipeline(self) -> PipelineConfig:
        ids = [job.job_id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("job_id values must be unique")
        if self.auto_promote:
            raise ValueError("auto_promote is forbidden; candidates require explicit review")
        return self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download free data, train CUDA MoE models and run locked backtests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipelines" / "windows_cuda_12gb.yaml",
    )
    parser.add_argument("--only", action="append", default=[], metavar="JOB_ID")
    parser.add_argument("--force", action="store_true", help="create new versions and rerun jobs")
    parser.add_argument("--offline", action="store_true", help="use cache without network calls")
    parser.add_argument("--allow-cpu", action="store_true", help="development/smoke use only")
    parser.add_argument("--check", action="store_true", help="validate configuration and exit")
    return parser.parse_args()


def load_config(path: Path) -> PipelineConfig:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid pipeline config: {resolved}")
    return PipelineConfig.model_validate(payload)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def setup_logging(run_root: Path) -> logging.Logger:
    run_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("market_moe.windows_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(run_root / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


@contextmanager
def prevent_windows_sleep() -> Iterator[None]:
    """Keep the system awake; the screen may still turn off."""

    continuous = 0x80000000
    system_required = 0x00000001
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(continuous | system_required)
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(continuous)


def environment_report(config: PipelineConfig, *, allow_cpu: bool) -> tuple[str, dict[str, object]]:
    cuda = torch.cuda.is_available()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        vram_gb = properties.total_memory / 1024**3
        report.update({"device": properties.name, "vram_gb": round(vram_gb, 2)})
        if vram_gb < config.minimum_vram_gb:
            raise RuntimeError(
                f"GPU has {vram_gb:.1f} GB VRAM; config requires {config.minimum_vram_gb:.1f} GB"
            )
        device = "cuda"
    elif config.require_cuda and not allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Run the BAT so it can install the CUDA PyTorch wheel, "
            "then verify the NVIDIA driver with nvidia-smi."
        )
    else:
        device = "cpu"
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    report["free_disk_gb"] = round(free_gb, 2)
    if free_gb < config.minimum_free_disk_gb:
        raise RuntimeError(
            f"only {free_gb:.1f} GB free disk; config requires {config.minimum_free_disk_gb:.1f} GB"
        )
    return device, report


def instruments_for(job: JobConfig, settings: Settings) -> list[Instrument]:
    discovered = discover_universes(settings.config_dir)
    selected: dict[str, Instrument] = {}
    for universe_id in job.universes:
        if universe_id not in discovered:
            raise ValueError(f"unknown universe in {job.job_id}: {universe_id}")
        universe = load_universe(discovered[universe_id])
        for instrument in universe.instruments:
            if instrument.active and instrument.asset_class.value == job.asset_class:
                selected[instrument.instrument_id] = instrument
    if not selected:
        raise ValueError(f"{job.job_id} selected no active instruments")
    return list(selected.values())


def job_signature(job: JobConfig, instruments: list[Instrument]) -> str:
    payload = {
        "pipeline_contract_version": PIPELINE_CONTRACT_VERSION,
        "job": job.model_dump(mode="json"),
        "instruments": [item.model_dump(mode="json") for item in instruments],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_providers(instrument: Instrument) -> list[str]:
    if instrument.asset_class == AssetClass.CRYPTO:
        return ["ccxt_binance", "ccxt_bitget"]
    providers = ["yfinance"]
    if instrument.country == "US" and "stooq" in instrument.provider_symbols:
        providers.append("stooq")
    return providers


def _best_cache(
    service: DataService, instrument: Instrument, timeframe: str
) -> tuple[pd.DataFrame | None, str | None]:
    choices: list[tuple[pd.DataFrame, str]] = []
    for provider in _candidate_providers(instrument):
        try:
            frame = service.cache.load(provider, instrument, timeframe)
            if frame is not None:
                choices.append((frame, provider))
        except Exception:
            continue
    if not choices:
        return None, None
    return max(choices, key=lambda item: len(item[0]))


def _cache_is_adequate(
    frame: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
    minimum_rows: int,
    latest_tolerance_days: int,
) -> bool:
    if not _cache_has_history(frame, start=start, end=end, minimum_rows=minimum_rows):
        return False
    stamps = pd.to_datetime(frame["open_time_utc"], utc=True)
    interval = timeframe_delta(timeframe)
    tolerance = (
        timedelta(days=latest_tolerance_days) if timeframe == "1d" else interval * 3
    )
    recent_enough = stamps.max() >= pd.Timestamp(end - tolerance)
    return bool(recent_enough)


def _cache_has_history(
    frame: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    minimum_rows: int,
) -> bool:
    if len(frame) < minimum_rows:
        return False
    stamps = pd.to_datetime(frame["open_time_utc"], utc=True)
    requested_span = end - start
    return bool(stamps.min() <= pd.Timestamp(start + requested_span * 0.05))


def load_or_download(
    service: DataService,
    instrument: Instrument,
    job: JobConfig,
    config: PipelineConfig,
    logger: logging.Logger,
    *,
    offline: bool,
) -> tuple[pd.DataFrame, str, str | None]:
    end = datetime.now(UTC)
    start = end - timedelta(days=job.history_days)
    cached, cached_provider = _best_cache(service, instrument, job.timeframe)
    if cached is not None and _cache_is_adequate(
        cached,
        start=start,
        end=end,
        timeframe=job.timeframe,
        minimum_rows=job.minimum_usable_rows,
        latest_tolerance_days=config.cache_latest_tolerance_days,
    ):
        return cached, str(cached_provider), None
    if offline:
        if cached is not None and len(cached) >= job.minimum_usable_rows:
            return cached, str(cached_provider), "offline mode used incomplete/stale cache"
        raise RuntimeError("offline cache is missing or too short")

    last_error: Exception | None = None
    for attempt in range(1, config.download_retries + 1):
        provider_order = _candidate_providers(instrument)
        if cached_provider in provider_order:
            provider_order.remove(str(cached_provider))
            provider_order.insert(0, str(cached_provider))
        for provider in provider_order:
            try:
                fetch_start = start
                if (
                    cached is not None
                    and cached_provider == provider
                    and _cache_has_history(
                        cached,
                        start=start,
                        end=end,
                        minimum_rows=job.minimum_usable_rows,
                    )
                ):
                    latest = pd.to_datetime(cached["open_time_utc"], utc=True).max()
                    fetch_start = (latest - timeframe_delta(job.timeframe) * 2).to_pydatetime()
                _fetched, quality_report = service.fetch(
                    instrument,
                    job.timeframe,
                    fetch_start,
                    end,
                    provider_name=provider,
                    adjusted=provider != "stooq",
                )
                loaded = service.cache.load(provider, instrument, job.timeframe)
                if loaded is None or len(loaded) < job.minimum_usable_rows:
                    count = 0 if loaded is None else len(loaded)
                    raise RuntimeError(
                        f"only {count} usable rows; need at least {job.minimum_usable_rows}"
                    )
                quality_note = (
                    ",".join(quality_report.warnings) if quality_report.warnings else None
                )
                return loaded, provider, quality_note
            except Exception as error:
                last_error = error
                logger.warning(
                    "%s provider %s failed: %s",
                    instrument.instrument_id,
                    provider,
                    error,
                )
        if last_error is not None:
            logger.warning(
                "%s download attempt %d/%d failed: %s",
                instrument.instrument_id,
                attempt,
                config.download_retries,
                last_error,
            )
        if attempt < config.download_retries:
            time.sleep(min(30, 2**attempt))
    if cached is not None and len(cached) >= job.minimum_usable_rows:
        return cached, str(cached_provider), f"network failed; cache used: {last_error}"
    raise RuntimeError(str(last_error))


def spec_from_job(job: JobConfig, *, version: str, providers: set[str], batch: int) -> PooledTrainingSpec:
    return PooledTrainingSpec(
        job_id=job.job_id,
        model_id=job.model_id,
        asset_class=job.asset_class,
        timeframe=job.timeframe,
        version=version,
        window=job.window,
        horizon_bars=job.horizon_bars,
        train_fraction=job.train_fraction,
        validation_fraction=job.validation_fraction,
        embed_dim=job.embed_dim,
        router_hidden_dim=job.router_hidden_dim,
        dropout=job.dropout,
        branch_dropout_probability=job.branch_dropout_probability,
        batch_size=batch,
        epochs=job.epochs,
        patience=job.patience,
        learning_rate=job.learning_rate,
        weight_decay=job.weight_decay,
        gradient_clip=job.gradient_clip,
        seed=job.seed,
        branch_dropout_start_epoch=job.branch_dropout_start_epoch,
        probability_threshold=job.probability_threshold,
        allow_short=job.allow_short,
        periods_per_year=job.periods_per_year,
        initial_cash=job.initial_cash,
        maximum_position_fraction=job.maximum_position_fraction,
        maximum_drawdown=job.maximum_drawdown,
        maximum_gap_fraction=job.maximum_gap_fraction,
        commission_bps=job.commission_bps,
        spread_bps=job.spread_bps,
        slippage_bps=job.slippage_bps,
        funding_bps_per_bar=job.funding_bps_per_bar,
        borrow_bps_per_bar=job.borrow_bps_per_bar,
        providers=tuple(sorted(providers)),
    )


def _complete_bundle(path: Path) -> bool:
    return path.is_dir() and ModelBundle.required_files <= {item.name for item in path.iterdir()}


def write_summary_html(run_root: Path, state: dict[str, object]) -> None:
    def file_link(value: object) -> str:
        label = str(value)
        if not label:
            return ""
        path = Path(label)
        target = path.as_uri() if path.is_absolute() else label
        return f'<a href="{html.escape(target)}">{html.escape(label)}</a>'

    jobs = state.get("jobs", {})
    rows = []
    if isinstance(jobs, dict):
        for job_id, raw in jobs.items():
            detail = raw if isinstance(raw, dict) else {}
            metrics = detail.get("metrics", {})
            metrics = metrics if isinstance(metrics, dict) else {}
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(job_id))}</td>"
                f"<td>{html.escape(str(detail.get('status', 'unknown')))}</td>"
                f"<td>{file_link(detail.get('bundle', ''))}</td>"
                f"<td>{file_link(detail.get('backtests', ''))}</td>"
                f"<td>{html.escape(str(metrics.get('mae', '')))}</td>"
                f"<td>{html.escape(str(metrics.get('sign_accuracy', '')))}</td>"
                f"<td>{html.escape(str(detail.get('error', '')))}</td>"
                "</tr>"
            )
    document = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>MarketMoE CUDA Pipeline</title><style>
body{{font:15px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:.6rem;border-bottom:1px solid #ddd;text-align:left}}
.note{{padding:1rem;background:#fff7d6;border-radius:.5rem}}
</style></head><body><h1>MarketMoE eğitim ve backtest özeti</h1>
<p class="note">Araştırma simülasyonudur; yatırım tavsiyesi değildir. Modeller otomatik olarak production'a alınmaz.</p>
<p>Durum: {html.escape(str(state.get('status', 'çalışıyor')))}<br>
Başlangıç: {html.escape(str(state.get('started_at_utc', '')))}<br>
Güncelleme: {html.escape(str(state.get('updated_at_utc', '')))}</p>
<table><thead><tr><th>İş</th><th>Durum</th><th>Model bundle</th><th>Backtest</th>
<th>Test MAE</th><th>Yön doğruluğu</th><th>Hata</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    (run_root / "summary.html").write_text(document, encoding="utf-8")


def run() -> int:
    args = parse_args()
    config = load_config(args.config)
    enabled = [job for job in config.jobs if job.enabled]
    if args.only:
        requested = set(args.only)
        known = {job.job_id for job in enabled}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown or disabled --only jobs: {sorted(unknown)}")
        enabled = [job for job in enabled if job.job_id in requested]
    if args.check:
        check_settings = Settings(project_root=PROJECT_ROOT)
        print(
            json.dumps(
                {
                    "valid": True,
                    "pipeline_id": config.pipeline_id,
                    "enabled_jobs": [
                        {
                            "job_id": job.job_id,
                            "expected_instruments": len(instruments_for(job, check_settings)),
                            "require_complete_universe": job.require_complete_universe,
                            "history_days": job.history_days,
                        }
                        for job in enabled
                    ],
                    "auto_promote": config.auto_promote,
                },
                indent=2,
            )
        )
        return 0

    settings = Settings(project_root=PROJECT_ROOT)
    settings.ensure_local_directories()
    run_root = settings.artifacts_dir / "pipeline_runs" / config.pipeline_id
    logger = setup_logging(run_root)
    state_path = run_root / "state.json"
    if state_path.exists() and not args.force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "pipeline_id": config.pipeline_id,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "jobs": {},
        }
    state_jobs = state.setdefault("jobs", {})
    if not isinstance(state_jobs, dict):
        raise ValueError("pipeline state jobs field is invalid")
    device, hardware = environment_report(config, allow_cpu=args.allow_cpu)
    state["hardware"] = hardware
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json(state_path, state)
    logger.info("Hardware: %s", json.dumps(hardware, ensure_ascii=False))
    logger.info("Selected jobs: %s", ", ".join(job.job_id for job in enabled))
    service = DataService(settings)
    failures = 0

    with prevent_windows_sleep():
        for job in enabled:
            previous = state_jobs.get(job.job_id, {})
            previous = previous if isinstance(previous, dict) else {}
            selected_instruments = instruments_for(job, settings)
            signature = job_signature(job, selected_instruments)
            signature_changed = previous.get("job_signature") != signature
            previous_bundle = Path(str(previous.get("bundle", "")))
            if (
                not args.force
                and not signature_changed
                and previous.get("status") == "completed"
                and _complete_bundle(previous_bundle)
            ):
                logger.info("%s already completed; skipping", job.job_id)
                continue
            if signature_changed and previous:
                logger.info("%s config/universe changed; creating a new model version", job.job_id)
            version = (
                str(previous.get("version", ""))
                if not args.force and not signature_changed
                else ""
            )
            if not version:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                version = f"{stamp}-{job.job_id}"
            job_state: dict[str, object] = {
                "status": "downloading",
                "version": version,
                "job_signature": signature,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "instruments": {},
            }
            state_jobs[job.job_id] = job_state
            state["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(state_path, state)
            sources: list[tuple[Instrument, pd.DataFrame]] = []
            providers: set[str] = set()
            warnings: list[str] = []
            logger.info("%s data stage started", job.job_id)
            job_state["expected_instruments"] = len(selected_instruments)
            for instrument in selected_instruments:
                try:
                    frame, provider, warning = load_or_download(
                        service,
                        instrument,
                        job,
                        config,
                        logger,
                        offline=args.offline,
                    )
                    sources.append((instrument, frame))
                    providers.add(provider)
                    instrument_state = {
                        "status": "ready",
                        "provider": provider,
                        "rows": len(frame),
                        "warning": warning,
                    }
                    if warning:
                        warnings.append(f"{instrument.instrument_id}: {warning}")
                        logger.info("%s quality notes: %s", instrument.instrument_id, warning)
                    logger.info(
                        "%s ready (%s, %d rows)", instrument.instrument_id, provider, len(frame)
                    )
                except Exception as error:
                    instrument_state = {"status": "failed", "error": str(error)}
                    warnings.append(f"{instrument.instrument_id}: {error}")
                    logger.error("%s unavailable: %s", instrument.instrument_id, error)
                instruments_state = job_state["instruments"]
                if isinstance(instruments_state, dict):
                    instruments_state[instrument.instrument_id] = instrument_state
                state["updated_at_utc"] = datetime.now(UTC).isoformat()
                atomic_json(state_path, state)

            ready_ratio = len(sources) / len(selected_instruments)
            job_state["ready_instruments"] = len(sources)
            job_state["data_coverage_ratio"] = ready_ratio
            if job.require_complete_universe and len(sources) != len(selected_instruments):
                missing = sorted(
                    instrument.instrument_id
                    for instrument in selected_instruments
                    if instrument.instrument_id not in {item.instrument_id for item, _ in sources}
                )
                error = (
                    f"incomplete universe: {len(sources)}/{len(selected_instruments)} ready; "
                    f"missing={missing}"
                )
                job_state.update(
                    {
                        "status": "failed",
                        "error": error,
                        "warnings": warnings,
                        "finished_at_utc": datetime.now(UTC).isoformat(),
                    }
                )
                failures += 1
                logger.error("%s failed: %s", job.job_id, error)
                atomic_json(state_path, state)
                continue

            bundle_path = settings.model_dir / job.model_id / version
            backtest_root = settings.backtest_dir / job.model_id / version
            checkpoint = run_root / "checkpoints" / f"{version}.pt"
            batch = job.batch_size
            job_state["status"] = "training"
            job_state["batch_size"] = batch
            atomic_json(state_path, state)
            while True:
                spec = spec_from_job(job, version=version, providers=providers, batch=batch)
                try:
                    result = train_pooled_model(
                        sources,
                        spec,
                        bundle_path=bundle_path,
                        backtest_root=backtest_root,
                        registry_root=settings.model_dir,
                        checkpoint_path=checkpoint,
                        resume=config.resume,
                        device=device,
                        log=logger.info,
                    )
                    job_state.update(
                        {
                            "status": "completed",
                            "bundle": str(result.bundle_path.resolve()),
                            "backtests": str(result.backtest_root.resolve()),
                            "metrics": result.metrics,
                            "backtest_count": len(result.backtests),
                            "warnings": [*warnings, *result.warnings],
                            "batch_size": batch,
                            "finished_at_utc": datetime.now(UTC).isoformat(),
                        }
                    )
                    logger.info("%s completed: %s", job.job_id, result.bundle_path)
                    break
                except torch.OutOfMemoryError as error:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    next_batch = batch // 2
                    if next_batch < job.minimum_batch_size:
                        job_state.update(
                            {
                                "status": "failed",
                                "error": f"CUDA OOM at minimum batch size: {error}",
                                "finished_at_utc": datetime.now(UTC).isoformat(),
                            }
                        )
                        failures += 1
                        logger.exception("%s exhausted CUDA OOM fallback", job.job_id)
                        break
                    batch = next_batch
                    job_state["batch_size"] = batch
                    logger.warning("%s CUDA OOM; retrying with batch_size=%d", job.job_id, batch)
                    atomic_json(state_path, state)
                except Exception as error:
                    job_state.update(
                        {
                            "status": "failed",
                            "error": str(error),
                            "warnings": warnings,
                            "finished_at_utc": datetime.now(UTC).isoformat(),
                        }
                    )
                    failures += 1
                    logger.exception("%s failed", job.job_id)
                    break
            state["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(state_path, state)
            write_summary_html(run_root, state)

    state["status"] = "completed" if failures == 0 else "completed_with_failures"
    state["failed_jobs"] = failures
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json(state_path, state)
    write_summary_html(run_root, state)
    logger.info("Pipeline finished with %d failed job(s)", failures)
    logger.info("Summary: %s", run_root / "summary.html")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same BAT again to resume from checkpoints.")
        raise SystemExit(130) from None
    except Exception as pipeline_error:
        print(f"FATAL: {pipeline_error}", file=sys.stderr)
        raise SystemExit(1) from pipeline_error
