"""
================================================================================
🤖 TELEGRAM AUTO-TRADING BOT
================================================================================
Multi-coin SPOT & FUTURES trading bot with 40 AI models.
Optimized for Raspberry Pi 4 (8GB RAM).

Features:
- Auto-trading with all 40 AI models (20 coins × 2 timeframes)
- SPOT and FUTURES trading support with live switching
- Telegram notifications for all trades
- CPU-only mode with LRU cache for models
- Testnet and mainnet support
- Position tracking per coin

Commands:
- /start - Start bot and show status
- /stop - Pause auto-trading
- /status - Show positions and P&L
- /spot - Switch to spot trading
- /futures - Switch to futures trading
- /help - Show available commands
================================================================================
"""

import os
import sys
import time
import json
import asyncio
import logging
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from decimal import Decimal, ROUND_DOWN
from collections import OrderedDict
from functools import lru_cache

import ccxt
import numpy as np
import pandas as pd
import torch

# Telegram
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# ============================================
# CONFIGURATION
# ============================================

# Supported coins (from kaggle_outputs)
SUPPORTED_COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "LINK", "LTC",
    "AVAX", "ATOM", "FIL", "TRX", "UNI", "MATIC", "APT", "ARB", "OP", "INJ"
]

TIMEFRAMES = ["15m", "1h"]

# Force CPU for Raspberry Pi
DEVICE = torch.device("cpu")


@dataclass
class BotConfig:
    """Bot configuration settings."""
    # API Settings (Testnet)
    api_key: str = ""
    api_secret: str = ""
    
    # API Settings (Real/Mainnet) - for switching between demo/real
    real_api_key: str = ""
    real_api_secret: str = ""
    
    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Trading Mode
    testnet: bool = True
    dry_run: bool = True
    trading_mode: str = "spot"  # "spot" or "futures"
    
    # Futures settings
    leverage: int = 5  # 1-125x leverage for futures
    margin_type: str = "isolated"  # "isolated" or "cross"
    
    # Multi-coin settings
    coins_to_trade: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    default_timeframe: str = "15m"
    max_positions: int = 5
    position_pct: float = 0.20  # 20% per position
    
    # Thresholds
    prediction_threshold: float = 0.003  # 0.30%
    prediction_scale: float = 0.5
    min_profit_to_exit: float = 0.0025
    stop_loss_pct: float = -0.05
    
    # Fees
    base_fee_rate: float = 0.001
    bnb_fee_rate: float = 0.00075
    use_bnb_for_fees: bool = True
    
    # Timing
    loop_interval_seconds: int = 60
    
    # Raspberry Pi optimization
    max_loaded_models: int = 5  # LRU cache size
    
    # Safety Limits (for real trading)
    daily_loss_limit_pct: float = -5.0  # Stop trading if daily loss exceeds -5%
    max_daily_trades: int = 20  # Maximum trades per day
    min_balance_usdt: float = 50.0  # Minimum balance to keep (don't trade below this)


# ============================================
# LOGGING
# ============================================

def setup_logging() -> logging.Logger:
    """Configure logging."""
    log_dir = PROJECT_ROOT / "bot_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"telegram_bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TelegramBot")


# ============================================
# LRU MODEL CACHE (Raspberry Pi Optimization)
# ============================================

class ModelCache:
    """LRU cache for models - keeps max N models in memory."""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.stats_cache: Dict = {}
        self.params_cache: Dict = {}
    
    def get_model(self, coin: str, timeframe: str) -> Optional[MultiBranchModel]:
        """Get model from cache, loading if necessary."""
        key = f"{coin}_{timeframe}"
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        # Load model
        model = self._load_model(coin, timeframe)
        if model is None:
            return None
        
        # Add to cache
        self.cache[key] = model
        self.cache.move_to_end(key)
        
        # Evict oldest if over capacity
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logging.info(f"🗑️ Evicted model from cache: {oldest_key}")
        
        return model
    
    def _load_model(self, coin: str, timeframe: str) -> Optional[MultiBranchModel]:
        """Load model from disk."""
        key = f"{coin}_{timeframe}"
        model_path = PROJECT_ROOT / "kaggle_outputs" / f"{key}_model.pth"
        params_path = PROJECT_ROOT / "kaggle_outputs" / f"{key}_params.json"
        stats_path = PROJECT_ROOT / "kaggle_outputs" / f"{key}_stats.json"
        
        if not model_path.exists():
            return None
        
        try:
            # Load params
            with open(params_path) as f:
                params = json.load(f)
            self.params_cache[key] = params
            
            # Load stats
            with open(stats_path) as f:
                stats = json.load(f)
            self.stats_cache[key] = stats
            
            # Create and load model
            model = MultiBranchModel(
                embed_dim=params.get('embed_dim', 96),
                dropout=params.get('dropout', 0.15)
            ).to(DEVICE)
            
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state)
            model.eval()
            
            logging.info(f"✅ Loaded model: {key}")
            return model
            
        except Exception as e:
            logging.error(f"❌ Failed to load model {key}: {e}")
            return None
    
    def get_stats(self, coin: str, timeframe: str) -> Optional[Dict]:
        """Get normalization stats for model."""
        key = f"{coin}_{timeframe}"
        if key not in self.stats_cache:
            # Try to load
            self.get_model(coin, timeframe)
        return self.stats_cache.get(key)


# ============================================
# POSITION TRACKING
# ============================================

@dataclass
class Position:
    """Tracks a single position."""
    coin: str
    side: str  # "LONG", "SHORT", or "FLAT"
    entry_price: float = 0.0
    amount: float = 0.0
    entry_time: Optional[datetime] = None
    entry_prediction: float = 0.0
    trade_id: Optional[str] = None  # Unique ID for tracking
    
    def pnl_pct(self, current_price: float) -> float:
        """Calculate P&L percentage."""
        if self.side == "FLAT" or self.entry_price == 0:
            return 0.0
        if self.side == "SHORT":
            # SHORT profits when price goes down
            return ((self.entry_price - current_price) / self.entry_price) * 100
        return ((current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class TradeRecord:
    """Stores a completed trade for history."""
    trade_id: str
    coin: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    amount: float
    entry_time: str
    exit_time: str
    pnl_pct: float
    pnl_usdt: float
    reason: str
    ai_prediction: float
    ai_confidence: float
    trading_mode: str  # "spot" or "futures"
    leverage: int = 1
    is_dry_run: bool = True


class TradeHistory:
    """Manages trade history with JSON persistence."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.trades: List[TradeRecord] = []
        self.load()
    
    def load(self):
        """Load trades from JSON file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**trade) for trade in data]
                logging.info(f"📜 Loaded {len(self.trades)} trades from history")
            except Exception as e:
                logging.error(f"Failed to load trade history: {e}")
                self.trades = []
        else:
            self.trades = []
    
    def save(self):
        """Save trades to JSON file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(trade) for trade in self.trades], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save trade history: {e}")
    
    def add_trade(self, trade: TradeRecord):
        """Add a new trade to history."""
        self.trades.append(trade)
        self.save()
        logging.info(f"📝 Trade saved: {trade.coin} {trade.side} ({trade.pnl_pct:+.2f}%)")
    
    def get_stats(self) -> Dict:
        """Calculate overall statistics."""
        if not self.trades:
            return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pnl_pct': 0, 'total_pnl_usdt': 0}
        
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        losses = sum(1 for t in self.trades if t.pnl_pct < 0)
        total_pnl_pct = sum(t.pnl_pct for t in self.trades)
        total_pnl_usdt = sum(t.pnl_usdt for t in self.trades)
        
        return {
            'total': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / len(self.trades)) * 100 if self.trades else 0,
            'total_pnl_pct': total_pnl_pct,
            'total_pnl_usdt': total_pnl_usdt
        }
    
    def get_recent(self, n: int = 10) -> List[TradeRecord]:
        """Get last N trades."""
        return self.trades[-n:] if len(self.trades) >= n else self.trades
    
    def export_csv(self, csv_path: Path):
        """Export trades to CSV."""
        try:
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if self.trades:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.trades[0]).keys())
                    writer.writeheader()
                    for trade in self.trades:
                        writer.writerow(asdict(trade))
            logging.info(f"📊 Exported {len(self.trades)} trades to {csv_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to export CSV: {e}")
            return False


# ============================================
# TELEGRAM AUTO-TRADING BOT
# ============================================

class TelegramAutoTradingBot:
    """
    Auto-trading bot with Telegram notifications.
    Optimized for Raspberry Pi 4 (8GB RAM).
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = setup_logging()
        self.model_cache = ModelCache(max_size=config.max_loaded_models)
        self.exchange: Optional[ccxt.binance] = None
        self.telegram_app: Optional[Application] = None
        
        # Positions per coin
        self.positions: Dict[str, Position] = {}
        for coin in SUPPORTED_COINS:
            self.positions[coin] = Position(coin=coin, side="FLAT")
        
        # Statistics
        self.cumulative_pnl_pct: float = 0.0
        self.total_trades: int = 0
        self.daily_trades: int = 0
        
        # Trade history
        history_file = PROJECT_ROOT / "trade_history" / "trades.json"
        self.trade_history = TradeHistory(history_file)
        
        # Control
        self.is_running: bool = False
        self.last_scan_time: Optional[datetime] = None
        
        # Pending trades (for tracking entry info until exit)
        self.pending_trades: Dict[str, Dict] = {}
        
        # Daily safety tracking
        self.daily_pnl_pct: float = 0.0
        self.safety_paused: bool = False
        self.safety_alert_sent: bool = False  # Prevent spam
        self.last_reset_date: Optional[datetime] = None
    
    # ========== TELEGRAM HANDLERS ==========
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        self.is_running = True
        
        mode_emoji = '🟢 SPOT' if self.config.trading_mode == 'spot' else '🟡 FUTURES'
        leverage_text = f" ({self.config.leverage}x)" if self.config.trading_mode == 'futures' else ""
        
        msg = f"""
🤖 <b>Crypto Auto-Trading Bot Aktif!</b>

📊 <b>Ayarlar:</b>
• Network: {'TESTNET' if self.config.testnet else 'MAINNET'}
• Trade Modu: {mode_emoji}{leverage_text}
• Dry Run: {'✓' if self.config.dry_run else '✗'}
• Coinler: {', '.join(self.config.coins_to_trade)}
• Timeframe: {self.config.default_timeframe}
• Max Pozisyon: {self.config.max_positions}

⏰ Bot artık otomatik trade yapacak!
📱 Tüm işlemler için bildirim alacaksınız.

Komutlar: /help
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
        self.logger.info("🚀 Bot started via Telegram")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command."""
        self.is_running = False
        await update.message.reply_text("⛔ Bot duraklatıldı. Tekrar başlatmak için /start")
        self.logger.info("⛔ Bot stopped via Telegram")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        # Get balances
        usdt_balance = self.get_balance("USDT")
        
        # Open positions
        open_positions = []
        for coin, pos in self.positions.items():
            if pos.side == "LONG":
                price = self.get_current_price(f"{coin}/USDT")
                pnl = pos.pnl_pct(price) if price else 0
                value = pos.amount * price if price else 0
                open_positions.append(f"  • {coin}: {pos.amount:.6f} @ ${pos.entry_price:,.2f} ({pnl:+.2f}%) = ${value:,.2f}")
        
        positions_text = "\n".join(open_positions) if open_positions else "  Açık pozisyon yok"
        
        msg = f"""
📊 <b>Bot Durumu</b>

💰 <b>Bakiye:</b> ${usdt_balance:,.2f} USDT

📈 <b>Açık Pozisyonlar:</b>
{positions_text}

📉 <b>İstatistikler:</b>
• Toplam Trade: {self.total_trades}
• Bugün: {self.daily_trades}
• Kümülatif P&L: {self.cumulative_pnl_pct:+.2f}%

🔄 Bot: {'✅ Çalışıyor' if self.is_running else '⏸️ Duraklatıldı'}
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        mode_text = f"{'SPOT' if self.config.trading_mode == 'spot' else 'FUTURES ' + str(self.config.leverage) + 'x'}"
        network_text = 'TESTNET' if self.config.testnet else 'MAINNET'
        dryrun_text = '✓ SİMÜLASYON' if self.config.dry_run else '✗ GERÇEK'
        
        msg = f"""
🤖 <b>Telegram Auto-Trading Bot</b>

📊 <b>Durum:</b> {network_text} | {mode_text} | {dryrun_text}

━━━━━━━━━━━━━━━━━━━━━━
<b>⚡ KONTROL</b>
━━━━━━━━━━━━━━━━━━━━━━
/start - Botu başlat
/stop - Botu duraklat
/status - Durum ve pozisyonlar

━━━━━━━━━━━━━━━━━━━━━━
<b>🔄 NETWORK MODU</b>
━━━━━━━━━━━━━━━━━━━━━━
/demo - Demo moda geç (testnet + simülasyon)
/real confirm - Gerçek moda geç ⚠️
/dryrun - Simülasyon AÇ (işlem yapmaz)
/dryrun off - Simülasyon KAPAT (testnet'te)

━━━━━━━━━━━━━━━━━━━━━━
<b>📈 TRADİNG MODU</b>
━━━━━━━━━━━━━━━━━━━━━━
/spot - Spot trading
/futures - Futures 5x
/futures 10 - Futures 10x

━━━━━━━━━━━━━━━━━━━━━━
<b>💰 COİN YÖNETİMİ</b>
━━━━━━━━━━━━━━━━━━━━━━
/coins - Aktif coinleri göster
/coins add BTC ETH - Coin ekle
/coins remove DOGE - Coin kaldır
/coins all - Tüm 20 coini aktif et

━━━━━━━━━━━━━━━━━━━━━━
<b>📜 GEÇMİŞ & GÜVENLİK</b>
━━━━━━━━━━━━━━━━━━━━━━
/history - Trade istatistikleri
/history export - CSV'ye aktar
/safety - Güvenlik durumu
/safety reset - Güvenlik kilidini sıfırla

━━━━━━━━━━━━━━━━━━━━━━
<b>✨ ÖZELLİKLER</b>
━━━━━━━━━━━━━━━━━━━━━━
• 40 AI model (20 coin × 2 timeframe)
• SPOT & FUTURES desteği
• Demo/Real mod geçişi
• Günlük kayıp limiti koruması
• Trade geçmişi kaydı
• Anlık Telegram bildirimleri
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
    
    async def cmd_spot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to spot trading mode."""
        if self.config.trading_mode == 'spot':
            await update.message.reply_text("🟢 Zaten SPOT modundasınız.")
            return
        
        # Check for open positions
        open_positions = sum(1 for p in self.positions.values() if p.side != "FLAT")
        if open_positions > 0:
            await update.message.reply_text(f"⚠️ Mod değiştirmek için önce {open_positions} açık pozisyonu kapatın.")
            return
        
        self.config.trading_mode = 'spot'
        self.initialize_exchange()  # Reinitialize for spot
        
        await update.message.reply_text("🟢 <b>SPOT</b> moduna geçildi!", parse_mode='HTML')
        await self.send_notification("🟢 Trading modu: <b>SPOT</b>")
        self.logger.info("🟢 Switched to SPOT mode")
    
    async def cmd_futures(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to futures trading mode."""
        # Parse leverage from args
        leverage = 5  # Default
        if context.args:
            try:
                leverage = int(context.args[0])
                leverage = max(1, min(125, leverage))  # Clamp 1-125
            except ValueError:
                pass
        
        if self.config.trading_mode == 'futures' and self.config.leverage == leverage:
            await update.message.reply_text(f"🟡 Zaten FUTURES {leverage}x modundasınız.")
            return
        
        # Check for open positions
        open_positions = sum(1 for p in self.positions.values() if p.side != "FLAT")
        if open_positions > 0:
            await update.message.reply_text(f"⚠️ Mod değiştirmek için önce {open_positions} açık pozisyonu kapatın.")
            return
        
        self.config.trading_mode = 'futures'
        self.config.leverage = leverage
        self.initialize_exchange()  # Reinitialize for futures
        
        await update.message.reply_text(f"🟡 <b>FUTURES {leverage}x</b> moduna geçildi!", parse_mode='HTML')
        await self.send_notification(f"🟡 Trading modu: <b>FUTURES {leverage}x</b>")
        self.logger.info(f"🟡 Switched to FUTURES {leverage}x mode")
    
    async def cmd_demo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to demo mode (testnet + dry run)."""
        # Check for open positions
        open_positions = sum(1 for p in self.positions.values() if p.side != "FLAT")
        if open_positions > 0:
            await update.message.reply_text(f"⚠️ Mod değiştirmek için önce {open_positions} açık pozisyonu kapatın.")
            return
        
        # Check if already in demo mode
        if self.config.testnet and self.config.dry_run:
            await update.message.reply_text("🎮 Zaten DEMO modundasınız.")
            return
        
        # Switch to demo mode
        self.config.testnet = True
        self.config.dry_run = True
        
        # Reinitialize exchange with testnet
        self.initialize_exchange()
        
        msg = """
🎮 <b>DEMO MODU AKTİF</b>

✅ Testnet: Açık
✅ Dry Run: Açık

⚠️ Gerçek işlem yapılmayacak!
📊 Simülasyon modunda çalışıyor.

Gerçek moda geçmek için: /real
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
        await self.send_notification("🎮 DEMO moduna geçildi - Simülasyon aktif")
        self.logger.info("🎮 Switched to DEMO mode (testnet + dry_run)")
    
    async def cmd_real(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to real trading mode (mainnet + live orders)."""
        # Check for open positions
        open_positions = sum(1 for p in self.positions.values() if p.side != "FLAT")
        if open_positions > 0:
            await update.message.reply_text(f"⚠️ Mod değiştirmek için önce {open_positions} açık pozisyonu kapatın.")
            return
        
        # Check if real API keys are configured
        if not self.config.real_api_key or not self.config.real_api_secret:
            await update.message.reply_text("❌ Gerçek API anahtarları ayarlanmamış!\n\n.env dosyasına REAL_API_KEY ve REAL_API_SECRET ekleyin.")
            return
        
        # Already in real mode?
        if not self.config.testnet and not self.config.dry_run:
            await update.message.reply_text("💰 Zaten GERÇEK moddasınız!")
            return
        
        # Require confirmation
        if not context.args or context.args[0].lower() != 'confirm':
            msg = """
⚠️ <b>DİKKAT: GERÇEK PARA!</b>

Bu komut gerçek Binance hesabınızla işlem yapacak!
Gerçek paranızı kaybedebilirsiniz!

Onaylamak için yazın:
<code>/real confirm</code>
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            return
        
        # Switch to real mode
        self.config.testnet = False
        self.config.dry_run = False
        self.config.api_key = self.config.real_api_key
        self.config.api_secret = self.config.real_api_secret
        
        # Reinitialize exchange with real credentials
        try:
            success = self.initialize_exchange()
            if not success:
                raise Exception("Exchange initialization returned False")
        except Exception as e:
            # Rollback on failure
            self.config.testnet = True
            self.config.dry_run = True
            error_msg = str(e)
            await update.message.reply_text(f"❌ Gerçek hesaba bağlanılamadı!\n\n<b>Hata:</b> <code>{error_msg}</code>\n\nDemo moduna geri dönüldü.", parse_mode='HTML')
            self.logger.error(f"Real mode connection failed: {e}")
            return
        
        msg = """
💰 <b>GERÇEK MOD AKTİF!</b>

⚠️ Testnet: KAPALI
⚠️ Dry Run: KAPALI

🚨 <b>GERÇEK PARA İLE İŞLEM YAPILIYOR!</b>

Demo moda geçmek için: /demo
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
        await self.send_notification("🚨 GERÇEK MODA GEÇİLDİ - Dikkatli olun!")
        self.logger.warning("🚨 SWITCHED TO REAL TRADING MODE!")
    
    async def cmd_dryrun(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle dry run mode on/off without changing network."""
        # Check for open positions
        open_positions = sum(1 for p in self.positions.values() if p.side != "FLAT")
        if open_positions > 0:
            await update.message.reply_text(f"⚠️ Mod değiştirmek için önce {open_positions} açık pozisyonu kapatın.")
            return
        
        # Toggle dry run
        if context.args and context.args[0].lower() == 'off':
            # Turn OFF dry run (enable real trading)
            if not self.config.testnet:
                # On mainnet, require confirmation
                await update.message.reply_text("⚠️ Gerçek trading için /real confirm kullanın.")
                return
            self.config.dry_run = False
            msg = """
⚠️ <b>DRY RUN KAPALI</b>

Artık gerçek işlemler yapılacak!
Network: {}

Simülasyona dönmek için: /dryrun on
""".format('TESTNET' if self.config.testnet else 'MAINNET')
        else:
            # Turn ON dry run (simulation mode)
            self.config.dry_run = True
            msg = """
🎮 <b>DRY RUN AÇIK</b>

Simülasyon modu aktif - İşlem yapılmayacak.
Network: {}

Gerçek işlem için:
• Testnet: /dryrun off
• Mainnet: /real confirm
""".format('TESTNET' if self.config.testnet else 'MAINNET')
        
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
        self.logger.info(f"Dry run mode: {self.config.dry_run}")
    
    async def cmd_coins(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manage trading coins: /coins, /coins add BTC, /coins remove ETH"""
        args = context.args
        
        # No args - show current coins
        if not args:
            current = ', '.join(self.config.coins_to_trade)
            available = ', '.join([c for c in SUPPORTED_COINS if c not in self.config.coins_to_trade])
            msg = f"""
💰 <b>Coin Ayarları</b>

<b>Aktif Coinler:</b>
{current}

<b>Eklenebilir:</b>
{available}

<b>Kullanım:</b>
/coins add BTC ETH SOL
/coins remove DOGE
/coins set BTC,ETH,SOL
/coins all
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            return
        
        action = args[0].lower()
        
        # Add coins
        if action == 'add' and len(args) > 1:
            added = []
            for coin in args[1:]:
                coin = coin.upper().replace(',', '')
                if coin in SUPPORTED_COINS and coin not in self.config.coins_to_trade:
                    self.config.coins_to_trade.append(coin)
                    added.append(coin)
            if added:
                await update.message.reply_text(f"✅ Eklendi: {', '.join(added)}")
            else:
                await update.message.reply_text("⚠️ Eklenecek geçerli coin bulunamadı.")
            return
        
        # Remove coins
        if action == 'remove' and len(args) > 1:
            removed = []
            for coin in args[1:]:
                coin = coin.upper().replace(',', '')
                if coin in self.config.coins_to_trade:
                    # Check if position is open
                    if self.positions.get(coin) and self.positions[coin].side != "FLAT":
                        await update.message.reply_text(f"⚠️ {coin} açık pozisyonu var, önce kapatın.")
                        continue
                    self.config.coins_to_trade.remove(coin)
                    removed.append(coin)
            if removed:
                await update.message.reply_text(f"✅ Kaldırıldı: {', '.join(removed)}")
            else:
                await update.message.reply_text("⚠️ Kaldırılacak coin bulunamadı.")
            return
        
        # Set specific coins
        if action == 'set' and len(args) > 1:
            coins_str = ' '.join(args[1:])
            new_coins = [c.strip().upper() for c in coins_str.replace(',', ' ').split()]
            new_coins = [c for c in new_coins if c in SUPPORTED_COINS]
            if new_coins:
                self.config.coins_to_trade = new_coins
                await update.message.reply_text(f"✅ Coinler ayarlandı: {', '.join(new_coins)}")
            else:
                await update.message.reply_text("⚠️ Geçerli coin bulunamadı.")
            return
        
        # All coins
        if action == 'all':
            self.config.coins_to_trade = SUPPORTED_COINS.copy()
            await update.message.reply_text(f"✅ Tüm {len(SUPPORTED_COINS)} coin aktif.")
            return
        
        await update.message.reply_text("❌ Geçersiz komut. /coins yazarak kullanımı görün.")
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command - Show trade history and stats."""
        stats = self.trade_history.get_stats()
        recent = self.trade_history.get_recent(5)
        
        # Build recent trades text
        if recent:
            trades_text = ""
            for trade in reversed(recent):  # Most recent first
                emoji = "🟢" if trade.pnl_pct >= 0 else "🔴"
                trades_text += f"  {emoji} {trade.coin}: {trade.pnl_pct:+.2f}% (${trade.pnl_usdt:+.2f})\n"
        else:
            trades_text = "  Henüz trade yok\n"
        
        # Win/loss emoji
        if stats['total'] > 0:
            wr_emoji = "🟢" if stats['win_rate'] >= 50 else "🔴"
        else:
            wr_emoji = "⚪"
        
        pnl_emoji = "🟢" if stats['total_pnl_pct'] >= 0 else "🔴"
        
        msg = f"""
📜 <b>Trade History</b>

📊 <b>İstatistikler:</b>
• Toplam Trade: {stats['total']}
• ✅ Kazanan: {stats['wins']}
• ❌ Kaybeden: {stats['losses']}
• {wr_emoji} Win Rate: {stats['win_rate']:.1f}%
• {pnl_emoji} Toplam P&L: {stats['total_pnl_pct']:+.2f}% (${stats['total_pnl_usdt']:+.2f})

📋 <b>Son 5 Trade:</b>
{trades_text}
<b>Komutlar:</b>
/history export - CSV'ye aktar
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
    
    async def cmd_history_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history export command."""
        if context.args and context.args[0].lower() == 'export':
            csv_path = PROJECT_ROOT / "trade_history" / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            if self.trade_history.export_csv(csv_path):
                await update.message.reply_text(f"✅ CSV dosyası oluşturuldu:\n{csv_path}")
            else:
                await update.message.reply_text("❌ CSV dosyası oluşturulamadı.")
        else:
            await self.cmd_history(update, context)
    
    async def cmd_safety(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /safety command - Show safety status and limits."""
        usdt_balance = self.get_balance("USDT")
        can_trade, safety_msg = self.check_safety_limits(usdt_balance)
        
        # Status emojis
        status_emoji = "🟢" if can_trade else "🔴"
        pause_emoji = "🔴 DURDURULDU" if self.safety_paused else "🟢 AKTİF"
        
        # Daily stats
        daily_pnl_emoji = "🟢" if self.daily_pnl_pct >= 0 else "🔴"
        loss_limit_pct = (self.daily_pnl_pct / self.config.daily_loss_limit_pct * 100) if self.config.daily_loss_limit_pct != 0 else 0
        loss_limit_pct = max(0, min(100, loss_limit_pct))
        
        # Progress bars
        trade_bar = self._progress_bar(self.daily_trades, self.config.max_daily_trades)
        loss_bar = self._progress_bar(loss_limit_pct, 100)
        
        msg = f"""
🛡️ <b>Güvenlik Durumu</b>

{status_emoji} <b>Durum:</b> {safety_msg}

📊 <b>Günlük İstatistikler:</b>
• {daily_pnl_emoji} Günlük P&L: {self.daily_pnl_pct:+.2f}%
• Trade Sayısı: {self.daily_trades}/{self.config.max_daily_trades}
  {trade_bar}
• Kayıp Limiti: {loss_limit_pct:.0f}%
  {loss_bar}

⚙️ <b>Limitler:</b>
• Günlük Kayıp Limiti: {self.config.daily_loss_limit_pct:.1f}%
• Max Günlük Trade: {self.config.max_daily_trades}
• Min Bakiye: ${self.config.min_balance_usdt:.0f}

💰 <b>Mevcut Bakiye:</b> ${usdt_balance:,.2f}
🔄 <b>Trading:</b> {pause_emoji}

<b>Komutlar:</b>
/safety reset - Güvenlik kilidini sıfırla
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
    
    def _progress_bar(self, current: float, maximum: float, length: int = 10) -> str:
        """Generate a text progress bar."""
        if maximum <= 0:
            return "░" * length
        ratio = min(1.0, max(0.0, current / maximum))
        filled = int(ratio * length)
        return "█" * filled + "░" * (length - filled)
    
    async def cmd_safety_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /safety reset command."""
        if context.args and context.args[0].lower() == 'reset':
            self.safety_paused = False
            self.daily_pnl_pct = 0.0
            await update.message.reply_text("✅ Güvenlik kilidi sıfırlandı. Trading tekrar aktif.")
            await self.send_notification("🔓 Güvenlik kilidi manuel olarak sıfırlandı.")
            self.logger.info("🔓 Safety lock manually reset")
        else:
            await self.cmd_safety(update, context)
    
    # ========== NOTIFICATIONS ==========
    
    async def send_notification(self, message: str):
        """Send a message to Telegram."""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return
        
        try:
            bot = Bot(token=self.config.telegram_bot_token)
            await bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            self.logger.error(f"Telegram error: {e}")
    
    async def send_buy_alert(self, coin: str, entry_price: float, amount: float,
                             prediction_pct: float, confidence: float):
        """Send buy notification."""
        value = entry_price * amount
        target = entry_price * (1 + self.config.prediction_threshold)
        stop = entry_price * (1 + self.config.stop_loss_pct)
        
        msg = f"""
🟢 <b>{coin} ALIŞ</b>

📊 <b>İşlem:</b>
• Miktar: {amount:.6f} {coin}
• Değer: ${value:,.2f}
• AI Tahmin: {prediction_pct:+.2f}%
• Güven: {confidence:.0f}%

📍 <b>Seviyeler:</b>
• Giriş: ${entry_price:,.2f}
• Hedef: ${target:,.2f}
• Stop: ${stop:,.2f}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_notification(msg.strip())
    
    async def send_sell_alert(self, coin: str, exit_price: float, amount: float,
                              entry_price: float, pnl_pct: float, reason: str):
        """Send sell notification."""
        value = exit_price * amount
        profit = (exit_price - entry_price) * amount
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        
        msg = f"""
🔴 <b>{coin} SATIŞ</b>

📊 <b>İşlem:</b>
• Miktar: {amount:.6f} {coin}
• Giriş: ${entry_price:,.2f}
• Çıkış: ${exit_price:,.2f}

{emoji} <b>P&L:</b> {pnl_pct:+.2f}% (${profit:+,.2f})
📝 Sebep: {reason}

📈 <b>Toplam:</b>
• Trade: {self.total_trades}
• Kümülatif: {self.cumulative_pnl_pct:+.2f}%

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_notification(msg.strip())
    
    async def send_daily_summary(self):
        """Send daily P&L summary."""
        msg = f"""
📊 <b>Günlük Özet</b>

• İşlem Sayısı: {self.daily_trades}
• Kümülatif P&L: {self.cumulative_pnl_pct:+.2f}%

Açık Pozisyonlar: {sum(1 for p in self.positions.values() if p.side == "LONG")}
"""
        await self.send_notification(msg.strip())
    
    # ========== EXCHANGE ==========
    
    def initialize_exchange(self) -> bool:
        """Initialize Binance connection for spot or futures."""
        try:
            is_futures = self.config.trading_mode == 'futures'
            
            params = {
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if is_futures else 'spot'
                }
            }
            
            if self.config.testnet:
                params['options']['sandboxMode'] = True
                if is_futures:
                    # Binance Futures testnet
                    params['urls'] = {
                        'api': {
                            'public': 'https://testnet.binancefuture.com',
                            'private': 'https://testnet.binancefuture.com',
                        }
                    }
                else:
                    # Binance Spot testnet
                    params['urls'] = {
                        'api': {
                            'public': 'https://testnet.binance.vision/api',
                            'private': 'https://testnet.binance.vision/api',
                        }
                    }
            
            self.exchange = ccxt.binance(params)
            
            if self.config.testnet:
                self.exchange.set_sandbox_mode(True)
            
            self.exchange.load_markets()
            
            # Set leverage for futures
            if is_futures:
                for coin in self.config.coins_to_trade:
                    try:
                        symbol = f"{coin}/USDT:USDT"
                        self.exchange.set_leverage(self.config.leverage, symbol)
                        self.exchange.set_margin_mode(self.config.margin_type, symbol)
                    except Exception as e:
                        self.logger.warning(f"Could not set leverage for {coin}: {e}")
            
            mode_str = f"{'FUTURES ' + str(self.config.leverage) + 'x' if is_futures else 'SPOT'}"
            net_str = 'Testnet' if self.config.testnet else 'Mainnet'
            self.logger.info(f"✅ Connected to Binance {net_str} ({mode_str})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect: {e}")
            raise  # Re-raise so calling code can see the error
    
    def get_balance(self, asset: str) -> float:
        """Get wallet balance."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(asset, {}).get('free', 0))
        except Exception as e:
            self.logger.error(f"Balance error: {e}")
            return 0.0
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"Price error for {symbol}: {e}")
            return None
    
    # ========== PREDICTION ==========
    
    def get_prediction(self, coin: str, timeframe: str) -> Optional[Dict]:
        """Get AI prediction for a coin."""
        try:
            # Get model
            model = self.model_cache.get_model(coin, timeframe)
            if model is None:
                return None
            
            stats = self.model_cache.get_stats(coin, timeframe)
            if stats is None:
                return None
            
            # Fetch data
            tf_minutes = {'15m': 15, '1h': 60}.get(timeframe, 15)
            months_needed = (500 * tf_minutes / (30 * 24 * 60)) * 1.1
            
            df = get_crypto_history(
                symbol=f"{coin}/USDT",
                timeframe=timeframe,
                months_back=months_needed,
                exchange_name="binance"
            )
            
            if len(df) < 120:
                return None
            
            # Prepare data
            df_display, df_ai = prepare_dual_dataframes(df)
            
            if df_ai.isnull().values.any() or np.isinf(df_ai.values).any():
                return None
            
            # Get column indices
            cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
            lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
            tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
            
            # Normalize using model-specific stats
            mean = pd.Series(stats['mean'])
            std = pd.Series(stats['std'])
            std[std == 0] = 1.0
            
            df_normalized = (df_ai - mean) / std
            data = df_normalized.values
            t = len(data)
            
            # Prepare tensors
            x_cnn = torch.tensor(data[t-12:t, cnn_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_lstm = torch.tensor(data[t-120:t, lstm_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_tr = torch.tensor(data[t-120:t, tr_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Run prediction
            with torch.no_grad():
                pred_main, pred_cnn, pred_lstm, pred_tr = model(x_cnn, x_lstm, x_tr)
                prediction = pred_main.item() / 100.0
                
                # Confidence from branch agreement
                branches = [pred_cnn.item()/100.0, pred_lstm.item()/100.0, pred_tr.item()/100.0]
                signs = [1 if b > 0 else -1 for b in branches]
                confidence = abs(sum(signs)) / 3.0 * 100
            
            current_price = float(df_display.iloc[-1]['Close'])
            
            return {
                'coin': coin,
                'timeframe': timeframe,
                'prediction': prediction,
                'prediction_pct': prediction * 100,
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error for {coin}: {e}")
            return None
    
    # ========== SAFETY CHECKS ==========
    
    def check_safety_limits(self, usdt_balance: float) -> Tuple[bool, str]:
        """Check if trading should be allowed based on safety limits."""
        # Check if safety paused
        if self.safety_paused:
            return (False, "🛑 Günlük kayıp limiti aşıldı - trading durduruldu")
        
        # Check daily loss limit
        if self.daily_pnl_pct <= self.config.daily_loss_limit_pct:
            self.safety_paused = True
            return (False, f"🛑 Günlük kayıp limiti: {self.daily_pnl_pct:.2f}%")
        
        # Check max daily trades
        if self.daily_trades >= self.config.max_daily_trades:
            return (False, f"📊 Günlük trade limiti: {self.daily_trades}/{self.config.max_daily_trades}")
        
        # Check minimum balance
        if usdt_balance < self.config.min_balance_usdt:
            return (False, f"💰 Minimum bakiye: ${usdt_balance:.2f} < ${self.config.min_balance_usdt:.2f}")
        
        return (True, "✅ OK")
    
    def reset_daily_stats(self):
        """Reset daily statistics at midnight."""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_trades = 0
            self.daily_pnl_pct = 0.0
            self.safety_paused = False
            self.safety_alert_sent = False
            self.last_reset_date = today
            self.logger.info("📅 Günlük istatistikler sıfırlandı")
    
    async def send_safety_alert(self, reason: str):
        """Send safety limit alert to Telegram."""
        msg = f"""
🚨 <b>GÜVENLİK UYARISI</b>

{reason}

📊 <b>Günlük Durum:</b>
• P&L: {self.daily_pnl_pct:+.2f}%
• Trade: {self.daily_trades}/{self.config.max_daily_trades}

⚠️ Trading otomatik olarak durduruldu.
✅ Sıfırlamak için: /safety reset

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_notification(msg.strip())
    
    # ========== TRADING LOGIC ==========
    
    def should_trade(self, coin: str, prediction: Dict) -> Tuple[str, str]:
        """Decide whether to trade."""
        pred = prediction['prediction']
        price = prediction['price']
        threshold = self.config.prediction_threshold
        pos = self.positions[coin]
        
        # Count open positions
        open_count = sum(1 for p in self.positions.values() if p.side == "LONG")
        
        # SELL logic
        if pos.side == "LONG":
            current_pnl = pos.pnl_pct(price) / 100  # Convert to decimal
            
            # Stop-loss
            if current_pnl <= self.config.stop_loss_pct:
                return ('SELL', f'🛑 Stop-Loss: {current_pnl*100:.2f}%')
            
            # Take profit
            if pos.entry_prediction and current_pnl >= pos.entry_prediction:
                return ('SELL', f'🎯 Hedef: {current_pnl*100:+.2f}%')
            
            # Bearish + min profit
            if pred < 0 and current_pnl >= self.config.min_profit_to_exit:
                return ('SELL', f'📉 Bearish + kâr: {current_pnl*100:+.2f}%')
            
            # Still bullish
            if pred > threshold:
                return ('HOLD', 'Hala yükseliş sinyali')
            
            return ('HOLD', 'Bekleniyor')
        
        # BUY logic
        if pos.side == "FLAT":
            if open_count >= self.config.max_positions:
                return ('HOLD', 'Max pozisyon limiti')
            
            if pred > threshold:
                return ('BUY', f'📈 Yükseliş: +{pred*100:.3f}%')
        
        return ('HOLD', 'Sinyal yok')
    
    async def execute_buy(self, coin: str, usdt_amount: float, prediction: Dict) -> bool:
        """Execute buy order."""
        try:
            symbol = f"{coin}/USDT"
            price = self.get_current_price(symbol)
            if price is None:
                return False
            
            # Calculate amount
            fee_rate = self.config.bnb_fee_rate if self.config.use_bnb_for_fees else self.config.base_fee_rate
            fee = usdt_amount * fee_rate
            net_usdt = usdt_amount - fee
            amount = net_usdt / price
            
            # Round
            amount = float(Decimal(str(amount)).quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
            
            self.logger.info(f"📈 BUY {coin}: {amount:.6f} @ ${price:,.2f}")
            
            if not self.config.dry_run:
                self.exchange.create_market_buy_order(symbol, amount)
            
            # Update position
            pos = self.positions[coin]
            pos.side = "LONG"
            pos.entry_price = price
            pos.amount = amount
            pos.entry_time = datetime.now()
            pos.entry_prediction = prediction['prediction'] * self.config.prediction_scale
            pos.trade_id = str(uuid.uuid4())[:8]  # Short unique ID
            
            # Store pending trade info for history
            self.pending_trades[coin] = {
                'trade_id': pos.trade_id,
                'entry_time': pos.entry_time.isoformat(),
                'ai_prediction': prediction['prediction_pct'],
                'ai_confidence': prediction['confidence']
            }
            
            self.daily_trades += 1
            self.total_trades += 1
            
            # Notify
            await self.send_buy_alert(
                coin=coin,
                entry_price=price,
                amount=amount,
                prediction_pct=prediction['prediction_pct'],
                confidence=prediction['confidence']
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Buy error for {coin}: {e}")
            return False
    
    async def execute_sell(self, coin: str, reason: str) -> bool:
        """Execute sell order."""
        try:
            pos = self.positions[coin]
            if pos.side != "LONG":
                return False
            
            symbol = f"{coin}/USDT"
            price = self.get_current_price(symbol)
            if price is None:
                return False
            
            amount = pos.amount
            pnl_pct = pos.pnl_pct(price)
            
            self.logger.info(f"📉 SELL {coin}: {amount:.6f} @ ${price:,.2f} ({pnl_pct:+.2f}%)")
            
            if not self.config.dry_run:
                self.exchange.create_market_sell_order(symbol, amount)
            
            # Update stats
            self.cumulative_pnl_pct += pnl_pct
            self.daily_pnl_pct += pnl_pct  # Track daily P&L for safety limits
            self.daily_trades += 1
            self.total_trades += 1
            
            # Notify
            await self.send_sell_alert(
                coin=coin,
                exit_price=price,
                amount=amount,
                entry_price=pos.entry_price,
                pnl_pct=pnl_pct,
                reason=reason
            )
            
            # Save trade to history
            pending = self.pending_trades.get(coin, {})
            pnl_usdt = (price - pos.entry_price) * amount
            
            trade_record = TradeRecord(
                trade_id=pending.get('trade_id', pos.trade_id or str(uuid.uuid4())[:8]),
                coin=coin,
                side="COMPLETE",
                entry_price=pos.entry_price,
                exit_price=price,
                amount=amount,
                entry_time=pending.get('entry_time', pos.entry_time.isoformat() if pos.entry_time else ''),
                exit_time=datetime.now().isoformat(),
                pnl_pct=pnl_pct,
                pnl_usdt=pnl_usdt,
                reason=reason,
                ai_prediction=pending.get('ai_prediction', pos.entry_prediction * 100),
                ai_confidence=pending.get('ai_confidence', 0),
                trading_mode=self.config.trading_mode,
                leverage=self.config.leverage if self.config.trading_mode == 'futures' else 1,
                is_dry_run=self.config.dry_run
            )
            self.trade_history.add_trade(trade_record)
            
            # Clear pending trade
            if coin in self.pending_trades:
                del self.pending_trades[coin]
            
            # Reset position
            pos.side = "FLAT"
            pos.entry_price = 0.0
            pos.amount = 0.0
            pos.entry_time = None
            pos.entry_prediction = 0.0
            pos.trade_id = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sell error for {coin}: {e}")
            return False
    
    # ========== MAIN LOOP ==========
    
    async def trading_loop(self):
        """Main auto-trading loop."""
        self.logger.info("🔄 Starting trading loop...")
        
        last_candle_minute = -1
        
        while True:
            try:
                if not self.is_running:
                    await asyncio.sleep(5)
                    continue
                
                now = datetime.now()
                tf = self.config.default_timeframe
                tf_minutes = {'15m': 15, '1h': 60}.get(tf, 15)
                candle_minute = (now.minute // tf_minutes) * tf_minutes
                
                # Reset daily stats at midnight
                self.reset_daily_stats()
                if now.hour == 0 and now.minute == 0:
                    await self.send_daily_summary()
                
                # Execute at start of new candle
                if candle_minute != last_candle_minute and now.minute % tf_minutes < 1:
                    last_candle_minute = candle_minute
                    
                    self.logger.info(f"{'='*40}")
                    self.logger.info(f"🕐 Yeni {tf} mum: {now.strftime('%H:%M')}")
                    
                    usdt_balance = self.get_balance("USDT")
                    self.logger.info(f"💰 USDT: ${usdt_balance:,.2f}")
                    
                    # Check safety limits before trading
                    can_trade, safety_msg = self.check_safety_limits(usdt_balance)
                    if not can_trade:
                        self.logger.warning(f"⚠️ {safety_msg}")
                        # Send alert only once
                        if not self.safety_alert_sent:
                            await self.send_safety_alert(safety_msg)
                            self.safety_alert_sent = True
                        await asyncio.sleep(self.config.loop_interval_seconds)
                        continue
                    
                    self.logger.info(f"📊 Günlük P&L: {self.daily_pnl_pct:+.2f}% | Trades: {self.daily_trades}")
                    
                    # Scan all configured coins
                    for coin in self.config.coins_to_trade:
                        prediction = self.get_prediction(coin, tf)
                        if prediction is None:
                            continue
                        
                        self.logger.info(f"🧠 {coin}: {prediction['prediction_pct']:+.3f}% (güven: {prediction['confidence']:.0f}%)")
                        
                        action, reason = self.should_trade(coin, prediction)
                        self.logger.info(f"   → {action}: {reason}")
                        
                        if action == 'BUY':
                            open_count = sum(1 for p in self.positions.values() if p.side == "LONG")
                            trade_amount = (usdt_balance / (self.config.max_positions - open_count)) * self.config.position_pct
                            
                            if trade_amount > 10:
                                await self.execute_buy(coin, trade_amount, prediction)
                        
                        elif action == 'SELL':
                            await self.execute_sell(coin, reason)
                
                await asyncio.sleep(self.config.loop_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                await asyncio.sleep(60)
    
    async def run(self):
        """Start the bot."""
        self.logger.info("=" * 50)
        self.logger.info("🤖 Telegram Auto-Trading Bot")
        self.logger.info(f"   Mod: {'TESTNET' if self.config.testnet else 'MAINNET'}")
        self.logger.info(f"   Coinler: {', '.join(self.config.coins_to_trade)}")
        self.logger.info("=" * 50)
        
        # Initialize exchange
        if not self.initialize_exchange():
            return
        
        # Build Telegram app with longer timeout for Raspberry Pi
        from telegram.request import HTTPXRequest
        request = HTTPXRequest(
            connect_timeout=30.0,
            read_timeout=30.0,
            write_timeout=30.0,
            pool_timeout=30.0
        )
        self.telegram_app = Application.builder().token(self.config.telegram_bot_token).request(request).build()
        
        # Add handlers
        self.telegram_app.add_handler(CommandHandler("start", self.cmd_start))
        self.telegram_app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.telegram_app.add_handler(CommandHandler("status", self.cmd_status))
        self.telegram_app.add_handler(CommandHandler("help", self.cmd_help))
        self.telegram_app.add_handler(CommandHandler("spot", self.cmd_spot))
        self.telegram_app.add_handler(CommandHandler("futures", self.cmd_futures))
        self.telegram_app.add_handler(CommandHandler("coins", self.cmd_coins))
        self.telegram_app.add_handler(CommandHandler("history", self.cmd_history_export))
        self.telegram_app.add_handler(CommandHandler("safety", self.cmd_safety_reset))
        self.telegram_app.add_handler(CommandHandler("demo", self.cmd_demo))
        self.telegram_app.add_handler(CommandHandler("real", self.cmd_real))
        self.telegram_app.add_handler(CommandHandler("dryrun", self.cmd_dryrun))
        
        # Start Telegram
        await self.telegram_app.initialize()
        await self.telegram_app.start()
        await self.telegram_app.updater.start_polling()
        
        # Send startup message
        await self.send_notification(f"🤖 Bot başlatıldı!\n\nMod: {'TESTNET' if self.config.testnet else 'MAINNET'}\nCoinler: {', '.join(self.config.coins_to_trade)}\n\n/start ile aktif edin.")
        
        self.logger.info("✅ Telegram active, starting trading loop...")
        
        # Run trading loop
        try:
            await self.trading_loop()
        finally:
            await self.telegram_app.updater.stop()
            await self.telegram_app.stop()
            await self.telegram_app.shutdown()


# ============================================
# ENV LOADER
# ============================================

def load_env(env_path: Path) -> Dict[str, str]:
    """Load .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


# ============================================
# MAIN
# ============================================

def main():
    """Entry point."""
    print("=" * 60)
    print("  🤖 Telegram Auto-Trading Bot")
    print("  Multi-Coin | 40 AI Models | Raspberry Pi Optimized")
    print("=" * 60)
    
    # Load config
    env_path = PROJECT_ROOT / '.env'
    env = load_env(env_path)
    
    # Parse coins
    coins_str = env.get('AUTO_TRADE_COINS', 'BTC,ETH,SOL')
    if coins_str.lower() == 'all':
        coins = SUPPORTED_COINS
    else:
        coins = [c.strip().upper() for c in coins_str.split(',')]
        coins = [c for c in coins if c in SUPPORTED_COINS]
    
    config = BotConfig(
        api_key=env.get('BINANCE_API_KEY', ''),
        api_secret=env.get('BINANCE_API_SECRET', ''),
        real_api_key=env.get('REAL_API_KEY', ''),
        real_api_secret=env.get('REAL_API_SECRET', ''),
        telegram_bot_token=env.get('TELEGRAM_BOT_TOKEN', ''),
        telegram_chat_id=env.get('TELEGRAM_CHAT_ID', ''),
        testnet=env.get('TESTNET', 'true').lower() == 'true',
        dry_run=env.get('DRY_RUN', 'true').lower() == 'true',
        trading_mode=env.get('TRADING_MODE', 'spot').lower(),
        leverage=int(env.get('LEVERAGE', '5')),
        margin_type=env.get('MARGIN_TYPE', 'isolated').lower(),
        coins_to_trade=coins,
        default_timeframe=env.get('DEFAULT_TIMEFRAME', '15m'),
        max_positions=int(env.get('MAX_POSITIONS', '5')),
        position_pct=float(env.get('POSITION_PCT', '0.20')),
        prediction_threshold=float(env.get('PREDICTION_THRESHOLD', '0.003')),
        prediction_scale=float(env.get('PREDICTION_SCALE', '0.5')),
        min_profit_to_exit=float(env.get('MIN_PROFIT_TO_EXIT', '0.0025')),
        stop_loss_pct=float(env.get('STOP_LOSS_PCT', '-0.05')),
        max_loaded_models=int(env.get('MAX_LOADED_MODELS', '5')),
        # Safety limits
        daily_loss_limit_pct=float(env.get('DAILY_LOSS_LIMIT_PCT', '-5.0')),
        max_daily_trades=int(env.get('MAX_DAILY_TRADES', '20')),
        min_balance_usdt=float(env.get('MIN_BALANCE_USDT', '50.0'))
    )
    
    # Check required settings
    if not config.telegram_bot_token or config.telegram_bot_token == 'your_bot_token_here':
        print("\n❌ TELEGRAM_BOT_TOKEN ayarlanmamış!")
        print("   @BotFather'dan token alın ve .env dosyasına ekleyin.")
        return
    
    if not config.telegram_chat_id or config.telegram_chat_id == 'your_chat_id_here':
        print("\n❌ TELEGRAM_CHAT_ID ayarlanmamış!")
        print("   @userinfobot'tan ID'nizi alın ve .env dosyasına ekleyin.")
        return
    
    # Run bot
    bot = TelegramAutoTradingBot(config)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
