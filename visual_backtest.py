"""
================================================================================
🎮 VISUAL BACKTEST SIMULATOR - GERÇEK AI MODELİ
================================================================================
Gerçek uygulama gibi trade yapan, animasyonlu grafik simülatörü.
Tüm kurallar (saat filtresi, Fear&Greed, akıllı çıkış vb.) aktif.
GERÇEK AI MODELİ KULLANIR!

Kullanım:
    python visual_backtest.py --coin BTC --days 7

Gereksinimler:
    pip install matplotlib ccxt pandas numpy torch

# BTC son 7 gün, animasyonlu
python visual_backtest.py --coin BTC --days 7
# Hızlı animasyon (50ms)
python visual_backtest.py --coin ETH --days 14 --speed 50
# Animasyonsuz (sadece sonuçlar)
python visual_backtest.py --coin SOL --days 30 --no-anim
# Saat filtresi kapalı
python visual_backtest.py --coin BTC --days 7 --no-hour-filter

# SPOT modu (kaldıraçsız, sadece LONG)
python visual_backtest.py --coin BTC --days 30 --mode spot
# SPOT + Grid (kademeli alım)
python visual_backtest.py --coin BTC --days 30 --mode spot --grid --grid-levels 2

python visual_backtest.py --coin LTC --days 90 --meta-compare --meta-threshold 0.50 --speed 150 --no-hour-filter
================================================================================
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import ccxt
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

# ========== PATH SETUP ==========
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES
from meta_model_gate import MetaModelGate

DEVICE = torch.device("cpu")

# ========== CONFIGURATION ==========
@dataclass
class SimConfig:
    """Simülasyon yapılandırması - gerçek botla aynı parametreler"""
    # Trading Mode
    trading_mode: str = "futures"  # "spot" or "futures"
    
    # Trading settings
    leverage: int = 5  # Only used in futures mode
    initial_balance: float = 100.0 
    position_pct: float = 0.95
    max_positions: int = 1
    
    # Thresholds
    prediction_threshold: float = 0.003  # 0.3%
    min_confidence_threshold: float = 25.0
    min_prediction_pct: float = 10.0  # %10 effective move
    prediction_scale: float = 0.5
    tp_multiplier: float = 0.80
    
    # Spot SL/TP (ROE %)
    spot_sl_pct: float = -5.0    # -5% stop loss
    spot_tp_pct: float = 5.0     # +5% max take profit
    
    # Futures SL/TP (ROE %)
    futures_sl_pct: float = -5.0
    futures_tp_pct: float = 10.0
    
    # Grid Trading (Spot only)
    grid_enabled: bool = True
    grid_levels: int = 2  # 2, 3, or 4 levels
    
    # Trailing Stop-Loss
    trailing_sl_enabled: bool = True
    trailing_activation_pct: float = 2.0  # Activate at +2% profit
    trailing_distance_pct: float = 1.0    # Trail 1% behind
    
    # Hour filter
    hour_filter_enabled: bool = True
    blocked_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 17, 18])
    
    # Smart exit
    exit_signal_strength_short: float = 30.0
    exit_signal_strength_long: float = 70.0
    exit_min_effective_move: float = 5.0
    min_profit_to_exit: float = 0.01  # 1%
    
    # 2x Flip toggle
    flip_2x_enabled: bool = False  # Zararda 2x ters sinyal ile flip
    flip_2x_min_confidence: float = 50.0  # Flip için minimum güven
    
    # SL Cooldown
    sl_cooldown_candles: int = 16  # SL sonrası bekleme (16 mum = 4 saat)
    
    # Animation
    animation_speed: int = 100  # ms per candle (lower = faster)
    show_live: bool = True
    
    # Timeframe
    timeframe: str = "15m"
    
    def get_grid_allocations(self) -> list:
        """Get buy allocation percentages for each grid level."""
        if self.grid_levels == 2:
            return [0.60, 0.40]  # 60% + 40%
        elif self.grid_levels == 3:
            return [0.50, 0.30, 0.20]  # 50% + 30% + 20%
        elif self.grid_levels == 4:
            return [0.40, 0.30, 0.20, 0.10]  # 40% + 30% + 20% + 10%
        return [1.0]  # Fallback: single buy
    
    def get_grid_sell_targets(self) -> list:
        """Get sell target percentages for each grid level."""
        if self.grid_levels == 2:
            return [0.02, 0.04]  # +2%, +4%
        elif self.grid_levels == 3:
            return [0.015, 0.03, 0.05]  # +1.5%, +3%, +5%
        elif self.grid_levels == 4:
            return [0.01, 0.02, 0.035, 0.05]  # +1%, +2%, +3.5%, +5%
        return [0.05]  # Fallback: single target
    
    def get_sl_tp(self):
        """Get SL/TP based on trading mode."""
        if self.trading_mode == "spot":
            return self.spot_sl_pct, self.spot_tp_pct
        else:
            return self.futures_sl_pct, self.futures_tp_pct
    
    def get_effective_leverage(self):
        """Return leverage (1 for spot, configured for futures)."""
        return 1 if self.trading_mode == "spot" else self.leverage

@dataclass
class Position:
    """Pozisyon bilgisi"""
    coin: str
    side: str = "FLAT"
    entry_price: float = 0.0
    amount: float = 0.0
    entry_time: datetime = None
    entry_prediction: float = 0.0
    opposite_signal_count: int = 0
    
    # Grid trading fields (Spot only)
    buy_grid_level: int = 0
    sell_grid_level: int = 0
    grid_entries: list = None
    total_invested: float = 0.0
    last_grid_time: datetime = None
    
    # Trailing stop-loss
    trailing_sl_price: float = 0.0  # Current trailing SL level
    highest_price: float = 0.0      # Highest price since entry (for trailing)
    
    def __post_init__(self):
        if self.grid_entries is None:
            self.grid_entries = []
    
    def avg_entry_price(self) -> float:
        """Calculate weighted average entry price from all grid entries."""
        if not self.grid_entries:
            return self.entry_price
        total_value = sum(e['price'] * e['amount'] for e in self.grid_entries)
        total_amount = sum(e['amount'] for e in self.grid_entries)
        return total_value / total_amount if total_amount > 0 else 0.0
    
    def total_amount(self) -> float:
        """Total amount across all grid entries."""
        if not self.grid_entries:
            return self.amount
        return sum(e['amount'] for e in self.grid_entries)
    
    def reset_grid(self):
        """Reset all grid tracking when position is closed."""
        self.buy_grid_level = 0
        self.sell_grid_level = 0
        self.grid_entries = []
        self.total_invested = 0.0
        self.last_grid_time = None
        self.opposite_signal_count = 0
        self.trailing_sl_price = 0.0
        self.highest_price = 0.0
    
    def pnl_pct(self, current_price: float) -> float:
        if self.side == "FLAT" or self.entry_price == 0:
            return 0.0
        # Use avg entry price for grid trades
        entry = self.avg_entry_price() if self.grid_entries else self.entry_price
        if self.side == "LONG":
            return ((current_price - entry) / entry) * 100
        else:  # SHORT
            return ((entry - current_price) / entry) * 100

@dataclass
class Trade:
    """Trade kaydı"""
    id: int
    coin: str
    side: str
    entry_time: datetime
    entry_price: float
    entry_idx: int = 0
    exit_time: datetime = None
    exit_price: float = 0.0
    exit_idx: int = 0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    prediction: float = 0.0
    confidence: float = 0.0


class RealAIBacktester:
    """Gerçek AI Modeli ile Görsel Backtest Simülatörü"""
    
    def __init__(self, coin: str, start_date: datetime, end_date: datetime, config: SimConfig = None,
                 use_meta_model: bool = False, meta_threshold: float = 0.50):
        self.coin = coin
        self.start_date = start_date
        self.end_date = end_date
        self.config = config or SimConfig()
        
        # AI Model
        self.model = None
        self.model_stats = None
        self.model_params = None
        
        # Meta-Model Gate
        self.use_meta_model = use_meta_model
        self.meta_gate = None
        self.meta_threshold = meta_threshold
        self.meta_blocked_count = 0  # Trades blocked by meta-model
        
        # SL Cooldown
        self.sl_cooldown_until = None  # datetime until which new trades are blocked
        self.sl_cooldown_blocked = 0   # Count of signals blocked by cooldown
        
        # State
        self.balance = self.config.initial_balance
        self.position = Position(coin)
        self.trades: List[Trade] = []
        self.trade_counter = 0
        
        # Data
        self.df_display: pd.DataFrame = None  # Raw OHLCV
        self.df_ai: pd.DataFrame = None       # AI features
        self.current_idx = 0
        self.start_idx = 120  # Need 120 candles for LSTM
        
        # Predictions cache (computed once)
        self.predictions: Dict[int, Dict] = {}
        
        # Animation
        self.fig = None
        self.ax_price = None
        self.ax_pnl = None
        self.ax_info = None
        self.anim = None
        
        # Stats
        self.equity_curve = []
        self.pnl_history = []
        
        # Exchange for data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    
    def load_model(self) -> bool:
        """AI modelini yükle"""
        print(f"🧠 {self.coin} AI modeli yükleniyor...")
        
        key = f"{self.coin}_{self.config.timeframe}"
        model_path = PROJECT_ROOT / "kaggle_outputs" / f"{key}_model.pth"
        params_path = PROJECT_ROOT / "kaggle_outputs" / f"{key}_params.json"
        stats_path = PROJECT_ROOT / "kaggle_outputs" / f"{key}_stats.json"
        
        if not model_path.exists():
            print(f"❌ Model bulunamadı: {model_path}")
            return False
        
        try:
            # Load params
            with open(params_path) as f:
                self.model_params = json.load(f)
            
            # Load stats
            with open(stats_path) as f:
                self.model_stats = json.load(f)
            
            # Create and load model
            self.model = MultiBranchModel(
                embed_dim=self.model_params.get('embed_dim', 96),
                dropout=self.model_params.get('dropout', 0.15)
            ).to(DEVICE)
            
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(clean_state)
            self.model.eval()
            
            print(f"✅ Model yüklendi: {key}")
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def fetch_data(self) -> bool:
        """Binance'ten veri çek ve AI için hazırla"""
        print(f"📊 {self.coin} verisi çekiliyor...")
        
        try:
            # Need extra data for AI (120 candles lookback)
            tf_minutes = int(self.config.timeframe.replace('m', '').replace('h', '')) 
            if 'h' in self.config.timeframe:
                tf_minutes *= 60
            
            # Calculate months needed
            days_needed = (self.end_date - self.start_date).days + 10  # extra buffer
            candles_needed = (days_needed * 24 * 60) // tf_minutes + 200
            months_back = (candles_needed * tf_minutes / (30 * 24 * 60)) * 1.2
            
            symbol = f"{self.coin}/USDT"
            df = get_crypto_history(
                symbol=symbol,
                timeframe=self.config.timeframe,
                months_back=max(months_back, 3),
                exchange_name="binance"
            )
            
            if len(df) < 200:
                print(f"❌ Yeterli veri yok: {len(df)} mum")
                return False
            
            # Prepare dataframes
            self.df_display, self.df_ai = prepare_dual_dataframes(df)
            
            # Add datetime column
            self.df_display['datetime'] = pd.to_datetime(self.df_display.index)
            self.df_display['hour'] = self.df_display['datetime'].dt.hour
            
            # Find start/end indices
            mask = (self.df_display['datetime'] >= self.start_date) & (self.df_display['datetime'] <= self.end_date)
            valid_indices = self.df_display[mask].index.tolist()
            
            if not valid_indices:
                print(f"❌ Belirtilen tarih aralığında veri yok")
                return False
            
            # Convert to integer positions
            self.start_idx = max(120, self.df_display.index.get_loc(valid_indices[0]))
            self.end_idx = self.df_display.index.get_loc(valid_indices[-1])
            
            print(f"✅ {len(self.df_display)} mum verisi hazır")
            print(f"   📅 Test aralığı: mum {self.start_idx} → {self.end_idx} ({self.end_idx - self.start_idx} mum)")
            
            return True
            
        except Exception as e:
            print(f"❌ Veri çekme hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_prediction(self, idx: int) -> Optional[Dict]:
        """Belirli bir index için AI tahmini al"""
        if idx in self.predictions:
            return self.predictions[idx]
        
        try:
            if idx < 120:
                return None
            
            # Get column indices
            cnn_cols = [self.df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in self.df_ai.columns]
            lstm_cols = [self.df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in self.df_ai.columns]
            tr_cols = [self.df_ai.columns.get_loc(c) for c in TR_FEATURES if c in self.df_ai.columns]
            
            # Normalize using model stats
            mean = pd.Series(self.model_stats['mean'])
            std = pd.Series(self.model_stats['std'])
            std[std == 0] = 1.0
            
            df_normalized = (self.df_ai - mean) / std
            data = df_normalized.values
            
            # Get position in array
            pos = self.df_ai.index.get_loc(self.df_display.index[idx])
            
            # Prepare tensors
            x_cnn = torch.tensor(data[pos-12:pos, cnn_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_lstm = torch.tensor(data[pos-120:pos, lstm_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_tr = torch.tensor(data[pos-120:pos, tr_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Run prediction
            with torch.no_grad():
                pred_main, pred_cnn, pred_lstm, pred_tr = self.model(x_cnn, x_lstm, x_tr)
                prediction = pred_main.item() / 100.0
                
                # Confidence from branch agreement
                p_cnn = pred_cnn.item() / 100.0
                p_lstm = pred_lstm.item() / 100.0
                p_tr = pred_tr.item() / 100.0
                branches = [p_cnn, p_lstm, p_tr]
                signs = [1 if b > 0 else -1 for b in branches]
                confidence = abs(sum(signs)) / 3.0 * 100
            
            current_price = float(self.df_display.iloc[idx]['Close'])
            current_time = self.df_display.iloc[idx]['datetime']
            hour = self.df_display.iloc[idx]['hour']
            
            result = {
                'prediction': prediction * self.config.prediction_scale,
                'prediction_pct': prediction * self.config.prediction_scale * 100,
                'confidence': confidence,
                'price': current_price,
                'timestamp': current_time,
                'hour': hour,
                # Branch predictions for meta-model
                'pred_cnn': p_cnn,
                'pred_lstm': p_lstm,
                'pred_tr': p_tr,
                'df_display': self.df_display,
                'candle_idx': idx,
            }
            
            self.predictions[idx] = result
            return result
            
        except Exception as e:
            print(f"⚠️ Tahmin hatası idx={idx}: {e}")
            return None
    
    def should_trade(self, prediction: Dict) -> Tuple[str, str]:
        """Gerçek botla aynı trading mantığı - Spot ve Futures desteği"""
        pred = prediction['prediction']
        confidence = prediction['confidence']
        price = prediction['price']
        hour = prediction.get('hour', 12)
        threshold = self.config.prediction_threshold
        pos = self.position
        is_spot = self.config.trading_mode == "spot"
        leverage = self.config.get_effective_leverage()
        
        # Check confidence threshold
        if confidence < self.config.min_confidence_threshold:
            return ('HOLD', f'Düşük güven: {confidence:.0f}%')
        
        # Hour filter for new positions
        if self.config.hour_filter_enabled and pos.side == "FLAT":
            if hour in self.config.blocked_hours:
                return ('HOLD', f'🕐 Saat filtresi: {hour:02d}:00')
        
        # Get SL/TP limits based on mode
        sl_pct, tp_pct = self.config.get_sl_tp()
        sl_limit = sl_pct / 100
        tp_max = tp_pct / 100
        
        # ========== SPOT GRID MODE (separate logic like real bot) ==========
        is_spot_grid = is_spot and self.config.grid_enabled
        
        if is_spot_grid and pos.side == "LONG":
            avg_price = pos.avg_entry_price() if pos.grid_entries else pos.entry_price
            current_pnl = ((price - avg_price) / avg_price) if avg_price > 0 else 0
            
            # Update highest price for trailing SL
            if price > pos.highest_price:
                pos.highest_price = price
            
            # Stop-loss: sell everything immediately
            if current_pnl <= sl_limit:
                return ('SELL_ALL', f'🛑 Stop-Loss: {current_pnl*100:.2f}%')
            
            # Trailing stop-loss check
            if self.config.trailing_sl_enabled and pos.trailing_sl_price > 0:
                if price <= pos.trailing_sl_price:
                    trail_pnl = ((pos.trailing_sl_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
                    return ('SELL_ALL', f'🔒 Trailing SL: {trail_pnl:+.2f}%')
            
            # Update trailing SL if profit exceeds activation threshold
            if self.config.trailing_sl_enabled and current_pnl * 100 >= self.config.trailing_activation_pct:
                new_sl = price * (1 - self.config.trailing_distance_pct / 100)
                min_sl = avg_price * 1.002  # At least 0.2% above avg entry
                if new_sl > min_sl and new_sl > pos.trailing_sl_price:
                    pos.trailing_sl_price = new_sl
            
            # Check grid sell targets FIRST (before smart exit)
            sell_targets = self.config.get_grid_sell_targets()
            if pos.sell_grid_level < len(sell_targets):
                target = sell_targets[pos.sell_grid_level]
                if current_pnl >= target:
                    return ('GRID_SELL', f'🎯 Grid Hedef {pos.sell_grid_level+1}: {current_pnl*100:+.2f}%')
            
            # Smart exit: ONLY if all grid sell targets already hit
            if pos.sell_grid_level >= len(sell_targets):
                if pred < 0 and current_pnl >= self.config.min_profit_to_exit:
                    effective_move = abs(pred * 100)
                    signal_strong = (
                        confidence >= self.config.exit_signal_strength_short and
                        effective_move >= self.config.exit_min_effective_move
                    )
                    if signal_strong:
                        return ('SELL_ALL', f'📉 Güçlü Bearish + kâr: {current_pnl*100:+.2f}%')
            
            # Check if we can add more grid buys
            if pos.buy_grid_level < self.config.grid_levels and pred > threshold:
                price_drop = (avg_price - price) / avg_price if avg_price > 0 else 0
                time_since_last = None
                if pos.last_grid_time:
                    time_since_last = (prediction['timestamp'] - pos.last_grid_time).total_seconds() / 60
                
                # Add if price dropped 1%+ OR 30+ min since last grid
                if price_drop >= 0.01 or (time_since_last and time_since_last >= 30):
                    return ('GRID_BUY', f'📈 Grid Ekleme {pos.buy_grid_level+1}/{self.config.grid_levels}')
            
            if pred > threshold:
                return ('HOLD', 'Hala yükseliş - grid bekleniyor')
            
            return ('HOLD', 'Bekleniyor')
        
        # ========== STANDARD LONG CLOSE LOGIC (Futures or non-grid Spot) ==========
        if pos.side == "LONG":
            current_pnl = pos.pnl_pct(price) / 100 * leverage
            
            # Update highest price for trailing
            if price > pos.highest_price:
                pos.highest_price = price
            
            # Stop-loss
            if current_pnl <= sl_limit:
                return ('SELL', f'🛑 Stop-Loss: {current_pnl*100:.2f}%')
            
            # Trailing stop-loss check (futures)
            if self.config.trailing_sl_enabled and pos.trailing_sl_price > 0:
                if price <= pos.trailing_sl_price:
                    trail_pnl = ((pos.trailing_sl_price - pos.entry_price) / pos.entry_price) * 100 * leverage
                    return ('SELL', f'🔒 Trailing SL: {trail_pnl:+.2f}%')
            
            # Update trailing SL if profit exceeds activation threshold
            if self.config.trailing_sl_enabled:
                pnl_for_trail = current_pnl * 100  # ROE %
                if pnl_for_trail >= self.config.trailing_activation_pct:
                    distance_factor = (self.config.trailing_distance_pct / leverage) / 100
                    new_sl = price * (1 - distance_factor)
                    min_sl = pos.entry_price * 1.002
                    if new_sl > min_sl and new_sl > pos.trailing_sl_price:
                        pos.trailing_sl_price = new_sl
            
            # Take profit
            target_tp = abs(pos.entry_prediction) if pos.entry_prediction else tp_max
            target_tp = min(target_tp, tp_max)
            
            if current_pnl >= target_tp:
                return ('SELL', f'🎯 Take-Profit: {current_pnl*100:+.2f}%')
            
            # Smart exit: Bearish + profit
            if pred < 0 and current_pnl >= self.config.min_profit_to_exit:
                effective_move = abs(pred * 100) * leverage
                signal_strong = (
                    confidence >= self.config.exit_signal_strength_short and
                    effective_move >= self.config.exit_min_effective_move
                )
                if signal_strong:
                    pos.opposite_signal_count = 0
                    if is_spot:
                        return ('SELL', f'📉 Güçlü Bearish + kâr: {current_pnl*100:+.2f}%')
                    else:
                        return ('SELL_AND_FLIP', f'📉 Güçlü Bearish + kâr: {current_pnl*100:+.2f}%')
                else:
                    return ('HOLD', f'📉 Zayıf Bearish')
            
            # 2x opposite signal
            if self.config.flip_2x_enabled and pred < -threshold and current_pnl < 0:
                if confidence >= self.config.flip_2x_min_confidence:
                    pos.opposite_signal_count += 1
                    if pos.opposite_signal_count >= 2:
                        pos.opposite_signal_count = 0
                        return ('SELL_AND_FLIP', f'⚠️ 2x ters sinyal + flip')
                    else:
                        return ('HOLD', '⚠️ Ters sinyal 1/2')
                else:
                    return ('HOLD', f'⚠️ 2x flip için düşük güven: {confidence:.0f}%')
            
            if pred > threshold:
                pos.opposite_signal_count = 0
                return ('HOLD', 'Hala yükseliş')
            
            return ('HOLD', 'Bekleniyor')
        
        # ========== SHORT CLOSE LOGIC (Futures only) ==========
        if pos.side == "SHORT":
            current_pnl = pos.pnl_pct(price) / 100 * leverage
            
            # Stop-loss
            if current_pnl <= sl_limit:
                return ('CLOSE_SHORT', f'🛑 Stop-Loss: {current_pnl*100:.2f}%')
            
            # Take profit
            target_tp = abs(pos.entry_prediction) if pos.entry_prediction else tp_max
            target_tp = min(target_tp, tp_max)
            
            if current_pnl >= target_tp:
                return ('CLOSE_SHORT', f'🎯 Take-Profit: {current_pnl*100:+.2f}%')
            
            # Smart exit: Bullish + profit
            if pred > 0 and current_pnl >= self.config.min_profit_to_exit:
                effective_move = abs(pred * 100) * leverage
                signal_strong = (
                    confidence >= self.config.exit_signal_strength_long and
                    effective_move >= self.config.exit_min_effective_move
                )
                if signal_strong:
                    pos.opposite_signal_count = 0
                    return ('CLOSE_SHORT_AND_FLIP', f'📈 Güçlü Bullish + kâr: {current_pnl*100:+.2f}%')
                else:
                    return ('HOLD', f'📈 Zayıf Bullish')
            
            # 2x opposite signal
            if self.config.flip_2x_enabled and pred > threshold and current_pnl < 0:
                if confidence >= self.config.flip_2x_min_confidence:
                    pos.opposite_signal_count += 1
                    if pos.opposite_signal_count >= 2:
                        pos.opposite_signal_count = 0
                        return ('CLOSE_SHORT_AND_FLIP', f'⚠️ 2x ters sinyal + flip')
                    else:
                        return ('HOLD', '⚠️ Ters sinyal 1/2')
                else:
                    return ('HOLD', f'⚠️ 2x flip için düşük güven: {confidence:.0f}%')
            
            if pred < -threshold:
                pos.opposite_signal_count = 0
                return ('HOLD', 'Hala düşüş')
            
            return ('HOLD', 'Bekleniyor')
        
        # ========== OPEN NEW POSITION ==========
        if pos.side == "FLAT":
            # SL Cooldown check
            if self.sl_cooldown_until is not None:
                current_time = prediction.get('timestamp')
                if current_time and current_time < self.sl_cooldown_until:
                    self.sl_cooldown_blocked += 1
                    remaining = (self.sl_cooldown_until - current_time).total_seconds() / 60
                    return ('HOLD', f'⏳ SL Cooldown: {remaining:.0f}dk kaldı')
                else:
                    self.sl_cooldown_until = None  # Cooldown expired
            
            # LONG signal
            if pred > threshold:
                if confidence < 50:
                    return ('HOLD', f'⚠️ LONG için düşük güven: {confidence:.0f}%')
                
                effective_pct = pred * 100 * leverage
                if effective_pct < self.config.min_prediction_pct:
                    return ('HOLD', f'⚠️ Zayıf LONG: {effective_pct:.1f}%')
                
                # === META-MODEL GATE ===
                if self.use_meta_model and self.meta_gate and self.meta_gate.loaded:
                    df_disp = prediction.get('df_display')
                    candle_idx = prediction.get('candle_idx', -1)
                    allow, prob = self.meta_gate.should_allow_trade(prediction, df_disp, candle_idx)
                    if not allow:
                        self.meta_blocked_count += 1
                        return ('HOLD', f'🧠 Meta-model RED: {prob*100:.0f}% < {self.meta_threshold*100:.0f}%')
                
                # Grid mode: check if we should add to position (spot only)
                if is_spot and self.config.grid_enabled and pos.buy_grid_level > 0:
                    # Already have a position, this would be grid add
                    return ('GRID_BUY', f'📊 Grid Alım Seviye {pos.buy_grid_level + 1}')
                
                return ('BUY', f'📈 Yükselir: +{pred*100:.2f}%')
            
            # SHORT signal (Futures only - no shorting in spot)
            if not is_spot and pred < -threshold:
                effective_pct = abs(pred * 100) * leverage
                if effective_pct < self.config.min_prediction_pct:
                    return ('HOLD', f'⚠️ Zayıf SHORT: {effective_pct:.1f}%')
                
                # === META-MODEL GATE (SHORT) ===
                if self.use_meta_model and self.meta_gate and self.meta_gate.loaded:
                    df_disp = prediction.get('df_display')
                    candle_idx = prediction.get('candle_idx', -1)
                    allow, prob = self.meta_gate.should_allow_trade(prediction, df_disp, candle_idx)
                    if not allow:
                        self.meta_blocked_count += 1
                        return ('HOLD', f'🧠 Meta-model RED: {prob*100:.0f}% < {self.meta_threshold*100:.0f}%')
                
                return ('SHORT', f'📉 Düşüş: {pred*100:.2f}%')
            
            # In spot mode, bearish signal = wait (don't trade)
            if is_spot and pred < -threshold:
                return ('HOLD', f'🚫 Spot modda SHORT yok, bekleniyor')
        
        return ('HOLD', 'Sinyal yok')
    
    def execute_action(self, action: str, prediction: Dict, reason: str, idx: int):
        """Işlemi gerçekleştir - Spot ve Futures desteği"""
        price = prediction['price']
        current_time = prediction['timestamp']
        leverage = self.config.get_effective_leverage()
        is_spot = self.config.trading_mode == "spot"
        
        if action == 'BUY':
            trade_amount = self.balance * self.config.position_pct
            
            # Grid mode: first buy uses first allocation percentage
            if is_spot and self.config.grid_enabled:
                allocations = self.config.get_grid_allocations()
                trade_amount = self.balance * self.config.position_pct * allocations[0]
            
            amount = (trade_amount * leverage) / price
            
            # Subtract investment from balance (like real trading)
            if is_spot and self.config.grid_enabled:
                self.balance -= trade_amount
            
            self.position.side = "LONG"
            self.position.entry_price = price
            self.position.amount = amount
            self.position.entry_time = current_time
            self.position.entry_prediction = prediction['prediction'] * self.config.tp_multiplier
            self.position.opposite_signal_count = 0
            
            # Grid tracking
            if is_spot and self.config.grid_enabled:
                self.position.buy_grid_level = 1
                self.position.grid_entries = [{'price': price, 'amount': amount, 'time': current_time}]
                self.position.total_invested = trade_amount
                self.position.last_grid_time = current_time
            
            self.trade_counter += 1
            self.trades.append(Trade(
                id=self.trade_counter,
                coin=self.coin,
                side="LONG",
                entry_time=current_time,
                entry_price=price,
                entry_idx=idx,
                prediction=prediction['prediction_pct'],
                confidence=prediction['confidence']
            ))
            
        elif action == 'GRID_BUY':
            # Grid additional buy (spot only)
            allocations = self.config.get_grid_allocations()
            current_level = self.position.buy_grid_level
            
            if current_level < len(allocations):
                # Use available balance (already reduced by previous buys)
                trade_amount = self.balance * allocations[current_level] / sum(allocations[current_level:])
                trade_amount = min(trade_amount, self.balance * 0.95)
                
                if trade_amount > 1:
                    amount = trade_amount / price
                    
                    # Subtract from balance
                    self.balance -= trade_amount
                    
                    self.position.grid_entries.append({'price': price, 'amount': amount, 'time': current_time})
                    self.position.total_invested += trade_amount
                    self.position.amount += amount
                    self.position.buy_grid_level += 1
                    self.position.last_grid_time = current_time
                    
                    # Update entry price to weighted average
                    self.position.entry_price = self.position.avg_entry_price()
        
        elif action == 'GRID_SELL':
            # Grid partial sell (spot only) - sell portion based on grid level
            allocations = self.config.get_grid_allocations()
            current_sell_level = self.position.sell_grid_level
            max_levels = len(allocations)
            
            if current_sell_level < max_levels:
                sell_pct = allocations[current_sell_level]
                sell_amount = self.position.amount * sell_pct
                
                # Add coin value back to balance at current price
                self.balance += sell_amount * price
                self.position.amount -= sell_amount
                self.position.sell_grid_level += 1
                self.position.last_grid_time = current_time
                
                avg_entry = self.position.avg_entry_price()
                pnl_pct = ((price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
                
                # Check if all grids sold
                if self.position.sell_grid_level >= max_levels or self.position.amount < 0.000001:
                    if self.trades:
                        self.trades[-1].exit_time = current_time
                        self.trades[-1].exit_price = price
                        self.trades[-1].exit_idx = idx
                        self.trades[-1].pnl_pct = pnl_pct
                        self.trades[-1].exit_reason = f'🎯 Grid Satış Tamamlandı: {pnl_pct:+.2f}%'
                    
                    self.position.side = "FLAT"
                    self.position.entry_price = 0
                    self.position.amount = 0
                    self.position.reset_grid()
        
        elif action == 'SELL_ALL':
            # Emergency sell all (stop-loss or smart exit in grid mode)
            # Add remaining coin value back to balance at current price
            self.balance += self.position.amount * price
            
            avg_entry = self.position.avg_entry_price() if self.position.grid_entries else self.position.entry_price
            pnl_pct = ((price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
            
            if self.trades:
                self.trades[-1].exit_time = current_time
                self.trades[-1].exit_price = price
                self.trades[-1].exit_idx = idx
                self.trades[-1].pnl_pct = pnl_pct
                self.trades[-1].exit_reason = reason
            
            # Trigger SL cooldown
            if 'Stop-Loss' in reason and self.config.sl_cooldown_candles > 0:
                cooldown_minutes = self.config.sl_cooldown_candles * 15
                self.sl_cooldown_until = current_time + timedelta(minutes=cooldown_minutes)
            
            self.position.side = "FLAT"
            self.position.entry_price = 0
            self.position.amount = 0
            self.position.reset_grid()
            
        elif action == 'SHORT':
            trade_amount = self.balance * self.config.position_pct
            amount = (trade_amount * leverage) / price
            
            self.position.side = "SHORT"
            self.position.entry_price = price
            self.position.amount = amount
            self.position.entry_time = current_time
            self.position.entry_prediction = prediction['prediction'] * self.config.tp_multiplier
            self.position.opposite_signal_count = 0
            
            self.trade_counter += 1
            self.trades.append(Trade(
                id=self.trade_counter,
                coin=self.coin,
                side="SHORT",
                entry_time=current_time,
                entry_price=price,
                entry_idx=idx,
                prediction=prediction['prediction_pct'],
                confidence=prediction['confidence']
            ))
            
        elif action in ['SELL', 'CLOSE_SHORT']:
            pnl_pct = self.position.pnl_pct(price) * leverage
            pnl_usd = self.balance * (pnl_pct / 100)
            self.balance += pnl_usd
            
            if self.trades:
                self.trades[-1].exit_time = current_time
                self.trades[-1].exit_price = price
                self.trades[-1].exit_idx = idx
                self.trades[-1].pnl_pct = pnl_pct
                self.trades[-1].exit_reason = reason
            
            # Trigger SL cooldown
            if 'Stop-Loss' in reason and self.config.sl_cooldown_candles > 0:
                cooldown_minutes = self.config.sl_cooldown_candles * 15
                self.sl_cooldown_until = current_time + timedelta(minutes=cooldown_minutes)
            
            self.position.side = "FLAT"
            self.position.entry_price = 0
            self.position.amount = 0
            self.position.reset_grid()  # Reset grid tracking
            
        elif action == 'SELL_AND_FLIP':
            # Close LONG
            pnl_pct = self.position.pnl_pct(price) * leverage
            pnl_usd = self.balance * (pnl_pct / 100)
            self.balance += pnl_usd
            self.position.reset_grid()  # Reset grid tracking
            
            if self.trades:
                self.trades[-1].exit_time = current_time
                self.trades[-1].exit_price = price
                self.trades[-1].exit_idx = idx
                self.trades[-1].pnl_pct = pnl_pct
                self.trades[-1].exit_reason = reason + " → FLIP"
            
            # Open SHORT
            trade_amount = self.balance * self.config.position_pct
            amount = (trade_amount * leverage) / price
            
            self.position.side = "SHORT"
            self.position.entry_price = price
            self.position.amount = amount
            self.position.entry_time = current_time
            self.position.entry_prediction = prediction['prediction'] * self.config.tp_multiplier
            self.position.opposite_signal_count = 0
            
            self.trade_counter += 1
            self.trades.append(Trade(
                id=self.trade_counter,
                coin=self.coin,
                side="SHORT",
                entry_time=current_time,
                entry_price=price,
                entry_idx=idx,
                prediction=prediction['prediction_pct'],
                confidence=prediction['confidence']
            ))
            
        elif action == 'CLOSE_SHORT_AND_FLIP':
            # Close SHORT
            pnl_pct = self.position.pnl_pct(price) * leverage
            pnl_usd = self.balance * (pnl_pct / 100)
            self.balance += pnl_usd
            self.position.reset_grid()  # Reset grid tracking
            
            if self.trades:
                self.trades[-1].exit_time = current_time
                self.trades[-1].exit_price = price
                self.trades[-1].exit_idx = idx
                self.trades[-1].pnl_pct = pnl_pct
                self.trades[-1].exit_reason = reason + " → FLIP"
            
            # Open LONG
            trade_amount = self.balance * self.config.position_pct
            amount = (trade_amount * leverage) / price
            
            self.position.side = "LONG"
            self.position.entry_price = price
            self.position.amount = amount
            self.position.entry_time = current_time
            self.position.entry_prediction = prediction['prediction'] * self.config.tp_multiplier
            self.position.opposite_signal_count = 0
            
            self.trade_counter += 1
            self.trades.append(Trade(
                id=self.trade_counter,
                coin=self.coin,
                side="LONG",
                entry_time=current_time,
                entry_price=price,
                entry_idx=idx,
                prediction=prediction['prediction_pct'],
                confidence=prediction['confidence']
            ))
    
    def setup_plot(self):
        """Grafik kurulumu"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.suptitle(f'🎮 {self.coin} Visual Backtest - GERÇEK AI MODELİ', fontsize=16, fontweight='bold', color='cyan')
        
        # Grid layout
        gs = self.fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
        
        # Main price chart (big)
        self.ax_price = self.fig.add_subplot(gs[0:2, 0:4])
        self.ax_price.set_facecolor('#0a0a15')
        
        # Equity curve
        self.ax_pnl = self.fig.add_subplot(gs[2, 0:4])
        self.ax_pnl.set_facecolor('#0a0a15')
        
        # Info panel
        self.ax_info = self.fig.add_subplot(gs[0:3, 4])
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#0a0a15')
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Her frame'de grafik güncelle"""
        idx = self.start_idx + frame
        
        # Simülasyon bittiğinde durdur
        if idx >= self.end_idx:
            if self.anim:
                self.anim.event_source.stop()
            plt.close(self.fig)
            return
        
        self.current_idx = idx
        prediction = self.get_prediction(idx)
        
        if prediction is None:
            return
        
        # Progress bar
        progress = (frame + 1) / (self.end_idx - self.start_idx) * 100
        if frame % 10 == 0:
            print(f"\r[{'█' * int(progress/5)}{' ' * (20-int(progress/5))}] {progress:.0f}%", end='', flush=True)
        
        # Execute trading logic
        action, reason = self.should_trade(prediction)
        if action != 'HOLD':
            self.execute_action(action, prediction, reason, idx)
            print(f"\n[{prediction['timestamp'].strftime('%Y-%m-%d %H:%M')}] {action}: {reason}")
        
        # Update equity
        current_equity = self.balance
        if self.position.side != "FLAT":
            unrealized_pnl = self.position.pnl_pct(prediction['price']) * self.config.leverage
            current_equity = self.balance * (1 + unrealized_pnl / 100)
        self.equity_curve.append(current_equity)
        
        # Clear and redraw
        self.ax_price.clear()
        self.ax_pnl.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Price chart - show last 80 candles
        start_view = max(self.start_idx, idx - 80)
        visible_data = self.df_display.iloc[start_view:idx+1]
        
        # Candlestick chart
        for i, (_, r) in enumerate(visible_data.iterrows()):
            color = '#00ff88' if r['Close'] >= r['Open'] else '#ff4444'
            self.ax_price.plot([r['datetime'], r['datetime']], [r['Low'], r['High']], color=color, linewidth=0.8)
            self.ax_price.plot([r['datetime'], r['datetime']], [r['Open'], r['Close']], color=color, linewidth=3)
        
        # Draw trade markers
        for trade in self.trades:
            if trade.entry_idx >= start_view and trade.entry_idx <= idx:
                marker = '^' if trade.side == "LONG" else 'v'
                color = '#00ff00' if trade.side == "LONG" else '#ff0000'
                self.ax_price.scatter(trade.entry_time, trade.entry_price, marker=marker, color=color, s=150, zorder=5, edgecolors='white', linewidths=1)
            
            if trade.exit_time and trade.exit_idx >= start_view and trade.exit_idx <= idx:
                exit_color = '#00ffff' if trade.pnl_pct >= 0 else '#ff8800'
                self.ax_price.scatter(trade.exit_time, trade.exit_price, marker='o', color=exit_color, s=120, zorder=5, edgecolors='white', linewidths=1)
        
        # Current position line
        if self.position.side != "FLAT":
            color = '#00ffff' if self.position.side == "LONG" else '#ff00ff'
            self.ax_price.axhline(y=self.position.entry_price, color=color, linestyle='--', alpha=0.7, linewidth=2)
        
        current_time = self.df_display.iloc[idx]['datetime']
        self.ax_price.set_title(f'📈 {self.coin}/USDT - {current_time.strftime("%Y-%m-%d %H:%M")}', fontsize=12, color='cyan')
        self.ax_price.set_xlabel('Zaman', color='white')
        self.ax_price.set_ylabel('Fiyat (USDT)', color='white')
        self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        self.ax_price.tick_params(colors='white')
        
        # Equity curve
        if len(self.equity_curve) > 1:
            color = '#00ff88' if self.equity_curve[-1] >= self.config.initial_balance else '#ff4444'
            self.ax_pnl.fill_between(range(len(self.equity_curve)), self.config.initial_balance, self.equity_curve, 
                                      alpha=0.3, color=color)
            self.ax_pnl.plot(self.equity_curve, color=color, linewidth=2)
            self.ax_pnl.axhline(y=self.config.initial_balance, color='white', linestyle='--', alpha=0.5)
        self.ax_pnl.set_title('💰 Equity Curve', fontsize=12, color='lime')
        self.ax_pnl.set_ylabel('Bakiye ($)', color='white')
        self.ax_pnl.tick_params(colors='white')
        if self.equity_curve:
            self.ax_pnl.set_ylim(min(self.equity_curve) * 0.95, max(self.equity_curve) * 1.05)
        
        # Info panel
        total_pnl = ((current_equity - self.config.initial_balance) / self.config.initial_balance) * 100
        winning_trades = len([t for t in self.trades if t.exit_time and t.pnl_pct > 0])
        losing_trades = len([t for t in self.trades if t.exit_time and t.pnl_pct <= 0])
        total_closed = winning_trades + losing_trades
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0
        
        hour = prediction.get('hour', 0)
        hour_blocked = "[X]" if hour in self.config.blocked_hours else "[OK]"
        
        # ASCII-safe info panel (no emojis to avoid font warnings)
        info_text = f"""
+===========================+
|   GERCEK AI BACKTEST      |
+===========================+
| [$] Bakiye: ${current_equity:.2f}      
| [%] P&L: {total_pnl:+.2f}%            
|                           
| [#] Toplam Trade: {len(self.trades)}        
| [+] Kazanan: {winning_trades}             
| [-] Kaybeden: {losing_trades}            
| [*] Win Rate: {win_rate:.1f}%         
+===========================+
|      AI TAHMINI           |
+===========================+
| [AI] Tahmin: {prediction['prediction_pct']:+.2f}%      
| [C] Guven: {prediction['confidence']:.0f}%          
| [$] Fiyat: ${prediction['price']:.2f}    
| [H] Saat: {hour:02d}:00 {hour_blocked}         
|                           
| [P] Pozisyon: {self.position.side}         
"""
        
        if self.position.side != "FLAT":
            unrealized = self.position.pnl_pct(prediction['price']) * self.config.leverage
            info_text += f"""| [U] Unrealized: {unrealized:+.2f}%    
"""
        
        info_text += f"""|                           
| [A] Aksiyon: {action[:15]}      
| [R] {reason[:22]}
+===========================+
"""
        
        self.ax_info.text(0.02, 0.98, info_text, transform=self.ax_info.transAxes, 
                         fontsize=9, family='monospace', verticalalignment='top',
                         color='white', bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.95, edgecolor='#00ffff'))
    
    def _load_meta_model(self):
        """Meta-model gate'i yükle"""
        if not self.use_meta_model:
            return
        self.meta_gate = MetaModelGate(threshold=self.meta_threshold)
        if self.meta_gate.load():
            print(f"🧠 Meta-model yüklendi (threshold: {self.meta_threshold:.2f})")
        else:
            print("⚠️ Meta-model yüklenemedi, filtre devre dışı")
            self.use_meta_model = False

    def run_simulation(self):
        """Simülasyonu çalıştır"""
        # Load model
        if not self.load_model():
            return
        
        # Fetch data
        if not self.fetch_data():
            return
        
        # Load meta-model if needed
        self._load_meta_model()
        
        self.setup_plot()
        
        total_frames = self.end_idx - self.start_idx
        meta_str = " + 🧠 Meta-Model" if self.use_meta_model else ""
        print(f"\n🎮 Simülasyon başlatılıyor: {total_frames} mum{meta_str}")
        print(f"   📅 {self.start_date.strftime('%Y-%m-%d')} → {self.end_date.strftime('%Y-%m-%d')}")
        print(f"   💰 Başlangıç: ${self.config.initial_balance}")
        print(f"   ⚡ Kaldıraç: {self.config.leverage}x")
        print(f"   🕐 Saat filtresi: {'AÇIK' if self.config.hour_filter_enabled else 'KAPALI'}")
        if self.use_meta_model:
            print(f"   🧠 Meta-Model: AÇIK (threshold: {self.meta_threshold:.2f})")
        print("")
        
        if self.config.show_live:
            self.anim = FuncAnimation(
                self.fig, 
                self.update_plot, 
                frames=total_frames,
                interval=self.config.animation_speed,
                repeat=False
            )
            plt.show()
        else:
            # Run without animation (faster)
            for frame in range(total_frames):
                idx = self.start_idx + frame
                self.current_idx = idx
                prediction = self.get_prediction(idx)
                
                if prediction is None:
                    continue
                
                action, reason = self.should_trade(prediction)
                if action != 'HOLD':
                    self.execute_action(action, prediction, reason, idx)
                    print(f"[{prediction['timestamp'].strftime('%Y-%m-%d %H:%M')}] {action}: {reason}")
                
                # Update equity
                current_equity = self.balance
                if self.position.side != "FLAT":
                    unrealized_pnl = self.position.pnl_pct(prediction['price']) * self.config.leverage
                    current_equity = self.balance * (1 + unrealized_pnl / 100)
                self.equity_curve.append(current_equity)
        
        self.print_summary()
        self.save_results()
        self.show_completion_notification()
    
    def print_summary(self):
        """Sonuç özeti"""
        final_balance = self.equity_curve[-1] if self.equity_curve else self.balance
        total_pnl = ((final_balance - self.config.initial_balance) / self.config.initial_balance) * 100
        
        winning = [t for t in self.trades if t.exit_time and t.pnl_pct > 0]
        losing = [t for t in self.trades if t.exit_time and t.pnl_pct <= 0]
        
        meta_str = " + 🧠 Meta-Model" if self.use_meta_model else ""
        print("\n" + "="*60)
        print(f"📊 SİMÜLASYON SONUÇLARI{meta_str}")
        print("="*60)
        print(f"🧠 Model: {self.coin}_{self.config.timeframe}")
        print(f"💰 Başlangıç: ${self.config.initial_balance:.2f}")
        print(f"💰 Bitiş: ${final_balance:.2f}")
        print(f"📈 Toplam P&L: {total_pnl:+.2f}%")
        print(f"")
        print(f"📊 Toplam Trade: {len(self.trades)}")
        print(f"✅ Kazanan: {len(winning)}")
        print(f"❌ Kaybeden: {len(losing)}")
        print(f"🎯 Win Rate: {len(winning)/len(self.trades)*100:.1f}%" if self.trades else "N/A")
        if self.use_meta_model:
            print(f"🧠 Meta-model engelledi: {self.meta_blocked_count} trade")
        if self.config.sl_cooldown_candles > 0:
            print(f"⏳ SL Cooldown engelledi: {self.sl_cooldown_blocked} sinyal ({self.config.sl_cooldown_candles} mum = {self.config.sl_cooldown_candles*15} dk)")
        
        if winning:
            avg_win = sum(t.pnl_pct for t in winning) / len(winning)
            print(f"📈 Ort. Kazanç: +{avg_win:.2f}%")
        if losing:
            avg_loss = sum(t.pnl_pct for t in losing) / len(losing)
            print(f"📉 Ort. Kayıp: {avg_loss:.2f}%")
        
        print("\n" + "-"*60)
        print("📝 TRADE GEÇMİŞİ")
        print("-"*60)
        for t in self.trades:
            emoji = "🟢" if t.pnl_pct > 0 else "🔴" if t.pnl_pct < 0 else "⚪"
            exit_str = t.exit_time.strftime('%m-%d %H:%M') if t.exit_time else "AÇIK"
            print(f"{emoji} #{t.id} {t.side} | {t.entry_time.strftime('%m-%d %H:%M')} → {exit_str} | {t.pnl_pct:+.2f}% | {t.exit_reason[:35]}")
    
    def save_results(self):
        """Tüm sonuçları AI analizi için kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = PROJECT_ROOT / "backtest_results"
        results_dir.mkdir(exist_ok=True)
        
        final_balance = self.equity_curve[-1] if self.equity_curve else self.balance
        total_pnl = ((final_balance - self.config.initial_balance) / self.config.initial_balance) * 100
        
        winning = [t for t in self.trades if t.exit_time and t.pnl_pct > 0]
        losing = [t for t in self.trades if t.exit_time and t.pnl_pct <= 0]
        
        # ========== 1. TRADE HISTORY JSON (AI Analiz için) ==========
        trades_data = []
        for t in self.trades:
            trades_data.append({
                'id': t.id,
                'coin': t.coin,
                'side': t.side,
                'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                'entry_price': t.entry_price,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'exit_price': t.exit_price,
                'pnl_pct': round(t.pnl_pct, 4),
                'exit_reason': t.exit_reason,
                'ai_prediction': round(t.prediction, 4),
                'ai_confidence': round(t.confidence, 2),
                'entry_hour': t.entry_time.hour if t.entry_time else None,
                'exit_hour': t.exit_time.hour if t.exit_time else None,
                'hold_duration_minutes': int((t.exit_time - t.entry_time).total_seconds() / 60) if t.exit_time and t.entry_time else None,
                'is_winning': t.pnl_pct > 0,
                'is_stop_loss': 'Stop-Loss' in t.exit_reason,
                'is_take_profit': 'Take-Profit' in t.exit_reason,
                'is_flip': 'FLIP' in t.exit_reason,
                'is_smart_exit': 'Güçlü' in t.exit_reason,
                'is_2x_signal': '2x ters' in t.exit_reason
            })
        
        trades_file = results_dir / f"backtest_trades_{self.coin}_{timestamp}.json"
        with open(trades_file, 'w', encoding='utf-8') as f:
            json.dump(trades_data, f, indent=2, ensure_ascii=False)
        
        # ========== 2. SUMMARY JSON ==========
        summary = {
            'backtest_info': {
                'coin': self.coin,
                'timeframe': self.config.timeframe,
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'leverage': self.config.leverage,
                'initial_balance': self.config.initial_balance,
                'final_balance': round(final_balance, 2),
                'total_candles': self.end_idx - self.start_idx,
                'timestamp': datetime.now().isoformat()
            },
            'performance': {
                'total_pnl_pct': round(total_pnl, 2),
                'total_trades': len(self.trades),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': round(len(winning) / len(self.trades) * 100, 2) if self.trades else 0,
                'avg_win_pct': round(sum(t.pnl_pct for t in winning) / len(winning), 2) if winning else 0,
                'avg_loss_pct': round(sum(t.pnl_pct for t in losing) / len(losing), 2) if losing else 0,
                'best_trade_pct': round(max(t.pnl_pct for t in self.trades), 2) if self.trades else 0,
                'worst_trade_pct': round(min(t.pnl_pct for t in self.trades), 2) if self.trades else 0,
                'max_drawdown_pct': round((min(self.equity_curve) - self.config.initial_balance) / self.config.initial_balance * 100, 2) if self.equity_curve else 0
            },
            'config': {
                # Trading Mode
                'trading_mode': self.config.trading_mode,
                'leverage': self.config.get_effective_leverage(),
                'position_pct': self.config.position_pct,
                
                # Thresholds
                'prediction_threshold': self.config.prediction_threshold,
                'min_confidence_threshold': self.config.min_confidence_threshold,
                'min_prediction_pct': self.config.min_prediction_pct,
                'prediction_scale': self.config.prediction_scale,
                
                # Spot SL/TP
                'spot_sl_pct': self.config.spot_sl_pct,
                'spot_tp_pct': self.config.spot_tp_pct,
                
                # Futures SL/TP
                'futures_sl_pct': self.config.futures_sl_pct,
                'futures_tp_pct': self.config.futures_tp_pct,
                'tp_multiplier': self.config.tp_multiplier,
                
                # Grid Trading (Spot only)
                'grid_enabled': self.config.grid_enabled,
                'grid_levels': self.config.grid_levels,
                
                # Hour filter
                'hour_filter_enabled': self.config.hour_filter_enabled,
                'blocked_hours': self.config.blocked_hours,
                
                # Smart exit
                'exit_signal_strength_short': self.config.exit_signal_strength_short,
                'exit_signal_strength_long': self.config.exit_signal_strength_long,
                'exit_min_effective_move': self.config.exit_min_effective_move,
                'min_profit_to_exit': self.config.min_profit_to_exit,
                
                # 2x Flip
                'flip_2x_enabled': self.config.flip_2x_enabled,
                'flip_2x_min_confidence': self.config.flip_2x_min_confidence
            },
            'analysis': {
                'stop_loss_count': len([t for t in self.trades if 'Stop-Loss' in t.exit_reason]),
                'take_profit_count': len([t for t in self.trades if 'Take-Profit' in t.exit_reason]),
                'flip_count': len([t for t in self.trades if 'FLIP' in t.exit_reason]),
                'smart_exit_count': len([t for t in self.trades if 'Güçlü' in t.exit_reason]),
                'signal_2x_count': len([t for t in self.trades if '2x ters' in t.exit_reason]),
                'long_trades': len([t for t in self.trades if t.side == 'LONG']),
                'short_trades': len([t for t in self.trades if t.side == 'SHORT']),
                'long_win_rate': round(len([t for t in winning if t.side == 'LONG']) / len([t for t in self.trades if t.side == 'LONG']) * 100, 2) if [t for t in self.trades if t.side == 'LONG'] else 0,
                'short_win_rate': round(len([t for t in winning if t.side == 'SHORT']) / len([t for t in self.trades if t.side == 'SHORT']) * 100, 2) if [t for t in self.trades if t.side == 'SHORT'] else 0
            },
            'trades': trades_data
        }
        
        summary_file = results_dir / f"backtest_summary_{self.coin}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # ========== 3. EQUITY CURVE DATA ==========
        equity_file = results_dir / f"backtest_equity_{self.coin}_{timestamp}.json"
        with open(equity_file, 'w', encoding='utf-8') as f:
            json.dump({'equity_curve': self.equity_curve}, f)
        
        # ========== 4. FINAL CHART IMAGE ==========
        if self.fig:
            chart_file = results_dir / f"backtest_chart_{self.coin}_{timestamp}.png"
            self.fig.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor='#0a0a15')
            print(f"📊 Grafik kaydedildi: {chart_file}")
        
        print(f"\n" + "="*60)
        print("💾 SONUÇLAR KAYDEDİLDİ")
        print("="*60)
        print(f"📁 Dizin: {results_dir}")
        print(f"📄 Trade Geçmişi: {trades_file.name}")
        print(f"📄 Özet Rapor: {summary_file.name}")
        print(f"📄 Equity Data: {equity_file.name}")
        
        return summary_file, trades_file
    
    def show_completion_notification(self):
        """Simülasyon bittiğinde bildirim göster"""
        final_balance = self.equity_curve[-1] if self.equity_curve else self.balance
        total_pnl = ((final_balance - self.config.initial_balance) / self.config.initial_balance) * 100
        
        # Sesli bildirim (beep)
        try:
            import winsound
            # 3 kez beep
            for _ in range(3):
                winsound.Beep(1000, 300)
                time.sleep(0.1)
        except:
            # Windows değilse print bell
            print('\a\a\a')
        
        # Büyük bildirim
        result_emoji = "🎉" if total_pnl > 0 else "😢"
        print("\n")
        print("╔" + "═"*58 + "╗")
        print("║" + " "*15 + f"{result_emoji} SİMÜLASYON BİTTİ! {result_emoji}" + " "*15 + "║")
        print("╠" + "═"*58 + "╣")
        print(f"║  💰 Final Bakiye: ${final_balance:.2f}" + " "*(37-len(f"${final_balance:.2f}")) + "║")
        print(f"║  📊 Toplam P&L: {total_pnl:+.2f}%" + " "*(40-len(f"{total_pnl:+.2f}%")) + "║")
        print(f"║  📈 Trade Sayısı: {len(self.trades)}" + " "*(38-len(f"{len(self.trades)}")) + "║")
        print("╚" + "═"*58 + "╝")
        print("\n")


def run_comparison(coin: str, start_date: datetime, end_date: datetime, 
                   config: SimConfig, meta_threshold: float = 0.50):
    """
    A/B Comparison: Meta-model OFF vs ON
    Aynı veri ve tahminlerle iki simülasyon çalıştırıp karşılaştırır.
    """
    import copy
    
    print(f"\n{'='*70}")
    print(f"🔬 META-MODEL A/B KARŞILAŞTIRMASI")
    print(f"{'='*70}")
    print(f"  Coin: {coin} | Mode: {config.trading_mode} | Leverage: {config.get_effective_leverage()}x")
    print(f"  📅 {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"  Meta-model threshold: {meta_threshold:.2f}")
    print(f"{'='*70}")
    
    # === RUN 1: WITHOUT META-MODEL ===
    print(f"\n{'─'*70}")
    print(f"📊 [A] Meta-Model KAPALI — Tüm sinyaller ile trade")
    print(f"{'─'*70}")
    config_a = copy.deepcopy(config)
    config_a.show_live = False
    bt_a = RealAIBacktester(coin, start_date, end_date, config_a,
                            use_meta_model=False)
    bt_a.run_simulation()
    
    # === RUN 2: WITH META-MODEL ===
    print(f"\n{'─'*70}")
    print(f"🧠 [B] Meta-Model AÇIK — Sadece meta-model onaylı trade'ler")
    print(f"{'─'*70}")
    config_b = copy.deepcopy(config)
    config_b.show_live = False
    bt_b = RealAIBacktester(coin, start_date, end_date, config_b,
                            use_meta_model=True, meta_threshold=meta_threshold)
    bt_b.run_simulation()
    
    # === COMPARISON ===
    def get_stats(bt):
        final = bt.equity_curve[-1] if bt.equity_curve else bt.balance
        pnl = ((final - bt.config.initial_balance) / bt.config.initial_balance) * 100
        winning = [t for t in bt.trades if t.exit_time and t.pnl_pct > 0]
        losing = [t for t in bt.trades if t.exit_time and t.pnl_pct <= 0]
        total = len(bt.trades)
        wr = len(winning) / total * 100 if total > 0 else 0
        avg_w = sum(t.pnl_pct for t in winning) / len(winning) if winning else 0
        avg_l = sum(t.pnl_pct for t in losing) / len(losing) if losing else 0
        return {
            'final': final, 'pnl': pnl, 'trades': total,
            'wins': len(winning), 'losses': len(losing),
            'win_rate': wr, 'avg_win': avg_w, 'avg_loss': avg_l,
            'blocked': getattr(bt, 'meta_blocked_count', 0),
            'cooldown_blocked': getattr(bt, 'sl_cooldown_blocked', 0)
        }
    
    sa = get_stats(bt_a)
    sb = get_stats(bt_b)
    
    print(f"\n\n{'='*70}")
    print(f"🏆 KARŞILAŞTIRMA SONUÇLARI")
    print(f"{'='*70}")
    print(f"{'Metrik':<25} {'[A] Meta KAPALI':>18} {'[B] Meta AÇIK':>18} {'Fark':>12}")
    print(f"{'─'*73}")
    
    def row(label, va, vb, fmt=".2f", suffix=""):
        diff = vb - va
        arrow = "✅" if diff > 0 else "❌" if diff < 0 else "➖"
        print(f"{label:<25} {va:>17{fmt}}{suffix} {vb:>17{fmt}}{suffix} {arrow} {diff:>+8{fmt}}{suffix}")
    
    row("💰 Final Bakiye ($)", sa['final'], sb['final'], ".2f")
    row("📈 Toplam P&L (%)", sa['pnl'], sb['pnl'], ".2f", "%")
    row("📊 Trade Sayısı", sa['trades'], sb['trades'], ".0f")
    row("✅ Kazanan", sa['wins'], sb['wins'], ".0f")
    row("❌ Kaybeden", sa['losses'], sb['losses'], ".0f")
    row("🎯 Win Rate (%)", sa['win_rate'], sb['win_rate'], ".1f", "%")
    row("📈 Ort. Kazanç (%)", sa['avg_win'], sb['avg_win'], ".2f", "%")
    row("📉 Ort. Kayıp (%)", sa['avg_loss'], sb['avg_loss'], ".2f", "%")
    print(f"{'─'*73}")
    print(f"{'🧠 Meta-model engelledi':<25} {'—':>18} {sb['blocked']:>18.0f}")
    print(f"{'⏳ SL Cooldown engelledi':<25} {sa['cooldown_blocked']:>18.0f} {sb['cooldown_blocked']:>18.0f}")
    
    # Overall verdict
    if sb['pnl'] > sa['pnl']:
        print(f"\n🎉 META-MODEL KAZANDI! P&L farkı: {sb['pnl'] - sa['pnl']:+.2f}%")
    elif sb['pnl'] < sa['pnl']:
        print(f"\n😔 Meta-model bu periyotta faydasız. P&L farkı: {sb['pnl'] - sa['pnl']:+.2f}%")
    else:
        print(f"\n➖ İki strateji eşit performans gösterdi.")
    
    print(f"{'='*70}\n")

    # === EQUITY CURVE CHART ===
    try:
        _plot_comparison_equity(bt_a, bt_b, coin, start_date, end_date)
    except Exception as e:
        print(f"⚠️ Grafik oluşturulamadı: {e}")


def _plot_comparison_equity(bt_a, bt_b, coin, start_date, end_date):
    """Her trade sonrası bakiye grafiği — Meta OFF vs Meta ON."""

    def build_balance_series(bt):
        """Compute balance after each closed trade."""
        balance = bt.config.initial_balance
        balances = [balance]
        trade_labels = [0]  # trade number
        for i, t in enumerate(bt.trades):
            if t.exit_time and t.pnl_pct != 0:
                pnl_usd = balance * (t.pnl_pct / 100)
                balance += pnl_usd
                balances.append(balance)
                trade_labels.append(i + 1)
        return trade_labels, balances

    x_a, y_a = build_balance_series(bt_a)
    x_b, y_b = build_balance_series(bt_b)

    # Dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Starting balance line
    init_bal = bt_a.config.initial_balance
    ax.axhline(y=init_bal, color='#8b949e', linestyle='--', alpha=0.5,
               linewidth=1, label=f'Başlangıç (${init_bal:.0f})')

    # Meta OFF — red/orange
    ax.plot(x_a, y_a, color='#f85149', linewidth=2, alpha=0.9, label='[A] Meta KAPALI')
    ax.fill_between(x_a, init_bal, y_a,
                     where=[b < init_bal for b in y_a],
                     color='#f85149', alpha=0.1, interpolate=True)
    ax.fill_between(x_a, init_bal, y_a,
                     where=[b >= init_bal for b in y_a],
                     color='#3fb950', alpha=0.05, interpolate=True)

    # Meta ON — green/teal
    ax.plot(x_b, y_b, color='#3fb950', linewidth=2.5, alpha=0.95, label='[B] Meta AÇIK')
    ax.fill_between(x_b, init_bal, y_b,
                     where=[b >= init_bal for b in y_b],
                     color='#3fb950', alpha=0.15, interpolate=True)
    ax.fill_between(x_b, init_bal, y_b,
                     where=[b < init_bal for b in y_b],
                     color='#f85149', alpha=0.05, interpolate=True)

    # End markers
    if y_a:
        ax.plot(x_a[-1], y_a[-1], 'o', color='#f85149', markersize=8, zorder=5)
        ax.annotate(f'${y_a[-1]:.1f}', (x_a[-1], y_a[-1]),
                     textcoords="offset points", xytext=(10, -5),
                     fontsize=11, color='#f85149', fontweight='bold')
    if y_b:
        ax.plot(x_b[-1], y_b[-1], 'o', color='#3fb950', markersize=8, zorder=5)
        ax.annotate(f'${y_b[-1]:.1f}', (x_b[-1], y_b[-1]),
                     textcoords="offset points", xytext=(10, 5),
                     fontsize=11, color='#3fb950', fontweight='bold')

    # Styling
    ax.set_xlabel('Trade #', fontsize=12, color='#c9d1d9')
    ax.set_ylabel('Bakiye ($)', fontsize=12, color='#c9d1d9')
    ax.set_title(f'📊 Bakiye Değişimi — {coin} ({start_date.strftime("%Y-%m-%d")} → {end_date.strftime("%Y-%m-%d")})',
                  fontsize=14, color='#f0f6fc', pad=15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.15, color='#30363d')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent / f'equity_comparison_{coin}.png'
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"📊 Equity grafiği kaydedildi: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visual Backtest Simulator - Gerçek AI Modeli')
    parser.add_argument('--coin', type=str, default='BTC', help='Coin symbol (e.g., BTC, ETH)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to backtest')
    parser.add_argument('--speed', type=int, default=100, help='Animation speed (ms per candle)')
    parser.add_argument('--no-anim', action='store_true', help='Disable animation (faster)')
    parser.add_argument('--balance', type=float, default=100.0, help='Initial balance')
    parser.add_argument('--leverage', type=int, default=5, help='Leverage (futures only)')
    parser.add_argument('--no-hour-filter', action='store_true', help='Disable hour filter')
    parser.add_argument('--mode', type=str, default='futures', choices=['spot', 'futures'], 
                        help='Trading mode: spot or futures')
    parser.add_argument('--grid', action='store_true', help='Enable grid trading (spot only)')
    parser.add_argument('--grid-levels', type=int, default=2, choices=[2, 3, 4],
                        help='Number of grid levels (2-4)')
    # Meta-model arguments
    parser.add_argument('--meta-compare', action='store_true',
                        help='A/B comparison: meta-model ON vs OFF')
    parser.add_argument('--meta-threshold', type=float, default=0.50,
                        help='Meta-model threshold (default: 0.50)')
    
    args = parser.parse_args()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Create config based on mode
    config = SimConfig(
        trading_mode=args.mode,
        initial_balance=args.balance,
        leverage=args.leverage,
        animation_speed=args.speed,
        show_live=not args.no_anim,
        hour_filter_enabled=not args.no_hour_filter,
        grid_enabled=args.grid if args.mode == 'spot' else False,
        grid_levels=args.grid_levels
    )
    
    # A/B Comparison Mode
    if args.meta_compare:
        config.show_live = False  # Force no-anim for comparison
        run_comparison(args.coin, start_date, end_date, config, args.meta_threshold)
        return
    
    # Print mode info
    mode_str = f"🟢 SPOT" if args.mode == "spot" else f"🟡 FUTURES {args.leverage}x"
    grid_str = f" + Grid {args.grid_levels}L" if config.grid_enabled else ""
    print(f"\n{'='*60}")
    print(f"🎮 Visual Backtest - {args.coin} - {mode_str}{grid_str}")
    print(f"📅 {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} ({args.days} gün)")
    print(f"💰 Başlangıç: ${args.balance}")
    print(f"{'='*60}\n")
    
    backtester = RealAIBacktester(args.coin, start_date, end_date, config)
    backtester.run_simulation()


if __name__ == "__main__":
    main()
