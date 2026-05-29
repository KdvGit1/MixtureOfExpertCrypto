"""
Binance Spot Auto Trading Bot
Uses 15m MoE (Mixture of Experts) prediction model for BTC/USDT trading.
SPOT ONLY - No margin trading.

Configuration:
- Testnet mode (Binance testnet)
- 20% position sizing
- 0.003 (0.30%) prediction threshold
- 0.1% trading fee (0.075% with BNB)

Smart Exit Strategy:
- Tracks entry price for each position
- Only sells if either: bullish profit target OR minimum profit to cover fees
- Never sells at a loss unless stop-loss triggered
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from decimal import Decimal, ROUND_DOWN

import ccxt
import numpy as np
import pandas as pd
import torch
import requests  # For Telegram API

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# ============================================
# CONFIGURATION
# ============================================
@dataclass
class TradingConfig:
    """Bot configuration settings."""
    # API Settings (loaded from .env)
    api_key: str = ""
    api_secret: str = ""
    
    # Trading Mode
    testnet: bool = True  # Use Binance testnet
    dry_run: bool = True  # Paper trading mode (no real orders)
    
    # Trading Parameters
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    position_pct: float = 0.20  # 20% of wallet balance per trade
    
    # Prediction Thresholds (calculated from BTC data analysis)
    # 0.003 = 0.30% predicted move required to trade
    # This covers 0.20% round-trip fees and provides ~0.37% avg net profit
    prediction_threshold: float = 0.003
    
    # Prediction Scaling Factor
    # Model tends to overpredict movements - scale down for realistic targets
    # 0.5 = use 50% of predicted value (e.g., +0.4% prediction -> +0.2% target)
    prediction_scale: float = 0.5
    
    # Fee Settings
    base_fee_rate: float = 0.001  # 0.1% standard Binance fee
    bnb_fee_rate: float = 0.00075  # 0.075% with BNB discount
    use_bnb_for_fees: bool = True
    
    # Smart Exit Settings
    # Minimum profit required to exit position (covers round-trip fees + buffer)
    # 0.0025 = 0.25% (covers 0.20% fees + 0.05% safety margin)
    min_profit_to_exit: float = 0.0025
    # Stop-loss: maximum loss before forced exit (negative value)
    stop_loss_pct: float = -0.05  # -5% = exit to prevent further losses
    
    # Telegram Settings
    telegram_bot_token: str = ""  # Get from @BotFather
    telegram_chat_id: str = ""    # Get from @userinfobot or @getidsbot
    telegram_enabled: bool = True
    
    # Safety Limits
    max_daily_trades: int = 20
    min_balance_usdt: float = 10.0  # Minimum USDT to keep
    min_balance_btc: float = 0.0001  # Minimum BTC to keep
    
    # Timing
    loop_interval_seconds: int = 60  # Check every minute
    
# ============================================
# LOGGING SETUP
# ============================================
def setup_logging() -> logging.Logger:
    """Configure logging for the bot."""
    log_dir = PROJECT_ROOT / "bot_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TradingBot")

# ============================================
# MODEL LOADING
# ============================================
MODEL_MAP = {
    '15m': PROJECT_ROOT / 'train_models' / 'CryptoMoeApp' / '15m_Moe.pth',
    '1h': PROJECT_ROOT / 'train_models' / 'CryptoMoeApp' / '1h_Moe.pth'
}

MODEL_PARAMS = {
    '15m': {'embed_dim': 96, 'dropout': 0.31},
    '1h': {'embed_dim': 128, 'dropout': 0.32}
}

# Load params from JSON files if they exist
for tf in MODEL_MAP.keys():
    params_file = PROJECT_ROOT / 'train_models' / 'CryptoMoeApp' / f'best_params_{tf}.json'
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
            MODEL_PARAMS[tf] = {
                'embed_dim': params.get('embed_dim', 128),
                'dropout': params.get('dropout', 0.15)
            }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(timeframe: str) -> Optional[MultiBranchModel]:
    """Load the MoE model for a given timeframe."""
    if timeframe not in MODEL_MAP:
        return None
    
    model_path = MODEL_MAP[timeframe]
    if not model_path.exists():
        return None
    
    params = MODEL_PARAMS.get(timeframe, {'embed_dim': 128, 'dropout': 0.15})
    
    model = MultiBranchModel(embed_dim=params['embed_dim'], dropout=params['dropout']).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()
    
    return model

# ============================================
# TELEGRAM NOTIFICATIONS
# ============================================
class TelegramNotifier:
    """Send trade notifications via Telegram."""
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
    def send_message(self, message: str) -> bool:
        """Send a message to Telegram."""
        if not self.enabled:
            return False
        
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(self.api_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    
    def send_buy_alert(self, symbol: str, entry_price: float, amount: float, 
                       target_pct: float, stop_loss_pct: float, 
                       prediction_pct: float, confidence: float) -> bool:
        """Send BUY trade alert with levels."""
        target_price = entry_price * (1 + target_pct)
        stop_price = entry_price * (1 + stop_loss_pct)
        usd_value = entry_price * amount
        
        message = f"""
🟢 <b>BTC ALIŞ</b>

📊 <b>İşlem Detayları:</b>
• Miktar: {amount:.8f} BTC
• Değer: ${usd_value:,.2f}
• AI Tahmin: {prediction_pct:+.2f}%
• Güven: {confidence:.0f}%

📍 <b>Seviyeler:</b>
• Giriş: ${entry_price:,.2f}
• Hedef: ${target_price:,.2f} ({target_pct*100:+.2f}%)
• Stop-Loss: ${stop_price:,.2f} ({stop_loss_pct*100:.1f}%)

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())
    
    def send_sell_alert(self, symbol: str, exit_price: float, amount: float,
                        entry_price: float, realized_pnl_pct: float,
                        reason: str, cumulative_pnl_pct: float, 
                        total_trades: int) -> bool:
        """Send SELL trade alert with P&L."""
        usd_value = exit_price * amount
        profit_usd = (exit_price - entry_price) * amount
        emoji = "🟢" if realized_pnl_pct >= 0 else "🔴"
        
        message = f"""
🔴 <b>BTC SATIŞ</b>

📊 <b>İşlem Detayları:</b>
• Miktar: {amount:.8f} BTC
• Değer: ${usd_value:,.2f}
• Giriş: ${entry_price:,.2f}
• Çıkış: ${exit_price:,.2f}

{emoji} <b>Bu İşlem P&L:</b> {realized_pnl_pct:+.3f}% (${profit_usd:+,.2f})
📝 Sebep: {reason}

📈 <b>Toplam İstatistik:</b>
• İşlem Sayısı: {total_trades}
• Kümülatif P&L: {cumulative_pnl_pct:+.3f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())

# ============================================
# TRADING BOT CLASS
# ============================================
class SpotTradingBot:
    """Binance Spot Trading Bot using 15m AI Model."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logging()
        self.model: Optional[MultiBranchModel] = None
        self.exchange: Optional[ccxt.binance] = None
        
        # Telegram notifier
        self.telegram = TelegramNotifier(
            config.telegram_bot_token,
            config.telegram_chat_id,
            config.telegram_enabled
        )
        
        # Trade tracking
        self.daily_trades = 0
        self.last_trade_time: Optional[datetime] = None
        self.last_prediction_time: Optional[datetime] = None
        self.trade_history: List[Dict] = []
        
        # Cumulative P&L tracking
        self.cumulative_pnl_pct: float = 0.0
        self.total_trades_completed: int = 0
        
        # Position state with entry tracking
        self.current_position: str = "FLAT"  # FLAT, LONG (holding BTC)
        self.entry_price: Optional[float] = None  # Price when BTC was bought
        self.entry_amount: Optional[float] = None  # BTC amount bought
        self.entry_time: Optional[datetime] = None  # When position was opened
        self.entry_prediction: Optional[float] = None  # Original prediction when bought
        
    def initialize(self) -> bool:
        """Initialize the bot: load model, connect to exchange."""
        self.logger.info("=" * 50)
        self.logger.info("🚀 Initializing Binance Spot Trading Bot")
        self.logger.info(f"   Mode: {'TESTNET' if self.config.testnet else 'LIVE'}")
        self.logger.info(f"   Dry Run: {self.config.dry_run}")
        self.logger.info(f"   Symbol: {self.config.symbol}")
        self.logger.info(f"   Timeframe: {self.config.timeframe}")
        self.logger.info(f"   Position Size: {self.config.position_pct * 100:.0f}%")
        self.logger.info(f"   Prediction Threshold: {self.config.prediction_threshold * 100:.2f}%")
        self.logger.info(f"   Prediction Scale: {self.config.prediction_scale:.0%} (reduces overprediction)")
        self.logger.info("=" * 50)
        
        # Load AI model
        self.logger.info("🧠 Loading AI model...")
        self.model = load_model(self.config.timeframe)
        if self.model is None:
            self.logger.error(f"❌ Failed to load model for {self.config.timeframe}")
            return False
        self.logger.info("✅ AI model loaded successfully")
        
        # Initialize exchange connection
        self.logger.info("🔗 Connecting to Binance...")
        try:
            exchange_params = {
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            
            if self.config.testnet:
                exchange_params['options']['sandboxMode'] = True
                # Binance testnet URLs
                exchange_params['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api',
                    }
                }
            
            self.exchange = ccxt.binance(exchange_params)
            
            if self.config.testnet:
                self.exchange.set_sandbox_mode(True)
            
            # Test connection
            self.exchange.load_markets()
            self.logger.info(f"✅ Connected to Binance {'Testnet' if self.config.testnet else 'Mainnet'}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Binance: {e}")
            return False
        
        return True
    
    def get_balance(self, asset: str) -> float:
        """Get spot wallet balance for an asset."""
        try:
            balance = self.exchange.fetch_balance()
            free = float(balance.get(asset, {}).get('free', 0))
            return free
        except Exception as e:
            self.logger.error(f"❌ Error fetching {asset} balance: {e}")
            return 0.0
    
    def get_current_price(self) -> Optional[float]:
        """Get current BTC/USDT price."""
        try:
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"❌ Error fetching price: {e}")
            return None
    
    def calculate_fee(self, amount: float) -> float:
        """Calculate trading fee for an amount."""
        fee_rate = self.config.bnb_fee_rate if self.config.use_bnb_for_fees else self.config.base_fee_rate
        return amount * fee_rate
    
    def get_prediction(self) -> Optional[Dict]:
        """Fetch data and run AI prediction."""
        try:
            # Fetch historical data (need 500 candles for EMA 200 warmup)
            tf_minutes = {'15m': 15, '1h': 60}.get(self.config.timeframe, 15)
            minutes_needed = 500 * tf_minutes
            months_needed = (minutes_needed / (30 * 24 * 60)) * 1.1  # 10% buffer
            
            # Get data (use spot format for symbol)
            spot_symbol = self.config.symbol.replace("/", "")  # BTC/USDT -> BTCUSDT
            df = get_crypto_history(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                months_back=months_needed,
                exchange_name="binance"
            )
            
            if len(df) < 120:
                self.logger.warning(f"⚠️ Insufficient data: {len(df)} candles")
                return None
            
            # Prepare data
            df_display, df_ai = prepare_dual_dataframes(df)
            
            if df_ai.isnull().values.any() or np.isinf(df_ai.values).any():
                self.logger.warning("⚠️ Data contains NaN/Inf values")
                return None
            
            # Prepare model input
            cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
            lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
            tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
            
            # Normalize
            mean = df_ai.mean()
            std = df_ai.std()
            if 'Log_Ret' in mean:
                mean['Log_Ret'] = 0.0
            std[std == 0] = 1.0
            df_normalized = (df_ai - mean) / std
            
            data = df_normalized.values
            t = len(data)
            
            x_cnn = torch.tensor(data[t-12:t, cnn_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_lstm = torch.tensor(data[t-120:t, lstm_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_tr = torch.tensor(data[t-120:t, tr_cols], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Run prediction
            with torch.no_grad():
                pred_main, pred_cnn, pred_lstm, pred_tr = self.model(x_cnn, x_lstm, x_tr)
                
                # Reverse training scaling
                prediction = pred_main.item() / 100.0
                
                # Calculate confidence from branch agreement
                branches = [pred_cnn.item()/100.0, pred_lstm.item()/100.0, pred_tr.item()/100.0]
                signs = [1 if b > 0 else -1 for b in branches]
                agreement = abs(sum(signs)) / 3.0
                confidence = agreement * 100
            
            last_candle = df_display.iloc[-1]
            current_price = float(last_candle['Close'])
            
            return {
                'prediction': prediction,  # Log return prediction
                'prediction_pct': prediction * 100,  # As percentage
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting prediction: {e}")
            return None
    
    def calculate_current_pnl(self, current_price: float) -> Optional[float]:
        """
        Calculate current profit/loss percentage for open position.
        Returns None if no position, else returns P&L as decimal (e.g., 0.01 = 1%).
        """
        if self.current_position != 'LONG' or self.entry_price is None:
            return None
        
        pnl = (current_price - self.entry_price) / self.entry_price
        return pnl
    
    def should_trade(self, prediction: Dict) -> tuple:
        """
        Decide whether to trade based on prediction and current P&L.
        Returns: (action, reason)
        - action: 'BUY', 'SELL', or 'HOLD'
        
        Smart Exit Logic:
        - BUY: When prediction > threshold and not already holding
        - SELL: When one of these conditions is met:
            1. Prediction negative AND current P&L >= min_profit_to_exit (fee coverage)
            2. Prediction still positive AND current P&L >= original prediction target
            3. Stop-loss triggered (P&L < stop_loss_pct)
        - HOLD: Otherwise, wait for better exit opportunity
        """
        pred = prediction['prediction']
        confidence = prediction['confidence']
        current_price = prediction['price']
        threshold = self.config.prediction_threshold
        
        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return ('HOLD', 'Daily trade limit reached')
        
        # Calculate current P&L for open positions
        current_pnl = self.calculate_current_pnl(current_price)
        
        # ========== SELL LOGIC (when holding BTC) ==========
        if self.current_position == 'LONG':
            if current_pnl is not None:
                pnl_pct = current_pnl * 100
                
                # Log current P&L
                self.logger.info(f"   📊 Position P&L: {pnl_pct:+.3f}% (entry: ${self.entry_price:,.2f}, now: ${current_price:,.2f})")
                
                # STOP-LOSS: Exit immediately if losing too much
                if current_pnl <= self.config.stop_loss_pct:
                    return ('SELL', f'🛑 STOP-LOSS triggered: {pnl_pct:.2f}% (limit: {self.config.stop_loss_pct*100:.1f}%)')
                
                # TAKE PROFIT: If reached original prediction target
                if self.entry_prediction and current_pnl >= self.entry_prediction:
                    return ('SELL', f'🎯 Target reached: {pnl_pct:+.2f}% (target was {self.entry_prediction*100:+.2f}%)')
                
                # SMART EXIT: If prediction turned negative, exit only if covering fees
                if pred < 0:  # Bearish signal
                    if current_pnl >= self.config.min_profit_to_exit:
                        return ('SELL', f'📉 Bearish + min profit: {pnl_pct:+.2f}% >= {self.config.min_profit_to_exit*100:.2f}%')
                    else:
                        return ('HOLD', f'⏳ Bearish but waiting for min profit ({pnl_pct:+.2f}% < {self.config.min_profit_to_exit*100:.2f}%)')
                
                # HOLD: Still bullish prediction, keep position
                if pred > threshold:
                    return ('HOLD', f'📈 Still bullish ({pred*100:+.2f}%), holding for target')
                
                # NEUTRAL prediction: only exit if profitable enough
                if current_pnl >= self.config.min_profit_to_exit:
                    return ('SELL', f'😐 Neutral signal + min profit: {pnl_pct:+.2f}%')
                else:
                    return ('HOLD', f'⏳ Neutral but waiting for min profit ({pnl_pct:+.2f}%)')
            
            return ('HOLD', 'Position open, waiting for exit signal')
        
        # ========== BUY LOGIC (when not holding BTC) ==========
        if self.current_position == 'FLAT':
            # Bullish signal - enter new position
            if pred > threshold:
                return ('BUY', f'📈 Bullish signal: +{pred*100:.3f}%')
            
            # Bearish or neutral - don't buy
            return ('HOLD', f'No buy signal (pred: {pred*100:+.3f}%)')
        
        return ('HOLD', 'No clear signal')
    
    def execute_buy(self, usdt_amount: float) -> Optional[Dict]:
        """Execute a market buy order for BTC."""
        try:
            price = self.get_current_price()
            if price is None:
                return None
            
            # Calculate BTC amount (minus fee)
            fee = self.calculate_fee(usdt_amount)
            net_usdt = usdt_amount - fee
            btc_amount = net_usdt / price
            
            # Round to acceptable precision (8 decimals for BTC)
            btc_amount = float(Decimal(str(btc_amount)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
            
            self.logger.info(f"📈 BUY ORDER: {btc_amount:.8f} BTC @ ~${price:,.2f}")
            self.logger.info(f"   USDT: ${usdt_amount:.2f}, Fee: ${fee:.2f}")
            
            if self.config.dry_run:
                self.logger.info("   [DRY RUN - Order not placed]")
                order = {
                    'id': f'dry_run_{int(time.time())}',
                    'symbol': self.config.symbol,
                    'side': 'buy',
                    'type': 'market',
                    'amount': btc_amount,
                    'price': price,
                    'cost': usdt_amount,
                    'fee': {'cost': fee, 'currency': 'USDT'},
                    'status': 'closed'
                }
            else:
                order = self.exchange.create_market_buy_order(
                    self.config.symbol,
                    btc_amount
                )
                self.logger.info(f"   ✅ Order placed: {order.get('id')}")
            
            # Update position tracking
            self.current_position = 'LONG'
            self.entry_price = price
            self.entry_amount = btc_amount
            self.entry_time = datetime.now()
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            
            self.logger.info(f"   📍 Entry tracked: {btc_amount:.8f} BTC @ ${price:,.2f}")
            
            # Send Telegram notification
            self.telegram.send_buy_alert(
                symbol=self.config.symbol,
                entry_price=price,
                amount=btc_amount,
                target_pct=self.entry_prediction if self.entry_prediction else self.config.prediction_threshold,
                stop_loss_pct=self.config.stop_loss_pct,
                prediction_pct=(self.entry_prediction or 0) * 100,
                confidence=0  # Will be set from prediction
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"❌ Buy order failed: {e}")
            return None
    
    def execute_sell(self, btc_amount: float) -> Optional[Dict]:
        """Execute a market sell order for BTC."""
        try:
            price = self.get_current_price()
            if price is None:
                return None
            
            # Round to acceptable precision
            btc_amount = float(Decimal(str(btc_amount)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
            usdt_value = btc_amount * price
            fee = self.calculate_fee(usdt_value)
            
            self.logger.info(f"📉 SELL ORDER: {btc_amount:.8f} BTC @ ~${price:,.2f}")
            self.logger.info(f"   Value: ${usdt_value:.2f}, Fee: ${fee:.2f}")
            
            if self.config.dry_run:
                self.logger.info("   [DRY RUN - Order not placed]")
                order = {
                    'id': f'dry_run_{int(time.time())}',
                    'symbol': self.config.symbol,
                    'side': 'sell',
                    'type': 'market',
                    'amount': btc_amount,
                    'price': price,
                    'cost': usdt_value,
                    'fee': {'cost': fee, 'currency': 'USDT'},
                    'status': 'closed'
                }
            else:
                order = self.exchange.create_market_sell_order(
                    self.config.symbol,
                    btc_amount
                )
                self.logger.info(f"   ✅ Order placed: {order.get('id')}")
            
            # Calculate and log realized P&L
            realized_pnl = 0.0
            entry_price_saved = self.entry_price
            if self.entry_price:
                realized_pnl = ((price - self.entry_price) / self.entry_price) * 100
                self.logger.info(f"   💰 Realized P&L: {realized_pnl:+.3f}%")
                
                # Update cumulative P&L
                self.cumulative_pnl_pct += realized_pnl
                self.total_trades_completed += 1
            
            # Reset position tracking
            self.current_position = 'FLAT'
            self.entry_price = None
            self.entry_amount = None
            self.entry_time = None
            self.entry_prediction = None
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            
            # Send Telegram notification
            if entry_price_saved:
                self.telegram.send_sell_alert(
                    symbol=self.config.symbol,
                    exit_price=price,
                    amount=btc_amount,
                    entry_price=entry_price_saved,
                    realized_pnl_pct=realized_pnl,
                    reason="Trade closed",
                    cumulative_pnl_pct=self.cumulative_pnl_pct,
                    total_trades=self.total_trades_completed
                )
            
            return order
            
        except Exception as e:
            self.logger.error(f"❌ Sell order failed: {e}")
            return None
    
    def check_15min_candle(self) -> bool:
        """Check if we're at the start of a new 15-minute candle."""
        now = datetime.now()
        # Check if we're within first minute of a 15-minute period
        return now.minute % 15 < 1
    
    def run_trading_loop(self):
        """Main trading loop."""
        self.logger.info("🔄 Starting trading loop...")
        self.logger.info(f"   Waiting for 15-minute candle alignment...")
        
        last_candle_minute = -1
        
        while True:
            try:
                now = datetime.now()
                candle_minute = (now.minute // 15) * 15
                
                # Reset daily counter at midnight
                if now.hour == 0 and now.minute == 0:
                    self.daily_trades = 0
                    self.logger.info("📅 New day - trade counter reset")
                
                # Only execute at start of new 15-minute candle
                if candle_minute != last_candle_minute and self.check_15min_candle():
                    last_candle_minute = candle_minute
                    
                    self.logger.info("-" * 40)
                    self.logger.info(f"🕐 New 15m candle: {now.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Get current balances
                    usdt_balance = self.get_balance('USDT')
                    btc_balance = self.get_balance('BTC')
                    price = self.get_current_price()
                    
                    if price:
                        total_usd = usdt_balance + (btc_balance * price)
                        self.logger.info(f"💰 Balance: ${usdt_balance:.2f} USDT + {btc_balance:.8f} BTC (${btc_balance*price:.2f})")
                        self.logger.info(f"   Total: ${total_usd:.2f}")
                    
                    # Get AI prediction
                    prediction = self.get_prediction()
                    if prediction is None:
                        self.logger.warning("⚠️ Could not get prediction, skipping...")
                        time.sleep(self.config.loop_interval_seconds)
                        continue
                    
                    self.logger.info(f"🧠 AI Prediction: {prediction['prediction_pct']:+.3f}% (confidence: {prediction['confidence']:.0f}%)")
                    
                    # Determine action
                    action, reason = self.should_trade(prediction)
                    self.logger.info(f"📊 Decision: {action} - {reason}")
                    
                    # Execute trade if needed
                    if action == 'BUY':
                        available_usdt = usdt_balance - self.config.min_balance_usdt
                        trade_amount = available_usdt * self.config.position_pct
                        
                        if trade_amount > 10:  # Minimum order ~$10
                            order = self.execute_buy(trade_amount)
                            if order:
                                # Store SCALED prediction for target tracking
                                # Scale down to avoid overprediction issues
                                raw_pred = prediction['prediction']
                                scaled_pred = raw_pred * self.config.prediction_scale
                                self.entry_prediction = scaled_pred
                                self.logger.info(f"   🎯 Target: {raw_pred*100:+.3f}% x {self.config.prediction_scale:.0%} = {scaled_pred*100:+.3f}%")
                                self.trade_history.append({
                                    'time': datetime.now(),
                                    'action': 'BUY',
                                    'prediction': prediction,
                                    'order': order,
                                    'entry_price': self.entry_price
                                })
                        else:
                            self.logger.warning(f"⚠️ Insufficient USDT for trade: ${trade_amount:.2f}")
                    
                    elif action == 'SELL':
                        available_btc = btc_balance - self.config.min_balance_btc
                        trade_amount = available_btc * self.config.position_pct
                        
                        if trade_amount * price > 10:  # Minimum order ~$10
                            order = self.execute_sell(trade_amount)
                            if order:
                                self.trade_history.append({
                                    'time': datetime.now(),
                                    'action': 'SELL',
                                    'prediction': prediction,
                                    'order': order
                                })
                        else:
                            self.logger.warning(f"⚠️ Insufficient BTC for trade: {trade_amount:.8f}")
                
                # Sleep until next check
                time.sleep(self.config.loop_interval_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("⛔ Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def stop(self):
        """Stop the bot gracefully."""
        self.logger.info("👋 Shutting down trading bot...")
        self.logger.info(f"   Total trades today: {self.daily_trades}")
        self.logger.info(f"   Trade history: {len(self.trade_history)} trades")

# ============================================
# ENV FILE LOADER
# ============================================
def load_env(env_path: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
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
    """Main entry point."""
    print("=" * 60)
    print("  Binance Spot Trading Bot - 15m AI Model")
    print("  SPOT ONLY - BTC/USDT")
    print("=" * 60)
    
    # Load configuration from .env
    env_path = PROJECT_ROOT / '.env'
    env_vars = load_env(env_path)
    
    # Create config
    config = TradingConfig(
        api_key=env_vars.get('BINANCE_API_KEY', ''),
        api_secret=env_vars.get('BINANCE_API_SECRET', ''),
        testnet=env_vars.get('TESTNET', 'true').lower() == 'true',
        dry_run=env_vars.get('DRY_RUN', 'true').lower() == 'true',
        use_bnb_for_fees=env_vars.get('USE_BNB_FOR_FEES', 'true').lower() == 'true',
        position_pct=float(env_vars.get('POSITION_PCT', '0.20')),
        prediction_threshold=float(env_vars.get('PREDICTION_THRESHOLD', '0.003')),
        prediction_scale=float(env_vars.get('PREDICTION_SCALE', '0.5')),
        min_profit_to_exit=float(env_vars.get('MIN_PROFIT_TO_EXIT', '0.0025')),
        stop_loss_pct=float(env_vars.get('STOP_LOSS_PCT', '-0.05')),
        # Telegram settings
        telegram_bot_token=env_vars.get('TELEGRAM_BOT_TOKEN', ''),
        telegram_chat_id=env_vars.get('TELEGRAM_CHAT_ID', ''),
        telegram_enabled=env_vars.get('TELEGRAM_ENABLED', 'true').lower() == 'true',
    )
    
    # Validate API keys
    if not config.api_key or not config.api_secret:
        print("\n⚠️  No API keys found!")
        print("   Please create a .env file with:")
        print("   BINANCE_API_KEY=your_key")
        print("   BINANCE_API_SECRET=your_secret")
        print("\n   For testnet, get keys at: https://testnet.binance.vision/")
        return
    
    # Create and run bot
    bot = SpotTradingBot(config)
    
    if not bot.initialize():
        print("❌ Failed to initialize bot")
        return
    
    try:
        bot.run_trading_loop()
    finally:
        bot.stop()

if __name__ == "__main__":
    main()
