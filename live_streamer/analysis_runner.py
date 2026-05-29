import asyncio
import logging
import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Import from existing project files
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

logger = logging.getLogger("analysis_runner")

class AnalysisRunner:
    def __init__(self, price_engine, interval=60):
        self.price_engine = price_engine
        self.interval = interval
        self.running = False
        
        self.models = {}
        self.meta_gate = None
        self.latest_analysis = {}
        self.callbacks = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_retraining = False
        
        # Streak counters for fun achievements
        self.streaks = {"ETH/USDT": {"wins": 0, "losses": 0}}
        self.last_price = {"ETH/USDT": None}
        self.last_signal = {"ETH/USDT": None}

    def register_callback(self, callback):
        self.callbacks.append(callback)

    async def initialize(self):
        """Load MoE models."""
        logger.info("🧠 Initializing AI Models for 15m and 1h timeframes...")
        
        # 1. Paths to models (aligned with app.py)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        autotrained_path = os.path.join(project_root, 'train_models', 'finalized_models', '3BranchApproach', '6try', 'AUTOTRAINED_LIVE_15M.pth')
        default_path = os.path.join(project_root, 'train_models', 'finalized_models', '3BranchApproach', '6try', 'BEST_MODEL_FINAL.pth')
        
        if os.path.exists(autotrained_path):
            logger.info("🧠 [BOOT] Found autotrained live fine-tuned weights. Prioritizing self-improved model...")
            active_15m = autotrained_path
        else:
            active_15m = default_path
            
        model_paths = {
            '15m': active_15m
        }
        
        # Model hyperparams
        model_params = {
            '15m': {'embed_dim': 96, 'dropout': 0.31}
        }
        
        # Try loading exact hyperparameters from CryptoMoeApp JSON
        for tf in ['15m']:
            params_file = os.path.join(project_root, 'train_models', 'CryptoMoeApp', f'best_params_{tf}.json')
            if os.path.exists(params_file):
                try:
                    with open(params_file) as f:
                        params = json.load(f)
                        model_params[tf] = {
                            'embed_dim': params.get('embed_dim', 128),
                            'dropout': params.get('dropout', 0.15)
                        }
                except Exception:
                    pass

        # 2. Load PyTorch MoE models
        for tf, path in model_paths.items():
            if os.path.exists(path):
                try:
                    p = model_params[tf]
                    logger.info(f"   🤖 Loading {tf} MoE Model (dim={p['embed_dim']}, drop={p['dropout']})...")
                    model = MultiBranchModel(embed_dim=p['embed_dim'], dropout=p['dropout']).to(self.device)
                    state_dict = torch.load(path, map_location=self.device, weights_only=True)
                    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    model.load_state_dict(clean_state)
                    model.eval()
                    self.models[tf] = model
                    logger.info(f"   ✅ {tf} model successfully loaded.")
                except Exception as e:
                    logger.error(f"   ❌ Failed to load {tf} model: {e}")
            else:
                logger.warning(f"   ⚠️ Model file not found: {path}")

    async def load_custom_model(self, timeframe: str, path: str):
        """Dynamically load a custom PyTorch MoE model from an absolute file path."""
        if timeframe != '15m':
            return False, "Only 15m timeframe is active."
            
        if not os.path.exists(path):
            return False, f"Model file not found at: {path}"
            
        try:
            # 1. Load the state dict
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # 2. Auto-detect embed_dim from weights
            # The conv layer weight has shape [embed_dim, in_channels, kernel_size]
            first_layer_key = 'cnn_branch.0.weight'
            if first_layer_key in clean_state:
                embed_dim = clean_state[first_layer_key].shape[0]
                logger.info(f"🔍 Auto-detected embed_dim={embed_dim} from state dict.")
            else:
                # Fallback to standard dimension
                embed_dim = 96 if timeframe == '15m' else 128
                logger.warning(f"⚠️ Could not find cnn_branch.0.weight in state dict. Using fallback embed_dim={embed_dim}")
            
            # Default dropout is fine since inference turns off dropout anyway
            dropout = 0.15
            
            # 3. Re-initialize model architecture
            model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(self.device)
            model.load_state_dict(clean_state)
            model.eval()
            
            # 4. Save into active models dict
            self.models[timeframe] = model
            logger.info(f"✅ Dynamically loaded custom {timeframe} model from {path} (embed_dim={embed_dim})")
            return True, f"Successfully loaded model (embed_dim={embed_dim}) from {path}"
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom model from {path}: {e}")
            return False, f"Failed to load model: {str(e)}"

    async def start(self):
        """Start the background analysis loop."""
        self.running = True
        asyncio.create_task(self._analysis_loop())
        logger.info("⚡ Live Analysis Engine started.")

    async def stop(self):
        self.running = False

    async def _analysis_loop(self):
        while self.running:
            try:
                logger.info("🔄 Running scheduled AI analysis cycle...")
                for symbol in ["ETH/USDT"]:
                    await self._analyze_symbol(symbol)
                    
                # Broadcast updates
                if self.callbacks:
                    payload = {
                        "type": "analysis",
                        "timestamp": datetime.now().isoformat(),
                        "data": self.latest_analysis
                    }
                    for cb in self.callbacks:
                        try:
                            if asyncio.iscoroutinefunction(cb):
                                await cb(payload)
                            else:
                                cb(payload)
                        except Exception as cb_err:
                            logger.error(f"❌ Analysis callback error: {cb_err}")
                            
            except Exception as e:
                logger.error(f"❌ Error in analysis runner loop: {e}")
                
            await asyncio.sleep(self.interval)

    def prepare_model_input(self, df_ai, cnn_window=12, lstm_window=120, tr_window=120):
        """Align input tensor structure with 3BranchApproach model requirements."""
        cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
        lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
        tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
        
        # Mean/std normalization (must mimic data_fetcher scaling)
        mean = df_ai.mean()
        std = df_ai.std()
        if 'Log_Ret' in mean:
            mean['Log_Ret'] = 0.0
        std[std == 0] = 1.0
        df_normalized = (df_ai - mean) / std
        
        data = df_normalized.values
        max_window = max(cnn_window, lstm_window, tr_window)
        
        if len(data) < max_window:
            return None, None, None
            
        t = len(data)
        x_cnn = data[t - cnn_window:t, cnn_cols]
        x_lstm = data[t - lstm_window:t, lstm_cols]
        x_tr = data[t - tr_window:t, tr_cols]
        
        # Convert to tensors
        x_cnn = torch.tensor(x_cnn, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_lstm = torch.tensor(x_lstm, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_tr = torch.tensor(x_tr, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return x_cnn, x_lstm, x_tr

    def calculate_risk(self, rsi, bb_pctb, vol_ratio):
        """Calculate risk scores based on extreme zone analysis (from subagent reports)."""
        reasons = []
        rsi_val = rsi / 100.0 if rsi > 1 else rsi
        
        if rsi_val < 0.35:
            reasons.append("RSI Extremely Oversold (<35%)")
        elif rsi_val > 0.66:
            reasons.append("RSI Extremely Overbought (>66%)")
            
        if bb_pctb is not None:
            if bb_pctb < 0.05:
                reasons.append("Bollinger Bottom Breakout (<0.05)")
            elif bb_pctb > 0.95:
                reasons.append("Bollinger Top Breakout (>0.95)")
                
        if vol_ratio is not None and vol_ratio > 2.0:
            reasons.append("Abnormal Volume Surge (>2.0x)")
            
        return {
            "high_risk": len(reasons) > 0,
            "reasons": reasons,
            "score": len(reasons)
        }

    async def _analyze_symbol(self, symbol):
        """Retrieve 500 candles, process indicators, run models, and evaluate risk/meta gates."""
        try:
            self.latest_analysis[symbol] = {}
            
            for timeframe in ["15m"]:
                months_back = 1.2
                df = await asyncio.to_thread(
                    get_crypto_history,
                    symbol=symbol,
                    timeframe=timeframe,
                    months_back=months_back,
                    exchange_name="binance"
                )
                
                if len(df) < 120:
                    logger.warning(f"⚠️ Yetersiz veri {symbol} ({timeframe}): {len(df)} mum")
                    continue
                    
                raw_df = df.tail(500)
                df_display, df_ai = prepare_dual_dataframes(raw_df)
                
                # 2. Extract technical indicators for HUD
                last_row = df_display.iloc[-1]
                last_ai_row = df_ai.iloc[-1]
                
                rsi = last_row.get("RSI", 50.0)
                bb_pctb = last_ai_row.get("BB_PctB", 0.5)
                vol_ratio = last_ai_row.get("Vol_Ratio", 1.0)
                macd_norm = last_ai_row.get("MACD_Norm", 0.0)
                atr_pct = last_ai_row.get("ATR_Pct", 0.0)
                
                # 3. Predict using MoE Model
                model = self.models.get(timeframe)
                pred_main = 0.0
                pred_cnn = 0.0
                pred_lstm = 0.0
                pred_tr = 0.0
                has_ai = False
                
                if model is not None:
                    x_cnn, x_lstm, x_tr = self.prepare_model_input(df_ai)
                    if x_cnn is not None and not np.isnan(df_ai.values).any() and not np.isinf(df_ai.values).any():
                        with torch.no_grad():
                            main_out, cnn_out, lstm_out, tr_out = model(x_cnn, x_lstm, x_tr)
                        
                        # Reverse log scaling (* 100 in training)
                        pred_main = main_out.item() / 100.0
                        pred_cnn = cnn_out.item() / 100.0
                        pred_lstm = lstm_out.item() / 100.0
                        pred_tr = tr_out.item() / 100.0
                        has_ai = True
                
                # 4. Generate Signal
                threshold = 0.003  # 0.30% prediction threshold
                signal = "HOLD"
                signal_text = "Idle"
                
                if pred_main > threshold:
                    signal = "BUY"
                    signal_text = "Bullish Outbreak"
                elif pred_main < -threshold:
                    signal = "SELL"
                    signal_text = "Bearish Outbreak"
                    
                # 5. Risk Assessment & Meta Gate Checks
                risk = self.calculate_risk(rsi, bb_pctb, vol_ratio)
                meta_allowed = True
                meta_prob = 1.0
                meta_msg = "Gate Disabled"
                
                # Streak achievements based on symbol + timeframe
                streak_key = f"{symbol}_{timeframe}"
                if streak_key not in self.streaks:
                    self.streaks[streak_key] = {"wins": 0, "losses": 0}
                if streak_key not in self.last_price:
                    self.last_price[streak_key] = None
                if streak_key not in self.last_signal:
                    self.last_signal[streak_key] = None
                    
                streak_event = None
                current_price = self.price_engine.latest_data.get(symbol, {}).get("price", last_row.get("Close", 0.0))
                
                if self.last_price[streak_key] is not None and self.last_signal[streak_key] is not None:
                    price_move = current_price - self.last_price[streak_key]
                    sig = self.last_signal[streak_key]
                    
                    if (sig == "BUY" and price_move > 0) or (sig == "SELL" and price_move < 0):
                        self.streaks[streak_key]["wins"] += 1
                        self.streaks[streak_key]["losses"] = 0
                        if self.streaks[streak_key]["wins"] >= 3:
                            streak_event = f"🔥 {symbol} ({timeframe}) Sniper Streak: {self.streaks[streak_key]['wins']} hits!"
                    elif sig != "HOLD" and ((sig == "BUY" and price_move < 0) or (sig == "SELL" and price_move > 0)):
                        self.streaks[streak_key]["losses"] += 1
                        self.streaks[streak_key]["wins"] = 0
                        if self.streaks[streak_key]["losses"] >= 3:
                            streak_event = f"💀 {symbol} ({timeframe}) Drawdown Streak: {self.streaks[streak_key]['losses']} misses."
                            asyncio.create_task(self.run_autonomous_retrain())
                            
                if signal != "HOLD":
                    self.last_price[streak_key] = current_price
                    self.last_signal[streak_key] = signal
                
                # Save into timeframe payload
                self.latest_analysis[symbol][timeframe] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "price": current_price,
                    "signal": signal,
                    "signal_text": signal_text,
                    "risk": risk,
                    "meta": {
                        "allowed": meta_allowed,
                        "probability": meta_prob,
                        "message": meta_msg
                    },
                    "brain": {
                        "has_ai": has_ai,
                        "main": pred_main,
                        "cnn": pred_cnn,
                        "lstm": pred_lstm,
                        "tr": pred_tr
                    },
                    "indicators": {
                        "rsi": rsi,
                        "bb_pctb": bb_pctb,
                        "vol_ratio": vol_ratio,
                        "macd_norm": macd_norm,
                        "atr_pct": atr_pct
                    },
                    "achievements": {
                        "win_streak": self.streaks[streak_key]["wins"],
                        "loss_streak": self.streaks[streak_key]["losses"],
                        "event": streak_event
                    },
                    "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            logger.info(f"📊 Completed multi-model analysis for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    async def _broadcast_system_alert(self, action, message):
        """Helper to broadcast custom otonom retraining alerts to the WebSocket pool."""
        payload = {
            "type": "retrain_status",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "action": action,
                "message": message
            }
        }
        for cb in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(payload)
                else:
                    cb(payload)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def run_autonomous_retrain(self):
        """Asynchronously fetches the last 500 candles, runs a micro-fine-tuning PyTorch loop,
        saves the optimized weights, and hot-swaps them instantly with live WebSocket logs."""
        if self.is_retraining:
            return
            
        self.is_retraining = True
        logger.info("🧠 [AUTONOMOUS RETRAIN] Prediction drawdown streak detected. Retraining starting...")
        
        try:
            # 1. Broadcast starting alert via WebSocket
            await self._broadcast_system_alert(
                "SYSTEM TRIGGER", 
                "🧠 Loss streak threshold breached! Launching otonom fine-tuning to adapt to the new market regime..."
            )
            
            # 2. Fetch last 500 15m candles from Binance perpetuals
            exchange = self.price_engine.exchange
            if not exchange:
                raise ValueError("Exchange connection not initialized.")
                
            logger.info("🧠 [AUTONOMOUS RETRAIN] Fetching 500 historical 15m candles...")
            ohlcv = await exchange.fetch_ohlcv("ETH/USDT", timeframe='15m', limit=500)
            
            # 3. Build dataframe and indicators
            data_list = []
            for candle in ohlcv:
                data_list.append({
                    "Timestamp": candle[0],
                    "Open": candle[1],
                    "High": candle[2],
                    "Low": candle[3],
                    "Close": candle[4],
                    "Volume": candle[5]
                })
            df = pd.DataFrame(data_list)
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            
            df_display, df_ai = prepare_dual_dataframes(df)
            
            # 4. Generate normalized training tensors
            cnn_window, lstm_window, tr_window = 12, 120, 120
            max_window = max(cnn_window, lstm_window, tr_window)
            
            if len(df_ai) < max_window + 10:
                raise ValueError(f"Insufficient training samples: {len(df_ai)}")
                
            mean = df_ai.mean()
            std = df_ai.std()
            if 'Log_Ret' in mean:
                mean['Log_Ret'] = 0.0
            std[std == 0] = 1.0
            df_normalized = (df_ai - mean) / std
            
            x_cnn_list, x_lstm_list, x_tr_list, y_list = [], [], [], []
            cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
            lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
            tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
            
            data_vals = df_normalized.values
            raw_close = df['Close'].values
            
            for i in range(max_window, len(df_normalized) - 1):
                x_cnn = data_vals[i - cnn_window:i, cnn_cols]
                x_lstm = data_vals[i - lstm_window:i, lstm_cols]
                x_tr = data_vals[i - tr_window:i, tr_cols]
                
                ret = np.log(raw_close[i + 1] / raw_close[i]) * 100.0
                
                x_cnn_list.append(x_cnn)
                x_lstm_list.append(x_lstm)
                x_tr_list.append(x_tr)
                y_list.append(ret)
                
            x_cnn_t = torch.tensor(np.array(x_cnn_list), dtype=torch.float32).to(self.device)
            x_lstm_t = torch.tensor(np.array(x_lstm_list), dtype=torch.float32).to(self.device)
            x_tr_t = torch.tensor(np.array(x_tr_list), dtype=torch.float32).to(self.device)
            y_t = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1).to(self.device)
            
            # 5. Execute PyTorch Micro Fine-Tuning
            model = self.models.get("15m")
            if model is None:
                raise ValueError("Active 15m model not loaded.")
                
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            criterion = torch.nn.MSELoss()
            
            logger.info("🧠 [AUTONOMOUS RETRAIN] Running 5 micro-fine-tuning epochs on active model parameters...")
            epochs = 5
            for epoch in range(epochs):
                optimizer.zero_grad()
                main_out, cnn_out, lstm_out, tr_out = model(x_cnn_t, x_lstm_t, x_tr_t)
                loss = criterion(main_out.view(-1), y_t.view(-1))
                loss.backward()
                optimizer.step()
                
            # 6. Save optimized state weights
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(project_root, 'train_models', 'finalized_models', '3BranchApproach', '6try')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'AUTOTRAINED_LIVE_15M.pth')
            torch.save(model.state_dict(), save_path)
            
            # 7. Hot-swap active inference weights
            model.eval()
            self.models["15m"] = model
            
            logger.info(f"🧠 [AUTONOMOUS RETRAIN] Successfully saved and hot-swapped weights to: {save_path}")
            
            await self._broadcast_system_alert(
                "SYSTEM SUCCESS",
                "🧠 Otonom retraining complete! 15m MoE weights successfully optimized and hot-swapped to new regime."
            )
            
        except Exception as err:
            logger.error(f"❌ Autonomous retraining failed: {err}")
            try:
                await self._broadcast_system_alert(
                    "SYSTEM ERROR",
                    f"❌ Otonom retraining failed: {str(err)}"
                )
            except Exception:
                pass
        finally:
            self.is_retraining = False
