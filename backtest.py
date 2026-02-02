"""
Backtest Module for AI Models (Auto All-Models Version)
Tests all AI prediction models from kaggle_outputs on historical cryptocurrency data.

Usage:
    # Tüm modelleri test et
    python backtest.py                              # 40 model (tüm coinler, tüm timeframe'ler)
    python backtest.py --charts                     # + Her model için grafik üret
    
    # Filtreli test
    python backtest.py --coin BTC                   # Sadece BTC modelleri (15m + 1h)
    python backtest.py --timeframe 1h               # Sadece 1h modelleri (tüm coinler)
    python backtest.py --coin BTC --timeframe 1h    # Tek model
    python backtest.py --months 6                   # 6 aylık veri (varsayılan: 3)
    
    # Strateji Karşılaştırması (HODL vs Grid 2/3/4 vs Normal Trading)
    python backtest.py --compare                    # TÜM modeller için karşılaştır
    python backtest.py --compare --timeframe 15m    # Sadece 15m modelleri karşılaştır
    python backtest.py --coin BTC --timeframe 15m --compare   # Tek model karşılaştır
    python backtest.py --compare --months 6         # 6 aylık veri ile
    
    Karşılaştırılan Stratejiler:
    - HODL (al ve tut)
    - Grid (2 levels) - %60/%40 alım, +2%/+4% satış
    - Grid (3 levels) - %50/%30/%20 alım, +1.5%/+3%/+5% satış
    - Grid (4 levels) - %40/%30/%20/%10 alım, +1%/+2%/+3.5%/+5% satış
    - Normal Trading (tek al/sat)
    
Arguments:
    -c, --coin          Coin filtresi (BTC, ETH, SOL, vb.)
    -t, --timeframe     Timeframe filtresi (15m veya 1h)
    -m, --months        Backtest süresi (ay, varsayılan: 3)
    --charts            Her model için ayrı grafik üret
    --compare           5 strateji karşılaştırması (tüm modeller veya filtreli)
"""
import os
import sys
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import from existing modules
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# ============================================
# CONFIGURATION
# ============================================
KAGGLE_OUTPUTS_DIR = PROJECT_ROOT / 'kaggle_outputs'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Window sizes
CNN_WINDOW = 12
LSTM_WINDOW = 120
TR_WINDOW = 120

# ============================================
# DATA CLASSES
# ============================================
@dataclass
class TradeResult:
    """Single trade/prediction result."""
    timestamp: datetime
    prediction: float  # Predicted log return
    actual: float  # Actual log return
    price: float
    direction_correct: bool
    
@dataclass
class BacktestResults:
    """Complete backtest results."""
    coin: str
    timeframe: str
    months: int
    total_candles: int
    total_predictions: int
    correct_direction: int
    accuracy: float
    avg_prediction: float
    avg_actual: float
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    trades: List[TradeResult] = field(default_factory=list)

@dataclass
class StrategyResult:
    """Results for a single strategy simulation."""
    strategy_name: str
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

# ============================================
# MODEL DISCOVERY
# ============================================
def discover_models() -> List[Dict]:
    """
    Automatically discover all models in kaggle_outputs directory.
    Returns list of dicts with coin, timeframe, and paths.
    """
    models = []
    
    # Find all model files
    model_files = list(KAGGLE_OUTPUTS_DIR.glob("*_model.pth"))
    
    for model_path in model_files:
        # Parse filename: {COIN}_{TF}_model.pth
        filename = model_path.stem  # e.g., "BTC_15m_model"
        parts = filename.replace("_model", "").rsplit("_", 1)  # Split from right
        
        if len(parts) == 2:
            coin, timeframe = parts
            
            # Check for required files
            params_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_params.json"
            stats_path = KAGGLE_OUTPUTS_DIR / f"{coin}_{timeframe}_stats.json"
            
            if params_path.exists() and stats_path.exists():
                models.append({
                    'coin': coin,
                    'timeframe': timeframe,
                    'model_path': model_path,
                    'params_path': params_path,
                    'stats_path': stats_path
                })
    
    # Sort by coin, then timeframe
    models.sort(key=lambda x: (x['coin'], x['timeframe']))
    return models

# ============================================
# BACKTESTER CLASS
# ============================================
class Backtester:
    """Backtest AI models on historical data using kaggle_outputs models."""
    
    def __init__(self, coin: str, timeframe: str, months_back: int, model_info: Optional[Dict] = None):
        self.coin = coin.upper()
        self.timeframe = timeframe
        self.months_back = months_back
        self.symbol = f"{self.coin}/USDT"
        self.model: Optional[MultiBranchModel] = None
        self.model_info = model_info
        self.stats = None
        
    def load_model(self) -> Optional[MultiBranchModel]:
        """Load the model from kaggle_outputs."""
        # Find model info if not provided
        if self.model_info is None:
            model_path = KAGGLE_OUTPUTS_DIR / f"{self.coin}_{self.timeframe}_model.pth"
            params_path = KAGGLE_OUTPUTS_DIR / f"{self.coin}_{self.timeframe}_params.json"
            stats_path = KAGGLE_OUTPUTS_DIR / f"{self.coin}_{self.timeframe}_stats.json"
            
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                return None
                
            self.model_info = {
                'model_path': model_path,
                'params_path': params_path,
                'stats_path': stats_path
            }
        
        try:
            # Load parameters
            with open(self.model_info['params_path']) as f:
                params = json.load(f)
            embed_dim = params.get('embed_dim', 96)
            dropout = params.get('dropout', 0.15)
            
            # Load normalization stats
            with open(self.model_info['stats_path']) as f:
                self.stats = json.load(f)
            
            print(f"🧠 Loading model: {self.coin}_{self.timeframe}")
            print(f"   Parameters: embed_dim={embed_dim}, dropout={dropout:.2f}")
            
            # Load model
            model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(DEVICE)
            state_dict = torch.load(self.model_info['model_path'], map_location=DEVICE, weights_only=True)
            
            # Handle DataParallel prefix if present
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict)
            model.eval()
            
            self.model = model
            print(f"✅ Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical data using existing data_fetcher function."""
        print(f"📊 Fetching {self.months_back} months of {self.timeframe} data for {self.symbol}...")
        
        try:
            df = get_crypto_history(
                symbol=self.symbol,
                timeframe=self.timeframe,
                months_back=self.months_back,
                exchange_name="binance"
            )
            print(f"✅ Fetched {len(df)} candles")
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Prepare data using existing prepare_dual_dataframes function."""
        try:
            df_display, df_ai = prepare_dual_dataframes(df)
            print(f"✅ Prepared data: {len(df_ai)} samples ready for AI")
            return df_display, df_ai
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return None, None
    
    def prepare_model_input(self, df_ai: pd.DataFrame, end_idx: int):
        """Prepare the input tensors using model-specific normalization stats."""
        # Get column indices for each branch
        cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
        lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
        tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
        
        # USE MODEL-SPECIFIC STATS (from training) instead of computing from data
        df_subset = df_ai.iloc[:end_idx]
        
        if self.stats:
            mean = pd.Series(self.stats['mean'])
            std = pd.Series(self.stats['std'])
        else:
            # Fallback to computing from data
            mean = df_subset.mean()
            std = df_subset.std()
            if 'Log_Ret' in mean:
                mean['Log_Ret'] = 0.0
        
        std[std == 0] = 1.0
        df_normalized = (df_subset - mean) / std
        
        data = df_normalized.values
        max_window = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)
        
        if len(data) < max_window:
            return None, None, None
        
        # Get the last window of data
        t = len(data)
        x_cnn = data[t - CNN_WINDOW:t, cnn_cols]
        x_lstm = data[t - LSTM_WINDOW:t, lstm_cols]
        x_tr = data[t - TR_WINDOW:t, tr_cols]
        
        # Convert to tensors and add batch dimension
        x_cnn = torch.tensor(x_cnn, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        x_lstm = torch.tensor(x_lstm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        x_tr = torch.tensor(x_tr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        return x_cnn, x_lstm, x_tr
    
    def run_backtest(self, verbose: bool = True) -> Optional[BacktestResults]:
        """Run the complete backtest."""
        # Load model
        if not self.load_model():
            return None
        
        # Fetch data
        df_raw = self.fetch_data()
        if df_raw is None or len(df_raw) < LSTM_WINDOW + 100:
            print(f"❌ Insufficient data. Need at least {LSTM_WINDOW + 100} candles.")
            return None
        
        # Prepare data
        df_display, df_ai = self.prepare_data(df_raw)
        if df_ai is None:
            return None
        
        if verbose:
            print(f"🚀 Starting backtest from index {LSTM_WINDOW} to {len(df_ai) - 1}...")
        
        trades: List[TradeResult] = []
        correct_direction = 0
        max_window = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)
        
        # Sliding window backtest
        total_iterations = len(df_ai) - max_window - 1
        for i in range(max_window, len(df_ai) - 1):
            # Progress indicator
            if verbose and (i - max_window) % 500 == 0:
                progress = (i - max_window + 1) / total_iterations * 100
                print(f"   Progress: {progress:.1f}% ({i - max_window + 1}/{total_iterations})")
            
            # Prepare input (using data up to index i)
            x_cnn, x_lstm, x_tr = self.prepare_model_input(df_ai, i + 1)
            if x_cnn is None:
                continue
            
            # Get prediction
            with torch.no_grad():
                pred_main, _, _, _ = self.model(x_cnn, x_lstm, x_tr)
                prediction = pred_main.item() / 100.0  # Reverse training scaling
            
            # Get actual next candle return
            actual_return = df_ai.iloc[i + 1]['Log_Ret']
            
            # Check direction
            pred_direction = 1 if prediction > 0 else -1
            actual_direction = 1 if actual_return > 0 else -1
            is_correct = pred_direction == actual_direction
            if is_correct:
                correct_direction += 1
            
            # Record trade
            trade = TradeResult(
                timestamp=df_ai.index[i],
                prediction=prediction,
                actual=actual_return,
                price=df_display.iloc[i]['Close'],
                direction_correct=is_correct
            )
            trades.append(trade)
        
        if len(trades) == 0:
            print("❌ No trades executed during backtest.")
            return None
        
        # Calculate metrics
        accuracy = correct_direction / len(trades) * 100
        avg_prediction = np.mean([t.prediction for t in trades])
        avg_actual = np.mean([t.actual for t in trades])
        
        # Calculate cumulative return (if we followed all predictions)
        strategy_returns = [t.actual if t.prediction > 0 else -t.actual for t in trades]
        cumulative_return = sum(strategy_returns)
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate Sharpe Ratio (annualized)
        if len(strategy_returns) > 1:
            returns_std = np.std(strategy_returns)
            if returns_std > 0:
                # Annualization factor depends on timeframe
                if self.timeframe == '15m':
                    periods_per_year = 365 * 24 * 4  # 4 per hour * 24 hours * 365 days
                elif self.timeframe == '1h':
                    periods_per_year = 365 * 24
                else:
                    periods_per_year = 365
                sharpe_ratio = (np.mean(strategy_returns) / returns_std) * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate Profit Factor
        gains = sum([r for r in strategy_returns if r > 0])
        losses = abs(sum([r for r in strategy_returns if r < 0]))
        profit_factor = gains / losses if losses > 0 else float('inf') if gains > 0 else 0.0
        
        results = BacktestResults(
            coin=self.coin,
            timeframe=self.timeframe,
            months=self.months_back,
            total_candles=len(df_ai),
            total_predictions=len(trades),
            correct_direction=correct_direction,
            accuracy=accuracy,
            avg_prediction=avg_prediction,
            avg_actual=avg_actual,
            cumulative_return=cumulative_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            trades=trades
        )
        
        if verbose:
            print(f"✅ Backtest complete! Processed {len(trades)} predictions.")
        return results
    
    def simulate_hodl(self, df_display: pd.DataFrame, initial_balance: float = 1000.0) -> StrategyResult:
        """Simulate buy and hold strategy."""
        prices = df_display['Close'].values
        timestamps = df_display.index.tolist()
        
        # Buy at start, hold until end
        start_price = prices[0]
        amount = initial_balance / start_price
        
        # Track equity curve
        equity_curve = [amount * p for p in prices]
        
        # Calculate metrics
        end_value = equity_curve[-1]
        total_return = ((end_value - initial_balance) / initial_balance) * 100
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max * 100
        max_drawdown = np.max(drawdowns)
        
        # Sharpe (daily returns approximation)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24 * 4) if np.std(returns) > 0 else 0
        
        return StrategyResult(
            strategy_name="HODL",
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown,
            total_trades=1,  # Just 1 buy
            win_rate=100.0 if total_return > 0 else 0.0,
            sharpe_ratio=sharpe,
            equity_curve=equity_curve,
            timestamps=timestamps
        )
    
    def simulate_grid_trading(self, df_display: pd.DataFrame, df_ai: pd.DataFrame, 
                              initial_balance: float = 1000.0, grid_levels: int = 2,
                              threshold: float = 0.001) -> StrategyResult:
        """Simulate grid trading strategy with DCA and partial sells."""
        # Grid allocations and targets
        if grid_levels == 2:
            buy_allocs = [0.6, 0.4]
            sell_targets = [0.02, 0.04]
        elif grid_levels == 3:
            buy_allocs = [0.5, 0.3, 0.2]
            sell_targets = [0.015, 0.03, 0.05]
        else:
            buy_allocs = [0.4, 0.3, 0.2, 0.1]
            sell_targets = [0.01, 0.02, 0.035, 0.05]
        
        stop_loss = -0.05
        
        prices = df_display['Close'].values
        timestamps = df_display.index.tolist()
        max_window = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)
        
        # State tracking
        balance = initial_balance
        position_amount = 0.0
        grid_entries = []  # List of (price, amount)
        buy_grid_level = 0
        sell_grid_level = 0
        total_trades = 0
        wins = 0
        equity_curve = []
        
        for i in range(len(prices)):
            current_price = prices[i]
            
            # Calculate equity
            equity = balance + (position_amount * current_price)
            equity_curve.append(equity)
            
            # Skip warmup period
            if i < max_window + 1 or i >= len(df_ai) - 1:
                continue
            
            # Get prediction
            x_cnn, x_lstm, x_tr = self.prepare_model_input(df_ai, i + 1)
            if x_cnn is None:
                continue
            
            with torch.no_grad():
                pred_main, _, _, _ = self.model(x_cnn, x_lstm, x_tr)
                prediction = pred_main.item() / 100.0
            
            # === GRID LOGIC ===
            if position_amount > 0:
                # Calculate average entry
                total_cost = sum(e[0] * e[1] for e in grid_entries)
                total_amount = sum(e[1] for e in grid_entries)
                avg_entry = total_cost / total_amount if total_amount > 0 else current_price
                pnl_pct = (current_price - avg_entry) / avg_entry
                
                # Stop-loss: sell all
                if pnl_pct <= stop_loss:
                    balance += position_amount * current_price
                    if pnl_pct > 0:
                        wins += 1
                    total_trades += 1
                    position_amount = 0.0
                    grid_entries = []
                    buy_grid_level = 0
                    sell_grid_level = 0
                    continue
                
                # Check grid sell targets
                if sell_grid_level < len(sell_targets):
                    target = sell_targets[sell_grid_level]
                    if pnl_pct >= target:
                        # Partial sell
                        sell_pct = buy_allocs[sell_grid_level]
                        sell_amount = position_amount * sell_pct
                        balance += sell_amount * current_price
                        position_amount -= sell_amount
                        sell_grid_level += 1
                        total_trades += 1
                        wins += 1  # Taking profit = win
                        
                        if position_amount < 0.0001:
                            grid_entries = []
                            buy_grid_level = 0
                            sell_grid_level = 0
                        continue
                
                # Check for additional grid buy
                if buy_grid_level < len(buy_allocs) and prediction > threshold:
                    price_drop = (avg_entry - current_price) / avg_entry
                    if price_drop >= 0.01:  # 1% drop
                        alloc = buy_allocs[buy_grid_level]
                        buy_usdt = initial_balance * alloc * 0.5  # Use half of allocation for adds
                        if balance >= buy_usdt:
                            amount = buy_usdt / current_price
                            grid_entries.append((current_price, amount))
                            position_amount += amount
                            balance -= buy_usdt
                            buy_grid_level += 1
                            total_trades += 1
            
            else:
                # No position - check for entry
                if prediction > threshold and balance > 10:
                    alloc = buy_allocs[0]
                    buy_usdt = min(balance, initial_balance * alloc)
                    amount = buy_usdt / current_price
                    grid_entries.append((current_price, amount))
                    position_amount = amount
                    balance -= buy_usdt
                    buy_grid_level = 1
                    sell_grid_level = 0
                    total_trades += 1
        
        # Final equity
        final_equity = balance + (position_amount * prices[-1])
        total_return = ((final_equity - initial_balance) / initial_balance) * 100
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - np.array(equity_curve)) / running_max * 100
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sharpe
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24 * 4) if np.std(returns) > 0 else 0
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        return StrategyResult(
            strategy_name=f"Grid ({grid_levels} levels)",
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            equity_curve=equity_curve,
            timestamps=timestamps
        )
    
    def simulate_normal_trading(self, df_display: pd.DataFrame, df_ai: pd.DataFrame,
                                 initial_balance: float = 1000.0, threshold: float = 0.001) -> StrategyResult:
        """Simulate normal single buy/sell trading based on AI predictions."""
        prices = df_display['Close'].values
        timestamps = df_display.index.tolist()
        max_window = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)
        
        stop_loss = -0.05
        take_profit = 0.03
        
        balance = initial_balance
        position_amount = 0.0
        entry_price = 0.0
        total_trades = 0
        wins = 0
        equity_curve = []
        
        for i in range(len(prices)):
            current_price = prices[i]
            equity = balance + (position_amount * current_price)
            equity_curve.append(equity)
            
            if i < max_window + 1 or i >= len(df_ai) - 1:
                continue
            
            x_cnn, x_lstm, x_tr = self.prepare_model_input(df_ai, i + 1)
            if x_cnn is None:
                continue
            
            with torch.no_grad():
                pred_main, _, _, _ = self.model(x_cnn, x_lstm, x_tr)
                prediction = pred_main.item() / 100.0
            
            if position_amount > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Stop-loss or take profit
                should_sell = False
                if pnl_pct <= stop_loss:
                    should_sell = True
                elif pnl_pct >= take_profit:
                    should_sell = True
                    wins += 1
                elif prediction < -threshold and pnl_pct > 0:
                    should_sell = True
                    wins += 1
                
                if should_sell:
                    balance += position_amount * current_price
                    position_amount = 0.0
                    total_trades += 1
            else:
                # Check for buy signal
                if prediction > threshold and balance > 10:
                    buy_usdt = balance * 0.95
                    position_amount = buy_usdt / current_price
                    entry_price = current_price
                    balance -= buy_usdt
                    total_trades += 1
        
        final_equity = balance + (position_amount * prices[-1])
        total_return = ((final_equity - initial_balance) / initial_balance) * 100
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - np.array(equity_curve)) / running_max * 100
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24 * 4) if np.std(returns) > 0 else 0
        
        win_rate = (wins / (total_trades // 2) * 100) if total_trades > 1 else 0
        
        return StrategyResult(
            strategy_name="Normal Trading",
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            equity_curve=equity_curve,
            timestamps=timestamps
        )
    
    def run_strategy_comparison(self, grid_levels: int = 2) -> Optional[List[StrategyResult]]:
        """Run comparison of HODL, Grid, and Normal trading strategies."""
        print(f"\n📊 Running Strategy Comparison for {self.coin}/{self.timeframe}...")
        
        if not self.load_model():
            return None
        
        df_raw = self.fetch_data()
        if df_raw is None or len(df_raw) < LSTM_WINDOW + 100:
            print(f"❌ Insufficient data.")
            return None
        
        df_display, df_ai = self.prepare_data(df_raw)
        if df_ai is None:
            return None
        
        print("🔄 Simulating strategies...")
        
        # Run all strategies: HODL + Grid (2,3,4 levels) + Normal
        hodl_result = self.simulate_hodl(df_display)
        grid2_result = self.simulate_grid_trading(df_display, df_ai, grid_levels=2)
        grid3_result = self.simulate_grid_trading(df_display, df_ai, grid_levels=3)
        grid4_result = self.simulate_grid_trading(df_display, df_ai, grid_levels=4)
        normal_result = self.simulate_normal_trading(df_display, df_ai)
        
        results = [hodl_result, grid2_result, grid3_result, grid4_result, normal_result]
        
        # Print comparison
        print("\n" + "=" * 80)
        print(f"📊 STRATEGY COMPARISON: {self.coin}/{self.timeframe} ({self.months_back} months)")
        print("=" * 80)
        print(f"\n{'Strategy':<20} {'Return':<12} {'Max DD':<12} {'Trades':<10} {'Win Rate':<12} {'Sharpe':<10}")
        print("-" * 80)
        
        for r in results:
            ret_color = "🟢" if r.total_return_pct > 0 else "🔴"
            print(f"{r.strategy_name:<20} {ret_color}{r.total_return_pct:>+8.2f}%   "
                  f"{r.max_drawdown_pct:>8.2f}%   {r.total_trades:>6}     "
                  f"{r.win_rate:>8.1f}%   {r.sharpe_ratio:>8.2f}")
        
        print("-" * 80)
        
        # Best strategy
        best = max(results, key=lambda x: x.total_return_pct)
        print(f"\n🏆 Best Strategy: {best.strategy_name} ({best.total_return_pct:+.2f}%)")
        
        # Best grid
        grid_results = [r for r in results if 'Grid' in r.strategy_name]
        best_grid = max(grid_results, key=lambda x: x.total_return_pct)
        print(f"📊 Best Grid: {best_grid.strategy_name} ({best_grid.total_return_pct:+.2f}%)")
        
        # Generate chart
        self.generate_comparison_chart(results)
        
        return results
    
    def generate_comparison_chart(self, results: List[StrategyResult]) -> None:
        """Generate comparison chart for all strategies."""
        output_dir = PROJECT_ROOT / "backtest_results"
        output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Strategy Comparison: {self.coin}/{self.timeframe} ({self.months_back} months)', 
                     fontsize=14, fontweight='bold')
        
        colors = {'HODL': '#3498db', 'Grid': '#2ecc71', 'Normal Trading': '#e74c3c'}
        
        # 1. Equity Curves
        ax1 = axes[0, 0]
        for r in results:
            color = colors.get(r.strategy_name.split()[0], '#9b59b6')
            # Normalize to percentage
            initial = r.equity_curve[0] if r.equity_curve else 1000
            normalized = [(e / initial - 1) * 100 for e in r.equity_curve]
            ax1.plot(r.timestamps, normalized, label=r.strategy_name, linewidth=1.5, color=color)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curves (% Return)', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Return Comparison Bar
        ax2 = axes[0, 1]
        names = [r.strategy_name for r in results]
        returns = [r.total_return_pct for r in results]
        bar_colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
        ax2.bar(names, returns, color=bar_colors, edgecolor='white')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Total Return Comparison', fontweight='bold')
        ax2.set_ylabel('Return (%)')
        for i, (n, r) in enumerate(zip(names, returns)):
            ax2.text(i, r + (1 if r >= 0 else -3), f'{r:.1f}%', ha='center', fontweight='bold')
        
        # 3. Max Drawdown Comparison
        ax3 = axes[1, 0]
        drawdowns = [r.max_drawdown_pct for r in results]
        ax3.bar(names, drawdowns, color='#e74c3c', edgecolor='white', alpha=0.7)
        ax3.set_title('Max Drawdown Comparison', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        for i, (n, d) in enumerate(zip(names, drawdowns)):
            ax3.text(i, d + 0.5, f'{d:.1f}%', ha='center')
        
        # 4. Summary Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        table_data = []
        for r in results:
            table_data.append([
                r.strategy_name,
                f'{r.total_return_pct:+.2f}%',
                f'{r.max_drawdown_pct:.2f}%',
                str(r.total_trades),
                f'{r.win_rate:.1f}%',
                f'{r.sharpe_ratio:.2f}'
            ])
        
        table = ax4.table(
            cellText=table_data,
            colLabels=['Strategy', 'Return', 'Max DD', 'Trades', 'Win Rate', 'Sharpe'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_path = output_dir / f"strategy_compare_{self.coin}_{self.timeframe}_{timestamp}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n📈 Comparison chart saved to: {chart_path}")

    def generate_report(self, results: BacktestResults) -> None:
        """Generate and display backtest report."""
        print("\n" + "=" * 60)
        print(f"📊 BACKTEST REPORT: {results.coin} / {results.timeframe}")
        print("=" * 60)
        
        print(f"\n📅 Period: {results.months} months")
        print(f"📈 Total Candles: {results.total_candles}")
        print(f"🎯 Total Predictions: {results.total_predictions}")
        
        print(f"\n{'─' * 40}")
        print("📊 PERFORMANCE METRICS")
        print(f"{'─' * 40}")
        
        print(f"   ✅ Correct Direction: {results.correct_direction} / {results.total_predictions}")
        print(f"   🎯 Accuracy: {results.accuracy:.2f}%")
        print(f"   📈 Avg Prediction: {results.avg_prediction * 100:.4f}%")
        print(f"   📊 Avg Actual Return: {results.avg_actual * 100:.4f}%")
        print(f"   💰 Cumulative Return: {results.cumulative_return * 100:.2f}%")
        print(f"   📉 Max Drawdown: {results.max_drawdown * 100:.2f}%")
        print(f"   📐 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   ⚖️ Profit Factor: {results.profit_factor:.2f}")
        
        # Save to CSV and get output path
        output_dir, file_prefix = self.save_results_to_csv(results)
        
        # Generate charts
        self.generate_charts(results, output_dir, file_prefix)
    
    def save_results_to_csv(self, results: BacktestResults) -> Tuple[Path, str]:
        """Save detailed results to CSV file."""
        # Create output directory
        output_dir = PROJECT_ROOT / "backtest_results"
        output_dir.mkdir(exist_ok=True)
        
        # Prepare DataFrame
        data = []
        for trade in results.trades:
            data.append({
                'timestamp': trade.timestamp,
                'price': trade.price,
                'prediction': trade.prediction,
                'actual': trade.actual,
                'direction_correct': trade.direction_correct,
                'prediction_pct': trade.prediction * 100,
                'actual_pct': trade.actual * 100
            })
        
        df = pd.DataFrame(data)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{results.coin}_{results.timeframe}_{results.months}m_{timestamp}.csv"
        filepath = output_dir / filename
        
        df.to_csv(filepath, index=False)
        print(f"\n💾 Results saved to: {filepath}")
        
        # Also save summary
        summary_filename = f"backtest_{results.coin}_{results.timeframe}_{results.months}m_{timestamp}_summary.txt"
        summary_filepath = output_dir / summary_filename
        
        with open(summary_filepath, 'w') as f:
            f.write(f"BACKTEST SUMMARY: {results.coin} / {results.timeframe}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Period: {results.months} months\n")
            f.write(f"Total Candles: {results.total_candles}\n")
            f.write(f"Total Predictions: {results.total_predictions}\n\n")
            f.write(f"Correct Direction: {results.correct_direction}\n")
            f.write(f"Accuracy: {results.accuracy:.2f}%\n")
            f.write(f"Avg Prediction: {results.avg_prediction * 100:.4f}%\n")
            f.write(f"Avg Actual Return: {results.avg_actual * 100:.4f}%\n")
            f.write(f"Cumulative Return: {results.cumulative_return * 100:.2f}%\n")
            f.write(f"Max Drawdown: {results.max_drawdown * 100:.2f}%\n")
            f.write(f"Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
            f.write(f"Profit Factor: {results.profit_factor:.2f}\n")
        
        print(f"📝 Summary saved to: {summary_filepath}")
        
        # Return filepath prefix for charts
        return output_dir, f"backtest_{results.coin}_{results.timeframe}_{results.months}m_{timestamp}"
    
    def generate_charts(self, results: BacktestResults, output_dir: Path, file_prefix: str) -> None:
        """Generate and save backtest result charts."""
        print("📊 Generating charts...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Backtest Results: {results.coin} / {results.timeframe} ({results.months} months)', 
                     fontsize=14, fontweight='bold')
        
        # Prepare data
        timestamps = [t.timestamp for t in results.trades]
        predictions = [t.prediction * 100 for t in results.trades]  # Convert to %
        actuals = [t.actual * 100 for t in results.trades]  # Convert to %
        cumulative_returns = np.cumsum([t.actual if t.prediction > 0 else -t.actual for t in results.trades]) * 100
        
        # 1. Equity Curve (Cumulative Returns)
        ax1 = axes[0, 0]
        ax1.plot(timestamps, cumulative_returns, color='#2ecc71', linewidth=1.5, label='Strategy')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(timestamps, 0, cumulative_returns, 
                         where=[r >= 0 for r in cumulative_returns], 
                         color='#2ecc71', alpha=0.3)
        ax1.fill_between(timestamps, 0, cumulative_returns, 
                         where=[r < 0 for r in cumulative_returns], 
                         color='#e74c3c', alpha=0.3)
        ax1.set_title('Equity Curve (Cumulative Return)', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Prediction vs Actual Scatter
        ax2 = axes[0, 1]
        colors = ['#2ecc71' if t.direction_correct else '#e74c3c' for t in results.trades]
        ax2.scatter(predictions, actuals, c=colors, alpha=0.5, s=10)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        max_val = max(max(abs(p) for p in predictions), max(abs(a) for a in actuals))
        ax2.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.3, label='Perfect')
        ax2.set_title('Prediction vs Actual Return', fontweight='bold')
        ax2.set_xlabel('Predicted Return (%)')
        ax2.set_ylabel('Actual Return (%)')
        ax2.legend(['Perfect Prediction', f'Trades (Acc: {results.accuracy:.1f}%)'])
        
        # 3. Rolling Accuracy (50-period window)
        ax3 = axes[1, 0]
        correct_list = [1 if t.direction_correct else 0 for t in results.trades]
        window_size = min(50, len(correct_list) // 4)
        if window_size > 0:
            rolling_acc = pd.Series(correct_list).rolling(window=window_size).mean() * 100
            ax3.plot(timestamps, rolling_acc, color='#3498db', linewidth=1.5)
            ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
            ax3.axhline(y=results.accuracy, color='green', linestyle='--', alpha=0.7, label=f'Avg ({results.accuracy:.1f}%)')
        ax3.set_title(f'Rolling Accuracy ({window_size}-period)', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Return Distribution
        ax4 = axes[1, 1]
        pred_clip = np.percentile(predictions, [1, 99])
        actual_clip = np.percentile(actuals, [1, 99])
        x_min = min(pred_clip[0], actual_clip[0]) - 0.5
        x_max = max(pred_clip[1], actual_clip[1]) + 0.5
        bins = np.linspace(x_min, x_max, 50)
        
        ax4.hist(actuals, bins=bins, alpha=0.7, color='#e74c3c', label='Actual Returns', edgecolor='white')
        ax4_twin = ax4.twinx()
        ax4_twin.hist(predictions, bins=bins, alpha=0.5, color='#3498db', label='Predictions', edgecolor='white')
        
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
        ax4.set_title('Return Distribution', fontweight='bold')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Actual Count', color='#e74c3c')
        ax4_twin.set_ylabel('Prediction Count', color='#3498db')
        ax4.tick_params(axis='y', labelcolor='#e74c3c')
        ax4_twin.tick_params(axis='y', labelcolor='#3498db')
        
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / f"{file_prefix}_charts.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📈 Charts saved to: {chart_path}")


# ============================================
# BATCH BACKTESTER - RUN ALL MODELS
# ============================================
class BatchBacktester:
    """Run backtest on all models and save consolidated results."""
    
    def __init__(self, months_back: int = 3, coin_filter: str = None, timeframe_filter: str = None):
        self.months_back = months_back
        self.coin_filter = coin_filter.upper() if coin_filter else None
        self.timeframe_filter = timeframe_filter if timeframe_filter else None
        self.results: List[BacktestResults] = []
        
    def run_all(self, generate_individual_charts: bool = False) -> List[BacktestResults]:
        """Run backtest on all discovered models."""
        # Discover models
        all_models = discover_models()
        
        # Apply filters
        if self.coin_filter:
            all_models = [m for m in all_models if m['coin'] == self.coin_filter]
        if self.timeframe_filter:
            all_models = [m for m in all_models if m['timeframe'] == self.timeframe_filter]
        
        if not all_models:
            print("❌ No models found matching the criteria.")
            return []
        
        print("\n" + "=" * 60)
        print("🔬 BATCH BACKTEST - ALL MODELS")
        print("=" * 60)
        print(f"   Models found: {len(all_models)}")
        print(f"   Backtest period: {self.months_back} months")
        print(f"   Device: {DEVICE}")
        if self.coin_filter:
            print(f"   Coin filter: {self.coin_filter}")
        if self.timeframe_filter:
            print(f"   Timeframe filter: {self.timeframe_filter}")
        print("=" * 60 + "\n")
        
        # Run backtests
        for idx, model_info in enumerate(all_models, 1):
            coin = model_info['coin']
            timeframe = model_info['timeframe']
            
            print(f"\n{'─' * 60}")
            print(f"📊 [{idx}/{len(all_models)}] Testing: {coin} / {timeframe}")
            print(f"{'─' * 60}")
            
            try:
                backtester = Backtester(
                    coin=coin,
                    timeframe=timeframe,
                    months_back=self.months_back,
                    model_info=model_info
                )
                
                results = backtester.run_backtest(verbose=False)
                
                if results:
                    self.results.append(results)
                    print(f"   ✅ Accuracy: {results.accuracy:.2f}%")
                    print(f"   💰 Cumulative Return: {results.cumulative_return * 100:.2f}%")
                    print(f"   📐 Sharpe Ratio: {results.sharpe_ratio:.2f}")
                    
                    # Generate individual charts if requested
                    if generate_individual_charts:
                        output_dir, file_prefix = backtester.save_results_to_csv(results)
                        backtester.generate_charts(results, output_dir, file_prefix)
                else:
                    print(f"   ❌ Backtest failed")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        # Generate consolidated report
        self.generate_consolidated_report()
        
        return self.results
    
    def generate_consolidated_report(self):
        """Generate and save consolidated report for all backtests."""
        if not self.results:
            print("\n❌ No successful backtests to report.")
            return
        
        # Create output directory
        output_dir = PROJECT_ROOT / "backtest_results"
        output_dir.mkdir(exist_ok=True)
        
        # Prepare summary data
        summary_data = []
        for r in self.results:
            summary_data.append({
                'coin': r.coin,
                'timeframe': r.timeframe,
                'months': r.months,
                'total_predictions': r.total_predictions,
                'correct_direction': r.correct_direction,
                'accuracy': round(r.accuracy, 2),
                'avg_prediction_pct': round(r.avg_prediction * 100, 4),
                'avg_actual_pct': round(r.avg_actual * 100, 4),
                'cumulative_return_pct': round(r.cumulative_return * 100, 2),
                'max_drawdown_pct': round(r.max_drawdown * 100, 2),
                'sharpe_ratio': round(r.sharpe_ratio, 2),
                'profit_factor': round(r.profit_factor, 2)
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Sort by accuracy descending
        df_summary = df_summary.sort_values('accuracy', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = output_dir / f"backtest_ALL_MODELS_{self.months_back}m_{timestamp}.csv"
        df_summary.to_csv(summary_file, index=False)
        
        # Print consolidated report
        print("\n" + "=" * 80)
        print("📊 CONSOLIDATED BACKTEST REPORT")
        print("=" * 80)
        print(f"\n📅 Period: {self.months_back} months")
        print(f"🔢 Models Tested: {len(self.results)}")
        
        # Statistics
        avg_accuracy = df_summary['accuracy'].mean()
        best_accuracy = df_summary['accuracy'].max()
        worst_accuracy = df_summary['accuracy'].min()
        above_50 = (df_summary['accuracy'] > 50).sum()
        
        avg_return = df_summary['cumulative_return_pct'].mean()
        best_return = df_summary['cumulative_return_pct'].max()
        worst_return = df_summary['cumulative_return_pct'].min()
        positive_returns = (df_summary['cumulative_return_pct'] > 0).sum()
        
        print(f"\n{'─' * 40}")
        print("📈 ACCURACY STATISTICS")
        print(f"{'─' * 40}")
        print(f"   Average: {avg_accuracy:.2f}%")
        print(f"   Best: {best_accuracy:.2f}%")
        print(f"   Worst: {worst_accuracy:.2f}%")
        print(f"   Above 50%: {above_50}/{len(self.results)} models")
        
        print(f"\n{'─' * 40}")
        print("💰 RETURN STATISTICS")
        print(f"{'─' * 40}")
        print(f"   Average: {avg_return:.2f}%")
        print(f"   Best: {best_return:.2f}%")
        print(f"   Worst: {worst_return:.2f}%")
        print(f"   Profitable: {positive_returns}/{len(self.results)} models")
        
        # Top 5 and Bottom 5
        print(f"\n{'─' * 40}")
        print("🏆 TOP 5 MODELS (by Accuracy)")
        print(f"{'─' * 40}")
        for _, row in df_summary.head(5).iterrows():
            print(f"   {row['coin']}_{row['timeframe']}: {row['accuracy']:.2f}% (Return: {row['cumulative_return_pct']:+.2f}%)")
        
        print(f"\n{'─' * 40}")
        print("📉 BOTTOM 5 MODELS (by Accuracy)")
        print(f"{'─' * 40}")
        for _, row in df_summary.tail(5).iterrows():
            print(f"   {row['coin']}_{row['timeframe']}: {row['accuracy']:.2f}% (Return: {row['cumulative_return_pct']:+.2f}%)")
        
        print(f"\n💾 Summary saved to: {summary_file}")
        
        # Generate summary chart
        self.generate_summary_chart(df_summary, output_dir, timestamp)
    
    def generate_summary_chart(self, df: pd.DataFrame, output_dir: Path, timestamp: str):
        """Generate summary chart for all models."""
        print("📊 Generating summary chart...")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'All Models Backtest Summary ({self.months_back} months)', 
                     fontsize=16, fontweight='bold')
        
        # Add model label column
        df['model'] = df['coin'] + '_' + df['timeframe']
        
        # 1. Accuracy Bar Chart
        ax1 = axes[0, 0]
        colors = ['#2ecc71' if acc > 50 else '#e74c3c' for acc in df['accuracy']]
        bars = ax1.barh(df['model'], df['accuracy'], color=colors, edgecolor='white')
        ax1.axvline(x=50, color='black', linestyle='--', linewidth=2, label='Random (50%)')
        ax1.set_xlabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_xlim(0, 100)
        
        # 2. Return Bar Chart
        ax2 = axes[0, 1]
        colors = ['#2ecc71' if ret > 0 else '#e74c3c' for ret in df['cumulative_return_pct']]
        ax2.barh(df['model'], df['cumulative_return_pct'], color=colors, edgecolor='white')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax2.set_xlabel('Cumulative Return (%)')
        ax2.set_title('Model Returns Comparison', fontweight='bold')
        
        # 3. Accuracy vs Return Scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['accuracy'], df['cumulative_return_pct'], 
                              c=df['sharpe_ratio'], cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Accuracy (%)')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_title('Accuracy vs Return (color = Sharpe)', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Sharpe Ratio')
        
        # Add labels for ALL models
        for idx, row in df.iterrows():
            ax3.annotate(row['model'], 
                        (row['accuracy'], row['cumulative_return_pct']),
                        fontsize=7, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points',
                        alpha=0.8)
        
        # 4. Accuracy Distribution with model names
        ax4 = axes[1, 1]
        # Create bar chart instead of histogram to show individual model names
        df_sorted = df.sort_values('accuracy', ascending=True)
        colors = ['#2ecc71' if acc > 50 else '#e74c3c' for acc in df_sorted['accuracy']]
        ax4.barh(df_sorted['model'], df_sorted['accuracy'], color=colors, edgecolor='white', alpha=0.8)
        ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        ax4.axvline(x=df['accuracy'].mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean ({df["accuracy"].mean():.1f}%)')
        ax4.set_xlabel('Accuracy (%)')
        ax4.set_title('Model Accuracy Ranking', fontweight='bold')
        ax4.legend(loc='lower right')
        ax4.set_xlim(0, 100)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = output_dir / f"backtest_ALL_MODELS_{self.months_back}m_{timestamp}_summary.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📈 Summary chart saved to: {chart_path}")


# ============================================
# CLI INTERFACE
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Backtest AI models on historical cryptocurrency data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py                          # Run ALL 40 models
  python backtest.py --coin BTC               # Run only BTC models (both timeframes)
  python backtest.py --timeframe 1h           # Run only 1h models (all coins)
  python backtest.py --coin BTC --timeframe 1h  # Run single model
  python backtest.py --months 6               # Set backtest period to 6 months
  python backtest.py --charts                 # Generate individual charts for each model
        """
    )
    
    parser.add_argument(
        '-c', '--coin',
        type=str,
        default=None,
        help='Coin symbol filter (e.g., BTC, ETH, SOL). If not specified, runs all coins.'
    )
    
    parser.add_argument(
        '-t', '--timeframe',
        type=str,
        default=None,
        choices=['15m', '1h'],
        help='Timeframe filter (15m or 1h). If not specified, runs both.'
    )
    
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=3,
        help='Months of historical data to test (default: 3)'
    )
    
    parser.add_argument(
        '--charts',
        action='store_true',
        help='Generate individual charts for each model (slower)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare HODL, Grid, and Normal trading strategies (requires --coin and --timeframe)'
    )
    
    parser.add_argument(
        '--grid-levels',
        type=int,
        default=2,
        choices=[2, 3, 4],
        help='Grid levels for strategy comparison (default: 2)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 AI MODEL BACKTESTER (Auto All-Models)")
    print("=" * 60)
    
    # Discover available models
    models = discover_models()
    coins = sorted(set(m['coin'] for m in models))
    timeframes = sorted(set(m['timeframe'] for m in models))
    
    print(f"   Available models: {len(models)}")
    print(f"   Coins: {', '.join(coins)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Device: {DEVICE}")
    print("=" * 60)
    
    # Single model mode
    if args.coin and args.timeframe:
        print(f"\n🎯 Single Model Mode: {args.coin.upper()} / {args.timeframe}")
        
        backtester = Backtester(
            coin=args.coin,
            timeframe=args.timeframe,
            months_back=args.months
        )
        
        # Strategy comparison mode
        if args.compare:
            print("\n📊 Strategy Comparison Mode")
            results = backtester.run_strategy_comparison()
            if not results:
                print("\n❌ Strategy comparison failed.")
                sys.exit(1)
        else:
            # Normal backtest
            results = backtester.run_backtest()
            
            if results:
                backtester.generate_report(results)
            else:
                print("\n❌ Backtest failed. Check the errors above.")
                sys.exit(1)
    
    # Batch mode (all models or filtered)
    else:
        # Batch compare mode - run strategy comparison for all models
        if args.compare:
            # Filter models
            filtered_models = models
            if args.coin:
                filtered_models = [m for m in filtered_models if m['coin'] == args.coin.upper()]
            if args.timeframe:
                filtered_models = [m for m in filtered_models if m['timeframe'] == args.timeframe]
            
            if not filtered_models:
                print("\n❌ No models found matching the criteria.")
                sys.exit(1)
            
            print(f"\n📊 Batch Strategy Comparison Mode: {len(filtered_models)} models")
            print("=" * 80)
            
            all_compare_results = []
            
            for idx, model_info in enumerate(filtered_models, 1):
                coin = model_info['coin']
                timeframe = model_info['timeframe']
                
                print(f"\n{'─' * 60}")
                print(f"📊 [{idx}/{len(filtered_models)}] Comparing strategies for: {coin}/{timeframe}")
                print(f"{'─' * 60}")
                
                try:
                    backtester = Backtester(
                        coin=coin,
                        timeframe=timeframe,
                        months_back=args.months,
                        model_info=model_info
                    )
                    
                    results = backtester.run_strategy_comparison()
                    if results:
                        # Find best strategy for this model
                        best = max(results, key=lambda x: x.total_return_pct)
                        all_compare_results.append({
                            'coin': coin,
                            'timeframe': timeframe,
                            'best_strategy': best.strategy_name,
                            'best_return': best.total_return_pct,
                            'results': results
                        })
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
            
            # Print consolidated summary
            if all_compare_results:
                print("\n" + "=" * 80)
                print("📊 CONSOLIDATED STRATEGY COMPARISON SUMMARY")
                print("=" * 80)
                print(f"\n{'Model':<15} {'Best Strategy':<20} {'Return':<12}")
                print("-" * 50)
                
                for r in sorted(all_compare_results, key=lambda x: x['best_return'], reverse=True):
                    emoji = "🟢" if r['best_return'] > 0 else "🔴"
                    print(f"{r['coin']}_{r['timeframe']:<10} {r['best_strategy']:<20} {emoji}{r['best_return']:>+8.2f}%")
                
                print("-" * 50)
                
                # Count best strategies
                from collections import Counter
                strategy_counts = Counter(r['best_strategy'] for r in all_compare_results)
                print("\n🏆 En İyi Strateji Dağılımı:")
                for strategy, count in strategy_counts.most_common():
                    print(f"   {strategy}: {count} model")
            else:
                print("\n❌ No successful comparisons.")
                sys.exit(1)
        else:
            # Normal batch backtest
            batch_tester = BatchBacktester(
                months_back=args.months,
                coin_filter=args.coin,
                timeframe_filter=args.timeframe
            )
            
            results = batch_tester.run_all(generate_individual_charts=args.charts)
            
            if not results:
                print("\n❌ No successful backtests.")
                sys.exit(1)


if __name__ == "__main__":
    main()
