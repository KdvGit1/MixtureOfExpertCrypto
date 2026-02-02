"""
Backtest Module for AI Models (Auto All-Models Version)
Tests all AI prediction models from kaggle_outputs on historical cryptocurrency data.

Usage:
    python backtest.py                        # Run all 40 models (all coins, all timeframes)
    python backtest.py --coin BTC             # Run only BTC models (both timeframes)
    python backtest.py --timeframe 1h         # Run only 1h models (all coins)
    python backtest.py --coin BTC --timeframe 1h  # Run single model
    python backtest.py --months 6             # Set backtest period (default: 3)
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
        
        # Add labels for best/worst models
        best_idx = df['accuracy'].idxmax()
        worst_idx = df['accuracy'].idxmin()
        for idx in [best_idx, worst_idx]:
            ax3.annotate(df.loc[idx, 'model'], 
                        (df.loc[idx, 'accuracy'], df.loc[idx, 'cumulative_return_pct']),
                        fontsize=8, ha='left')
        
        # 4. Distribution histograms
        ax4 = axes[1, 1]
        ax4.hist(df['accuracy'], bins=15, alpha=0.7, color='#3498db', label='Accuracy', edgecolor='white')
        ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        ax4.axvline(x=df['accuracy'].mean(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean ({df["accuracy"].mean():.1f}%)')
        ax4.set_xlabel('Accuracy (%)')
        ax4.set_ylabel('Number of Models')
        ax4.set_title('Accuracy Distribution', fontweight='bold')
        ax4.legend()
        
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
        
        results = backtester.run_backtest()
        
        if results:
            backtester.generate_report(results)
        else:
            print("\n❌ Backtest failed. Check the errors above.")
            sys.exit(1)
    
    # Batch mode (all models or filtered)
    else:
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
