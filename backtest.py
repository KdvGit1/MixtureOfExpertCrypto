"""
Backtest Module for AI Models
Tests AI prediction models on historical cryptocurrency data.

Usage:
    python backtest.py --coin BTC --timeframe 1h --months 6
    python backtest.py --coin ETH --timeframe 15m --months 3
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import from existing modules (same as app.py)
from train_models.data_fetcher import get_crypto_history, prepare_dual_dataframes
from train_models.CryptoMoeApp.ai_engine_improved import MultiBranchModel, CNN_FEATURES, LSTM_FEATURES, TR_FEATURES

# ============================================
# CONFIGURATION (Same as app.py)
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
import json
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

# Window sizes (same as app.py)
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
    trades: List[TradeResult] = field(default_factory=list)

# ============================================
# BACKTESTER CLASS
# ============================================
class Backtester:
    """Backtest AI models on historical data."""
    
    def __init__(self, coin: str, timeframe: str, months_back: int):
        self.coin = coin.upper()
        self.timeframe = timeframe
        self.months_back = months_back
        self.symbol = f"{self.coin}/USDT"
        self.model: Optional[MultiBranchModel] = None
        
    def load_model(self) -> Optional[MultiBranchModel]:
        """Load the MoE model for the specified timeframe. (Same as app.py)"""
        if self.timeframe not in MODEL_MAP:
            print(f"❌ No model defined for timeframe: {self.timeframe}")
            print(f"   Available: {list(MODEL_MAP.keys())}")
            return None
        
        model_path = MODEL_MAP[self.timeframe]
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return None
        
        params = MODEL_PARAMS.get(self.timeframe, {'embed_dim': 128, 'dropout': 0.15})
        embed_dim = params['embed_dim']
        dropout = params['dropout']
        
        print(f"🧠 Loading MoE model for {self.timeframe}: {model_path}")
        print(f"   Parameters: embed_dim={embed_dim}, dropout={dropout:.2f}")
        
        try:
            model = MultiBranchModel(embed_dim=embed_dim, dropout=dropout).to(DEVICE)
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            
            # Handle DataParallel prefix if present
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict)
            model.eval()
            
            self.model = model
            print(f"✅ Model loaded successfully for {self.timeframe}")
            return model
        except Exception as e:
            print(f"❌ Error loading model for {self.timeframe}: {e}")
            return None
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical data using existing data_fetcher function."""
        print(f"\n📊 Fetching {self.months_back} months of {self.timeframe} data for {self.symbol}...")
        
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
        """Prepare the input tensors for the MultiBranchModel. (Same as app.py)"""
        # Get column indices for each branch
        cnn_cols = [df_ai.columns.get_loc(c) for c in CNN_FEATURES if c in df_ai.columns]
        lstm_cols = [df_ai.columns.get_loc(c) for c in LSTM_FEATURES if c in df_ai.columns]
        tr_cols = [df_ai.columns.get_loc(c) for c in TR_FEATURES if c in df_ai.columns]
        
        # Normalize the data (use all available data up to end_idx for stats)
        df_subset = df_ai.iloc[:end_idx]
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
    
    def run_backtest(self) -> Optional[BacktestResults]:
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
        
        print(f"\n🚀 Starting backtest from index {LSTM_WINDOW} to {len(df_ai) - 1}...")
        
        trades: List[TradeResult] = []
        correct_direction = 0
        max_window = max(CNN_WINDOW, LSTM_WINDOW, TR_WINDOW)
        
        # Sliding window backtest
        total_iterations = len(df_ai) - max_window - 1
        for i in range(max_window, len(df_ai) - 1):
            # Progress indicator
            progress = (i - max_window + 1) / total_iterations * 100
            if (i - max_window) % 500 == 0:
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
        cumulative_return = sum([t.actual if t.prediction > 0 else -t.actual for t in trades])
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum([t.actual if t.prediction > 0 else -t.actual for t in trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
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
            trades=trades
        )
        
        print(f"\n✅ Backtest complete! Processed {len(trades)} predictions.")
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
        
        # Save to CSV and get output path
        output_dir, file_prefix = self.save_results_to_csv(results)
        
        # Generate charts
        self.generate_charts(results, output_dir, file_prefix)
    
    def save_results_to_csv(self, results: BacktestResults) -> None:
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
        
        print(f"📝 Summary saved to: {summary_filepath}")
        
        # Return filepath prefix for charts
        return output_dir, f"backtest_{results.coin}_{results.timeframe}_{results.months}m_{timestamp}"
    
    def generate_charts(self, results: BacktestResults, output_dir: Path, file_prefix: str) -> None:
        """Generate and save backtest result charts."""
        print("\n📊 Generating charts...")
        
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
        # Add diagonal reference line
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
        
        # 4. Return Distribution (with separate y-axes due to different scales)
        ax4 = axes[1, 1]
        
        # Clip to reasonable percentiles to avoid extreme outliers stretching the axis
        pred_clip = np.percentile(predictions, [1, 99])
        actual_clip = np.percentile(actuals, [1, 99])
        
        # Use combined range but clipped
        x_min = min(pred_clip[0], actual_clip[0]) - 0.5
        x_max = max(pred_clip[1], actual_clip[1]) + 0.5
        
        # Create bins within the clipped range
        bins = np.linspace(x_min, x_max, 50)
        
        # Plot actuals first (they're more important)
        ax4.hist(actuals, bins=bins, alpha=0.7, color='#e74c3c', label='Actual Returns', edgecolor='white')
        
        # Create twin axis for predictions (different scale)
        ax4_twin = ax4.twinx()
        ax4_twin.hist(predictions, bins=bins, alpha=0.5, color='#3498db', label='Predictions', edgecolor='white')
        
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
        ax4.set_title('Return Distribution', fontweight='bold')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Actual Count', color='#e74c3c')
        ax4_twin.set_ylabel('Prediction Count', color='#3498db')
        ax4.tick_params(axis='y', labelcolor='#e74c3c')
        ax4_twin.tick_params(axis='y', labelcolor='#3498db')
        
        # Combined legend
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
# CLI INTERFACE
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Backtest AI models on historical cryptocurrency data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --coin BTC --timeframe 1h --months 6
  python backtest.py --coin ETH --timeframe 15m --months 3
  python backtest.py -c SOL -t 1h -m 12
        """
    )
    
    parser.add_argument(
        '-c', '--coin',
        type=str,
        required=True,
        help='Coin symbol (e.g., BTC, ETH, SOL)'
    )
    
    parser.add_argument(
        '-t', '--timeframe',
        type=str,
        required=True,
        choices=['15m', '1h'],
        help='Timeframe for model (15m or 1h)'
    )
    
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=6,
        help='Months of historical data to test (default: 6)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 AI MODEL BACKTESTER")
    print("=" * 60)
    print(f"   Coin: {args.coin.upper()}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Months: {args.months}")
    print(f"   Device: {DEVICE}")
    print("=" * 60)
    
    # Run backtest
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


if __name__ == "__main__":
    main()
