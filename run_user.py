#!/usr/bin/env python3
"""
================================================================================
🚀 MULTI-USER BOT LAUNCHER
================================================================================
Kullanıcı adı veya config dosyası ile bot başlatma.

Kullanım:
    python run_user.py kaan              # users/kaan/config.json kullanır
    python run_user.py --config path.json # Özel config dosyası kullanır
    python run_user.py --list            # Kayıtlı kullanıcıları listele

================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent


def list_users():
    """List all registered users."""
    users_dir = PROJECT_ROOT / "users"
    if not users_dir.exists():
        print("❌ users/ klasörü bulunamadı!")
        return []
    
    users = []
    for item in users_dir.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            config_file = item / "config.json"
            if config_file.exists():
                users.append(item.name)
    return users


def load_user_config(username: str) -> dict:
    """Load config for a specific user."""
    config_path = PROJECT_ROOT / "users" / username / "config.json"
    
    if not config_path.exists():
        print(f"❌ Kullanıcı bulunamadı: {username}")
        print(f"   Beklenen config dosyası: {config_path}")
        print(f"\n💡 Yeni kullanıcı eklemek için: python setup_user.py")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Add user directory paths
    user_dir = PROJECT_ROOT / "users" / username
    config['_user_dir'] = str(user_dir)
    config['_log_dir'] = str(user_dir / "bot_logs")
    config['_history_dir'] = str(user_dir / "trade_history")
    
    # Ensure directories exist
    (user_dir / "bot_logs").mkdir(parents=True, exist_ok=True)
    (user_dir / "trade_history").mkdir(parents=True, exist_ok=True)
    
    return config


def load_config_from_path(config_path: str) -> dict:
    """Load config from a specific file path."""
    path = Path(config_path)
    
    if not path.exists():
        print(f"❌ Config dosyası bulunamadı: {config_path}")
        sys.exit(1)
    
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Use parent directory for user files
    user_dir = path.parent
    config['_user_dir'] = str(user_dir)
    config['_log_dir'] = str(user_dir / "bot_logs")
    config['_history_dir'] = str(user_dir / "trade_history")
    
    # Ensure directories exist
    (user_dir / "bot_logs").mkdir(parents=True, exist_ok=True)
    (user_dir / "trade_history").mkdir(parents=True, exist_ok=True)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='🤖 Multi-User Crypto Trading Bot Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python run_user.py kaan              Kaan'ın botunu başlat
  python run_user.py --config x.json   Özel config dosyasıyla başlat
  python run_user.py --list            Tüm kullanıcıları listele
        """
    )
    
    parser.add_argument('username', nargs='?', help='Kullanıcı adı (users/USER_ADI/config.json)')
    parser.add_argument('--config', '-c', help='Özel config dosyası yolu')
    parser.add_argument('--list', '-l', action='store_true', help='Kayıtlı kullanıcıları listele')
    
    args = parser.parse_args()
    
    # List users
    if args.list:
        users = list_users()
        if users:
            print("📋 Kayıtlı Kullanıcılar:")
            for user in users:
                print(f"   • {user}")
            print(f"\n💡 Başlatmak için: python run_user.py <kullanici_adi>")
        else:
            print("❌ Henüz kayıtlı kullanıcı yok.")
            print("💡 Yeni kullanıcı eklemek için: python setup_user.py")
        return
    
    # Load config
    if args.config:
        config = load_config_from_path(args.config)
        print(f"📂 Config yüklendi: {args.config}")
    elif args.username:
        config = load_user_config(args.username)
        print(f"👤 Kullanıcı yüklendi: {args.username}")
    else:
        parser.print_help()
        print("\n❌ Kullanıcı adı veya --config gerekli!")
        return
    
    # Print user info
    username = config.get('username', 'unknown')
    coins = config.get('coins_to_trade', [])
    mode = config.get('trading_mode', 'spot')
    testnet = config.get('testnet', True)
    dry_run = config.get('dry_run', True)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  🤖 CRYPTO TRADING BOT - {username.upper():^20}           ║
╠══════════════════════════════════════════════════════════════════╣
║  📊 Coinler: {', '.join(coins[:5]):<40} ║
║  📈 Mod: {mode.upper():<10} | {'TESTNET' if testnet else 'MAINNET':<10} | {'DRY-RUN' if dry_run else 'GERÇEK':<10} ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Import and run the bot with this config
    # Set environment variable for the bot to use
    os.environ['BOT_USER_CONFIG'] = json.dumps(config)
    os.environ['BOT_USER_DIR'] = config['_user_dir']
    os.environ['BOT_LOG_DIR'] = config['_log_dir']
    os.environ['BOT_HISTORY_DIR'] = config['_history_dir']
    os.environ['BOT_USERNAME'] = config.get('username', 'unknown')
    
    # Run the bot
    try:
        from telegram_trading_bot import TelegramAutoTradingBot, BotConfig
        
        # Create BotConfig from user config
        bot_config = BotConfig(
            # API Keys
            api_key=config.get('binance_api_key', ''),
            api_secret=config.get('binance_api_secret', ''),
            real_api_key=config.get('real_api_key', ''),
            real_api_secret=config.get('real_api_secret', ''),
            
            # Telegram
            telegram_bot_token=config.get('telegram_bot_token', ''),
            telegram_chat_id=config.get('telegram_chat_id', ''),
            
            # Trading Mode
            testnet=config.get('testnet', True),
            dry_run=config.get('dry_run', True),
            trading_mode=config.get('trading_mode', 'spot'),
            leverage=config.get('leverage', 5),
            
            # Coins
            coins_to_trade=config.get('coins_to_trade', ['BTC', 'ETH', 'SOL']),
            default_timeframe=config.get('default_timeframe', '15m'),
            max_positions=config.get('max_positions', 3),
            
            # Thresholds
            position_pct=config.get('position_pct', 0.40),
            prediction_threshold=config.get('prediction_threshold', 0.003),
            prediction_scale=config.get('prediction_scale', 0.5),
            min_profit_to_exit=config.get('min_profit_to_exit', 0.0025),
            min_confidence_threshold=config.get('min_confidence_threshold', 25.0),
            
            # SL/TP
            spot_sl_pct=config.get('spot_sl_pct', -5.0),
            spot_tp_pct=config.get('spot_tp_pct', 5.0),
            futures_sl_pct=config.get('futures_sl_pct', -20.0),
            futures_tp_pct=config.get('futures_tp_pct', 20.0),
            
            # Safety
            daily_loss_limit_pct=config.get('daily_loss_limit_pct', -25.0),
            max_daily_trades=config.get('max_daily_trades', 50),
            min_balance_usdt=config.get('min_balance_usdt', 5.0),
            
            # Grid
            grid_enabled=config.get('grid_enabled', True),
            grid_levels=config.get('grid_levels', 2),
            
            # Pi Optimization
            max_loaded_models=config.get('max_loaded_models', 5),
            loop_interval_seconds=config.get('loop_interval_seconds', 60),
        )
        
        # Create and run bot
        import asyncio
        bot = TelegramAutoTradingBot(bot_config)
        
        print("🚀 Bot başlatılıyor...")
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        print("\n⛔ Bot durduruldu (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
