#!/usr/bin/env python3
"""
================================================================================
👤 KULLANICI KURULUM SCRIPTI
================================================================================
Yeni kullanıcı eklemek için interaktif script.

Kullanım:
    python setup_user.py
    python setup_user.py --name kaan --quick

================================================================================
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def get_input(prompt: str, default: str = None, required: bool = True) -> str:
    """Get user input with optional default."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    value = input(prompt).strip()
    
    if not value and default:
        return default
    
    if not value and required:
        print("❌ Bu alan zorunlu!")
        return get_input(prompt.split(':')[0], default, required)
    
    return value


def get_bool_input(prompt: str, default: bool = True) -> bool:
    """Get boolean input."""
    default_str = "E/h" if default else "e/H"
    value = input(f"{prompt} [{default_str}]: ").strip().lower()
    
    if not value:
        return default
    
    return value in ['e', 'evet', 'y', 'yes', 'true', '1']


def get_list_input(prompt: str, default: list = None) -> list:
    """Get comma-separated list input."""
    default_str = ', '.join(default) if default else ''
    value = input(f"{prompt} [{default_str}]: ").strip()
    
    if not value and default:
        return default
    
    if not value:
        return []
    
    return [item.strip().upper() for item in value.split(',')]


def setup_new_user():
    """Interactive setup for a new user."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       👤 YENİ KULLANICI KURULUMU                                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Username
    username = get_input("Kullanıcı adı (boşluksuz, küçük harf)").lower().replace(' ', '_')
    
    user_dir = PROJECT_ROOT / "users" / username
    if user_dir.exists():
        overwrite = get_bool_input(f"⚠️ '{username}' zaten mevcut. Üzerine yazılsın mı?", False)
        if not overwrite:
            print("❌ İptal edildi.")
            return
    
    print(f"\n📁 Kullanıcı klasörü: {user_dir}")
    
    # Telegram
    print("\n" + "=" * 60)
    print("📱 TELEGRAM AYARLARI")
    print("=" * 60)
    print("💡 @BotFather'dan yeni bot oluşturun: https://t.me/BotFather")
    print("💡 Chat ID için @userinfobot kullanın: https://t.me/userinfobot")
    
    telegram_token = get_input("Telegram Bot Token")
    telegram_chat_id = get_input("Telegram Chat ID")
    
    # Binance API
    print("\n" + "=" * 60)
    print("🔐 BINANCE API AYARLARI (TESTNET)")
    print("=" * 60)
    print("💡 Testnet key: https://testnet.binance.vision/")
    
    testnet_api_key = get_input("Testnet API Key", required=False) or ""
    testnet_api_secret = get_input("Testnet API Secret", required=False) or ""
    
    print("\n" + "=" * 60)
    print("🔐 BINANCE API AYARLARI (MAINNET - Opsiyonel)")
    print("=" * 60)
    print("💡 Mainnet key: https://www.binance.com/en/my/settings/api-management")
    
    real_api_key = get_input("Mainnet API Key (boş bırakılabilir)", required=False) or ""
    real_api_secret = get_input("Mainnet API Secret (boş bırakılabilir)", required=False) or ""
    
    # Coins
    print("\n" + "=" * 60)
    print("💰 TRADE EDİLECEK COİNLER")
    print("=" * 60)
    print("💡 Mevcut: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, LINK, LTC")
    print("          AVAX, ATOM, FIL, TRX, UNI, MATIC, APT, ARB, OP, INJ")
    
    coins = get_list_input("Coinler (virgülle ayır)", ["BTC", "ETH", "SOL"])
    
    # Trading Mode
    print("\n" + "=" * 60)
    print("📈 TRADING AYARLARI")
    print("=" * 60)
    
    trading_mode = get_input("Trading modu (spot/futures)", "spot").lower()
    if trading_mode not in ['spot', 'futures']:
        trading_mode = 'spot'
    
    leverage = 5
    if trading_mode == 'futures':
        leverage = int(get_input("Kaldıraç (1-125)", "5"))
    
    testnet = get_bool_input("Testnet kullan (demo trading)?", True)
    dry_run = get_bool_input("Dry-run modu (simülasyon)?", True)
    
    max_positions = int(get_input("Maksimum pozisyon sayısı", "3"))
    position_pct = float(get_input("Bakiye yüzdesi (0.1-1.0)", "0.40"))
    
    # Safety
    print("\n" + "=" * 60)
    print("🛡️ GÜVENLİK AYARLARI")
    print("=" * 60)
    
    spot_sl = float(get_input("Spot Stop Loss % (örn: -5)", "-5"))
    spot_tp = float(get_input("Spot Take Profit % (örn: 5)", "5"))
    futures_sl = float(get_input("Futures Stop Loss % (örn: -20)", "-20"))
    futures_tp = float(get_input("Futures Take Profit % (örn: 20)", "20"))
    daily_loss_limit = float(get_input("Günlük kayıp limiti % (örn: -25)", "-25"))
    
    # Create config
    config = {
        "username": username,
        
        "binance_api_key": testnet_api_key,
        "binance_api_secret": testnet_api_secret,
        "real_api_key": real_api_key,
        "real_api_secret": real_api_secret,
        
        "telegram_bot_token": telegram_token,
        "telegram_chat_id": telegram_chat_id,
        
        "coins_to_trade": coins,
        "default_timeframe": "15m",
        
        "trading_mode": trading_mode,
        "leverage": leverage,
        "testnet": testnet,
        "dry_run": dry_run,
        
        "position_pct": position_pct,
        "max_positions": max_positions,
        "prediction_threshold": 0.003,
        "prediction_scale": 0.5,
        "min_profit_to_exit": 0.0025,
        "min_confidence_threshold": 25.0,
        
        "spot_sl_pct": spot_sl,
        "spot_tp_pct": spot_tp,
        "futures_sl_pct": futures_sl,
        "futures_tp_pct": futures_tp,
        
        "daily_loss_limit_pct": daily_loss_limit,
        "max_daily_trades": 50,
        "min_balance_usdt": 5.0,
        
        "grid_enabled": True,
        "grid_levels": 2,
        
        "max_loaded_models": 5,
        "loop_interval_seconds": 60
    }
    
    # Create directories
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "bot_logs").mkdir(exist_ok=True)
    (user_dir / "trade_history").mkdir(exist_ok=True)
    
    # Save config
    config_path = user_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ✅ KULLANICI BAŞARIYLA OLUŞTURULDU!                            ║
╠══════════════════════════════════════════════════════════════════╣
║  👤 Kullanıcı: {username:<47} ║
║  📁 Klasör: users/{username:<43} ║
║  📊 Coinler: {', '.join(coins[:5]):<48} ║
║  📈 Mod: {trading_mode.upper():<10} | {'TESTNET' if testnet else 'MAINNET':<10} | {'DRY-RUN' if dry_run else 'GERÇEK':<10} ║
╚══════════════════════════════════════════════════════════════════╝

🚀 Botu başlatmak için:
   python run_user.py {username}

🔧 Raspberry Pi'da service olarak çalıştırmak için:
   sudo systemctl enable bot@{username}
   sudo systemctl start bot@{username}
    """)


def main():
    parser = argparse.ArgumentParser(description='👤 Yeni kullanıcı kurulum scripti')
    parser.add_argument('--name', '-n', help='Kullanıcı adı (interaktif mod atlanır)')
    parser.add_argument('--quick', '-q', action='store_true', help='Varsayılan değerlerle hızlı kurulum')
    
    args = parser.parse_args()
    
    if args.name and args.quick:
        print(f"⚡ Hızlı kurulum: {args.name}")
        # Create with defaults - requires manual config edit
        user_dir = PROJECT_ROOT / "users" / args.name
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "bot_logs").mkdir(exist_ok=True)
        (user_dir / "trade_history").mkdir(exist_ok=True)
        
        # Copy template
        template_path = PROJECT_ROOT / "users" / "_template" / "config.json"
        if template_path.exists():
            shutil.copy(template_path, user_dir / "config.json")
            print(f"✅ {args.name} oluşturuldu. Config dosyasını düzenleyin:")
            print(f"   {user_dir / 'config.json'}")
        else:
            print("❌ Template bulunamadı: users/_template/config.json")
    else:
        setup_new_user()


if __name__ == "__main__":
    main()
