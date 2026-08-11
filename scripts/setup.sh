#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VENV_PATH="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install -e "$PROJECT_ROOT[dev]"
"$VENV_PATH/bin/python" -m market_moe.cli doctor

printf '%s\n' "MarketMoE kurulumu tamamlandı."
printf '%s\n' "Başlatmak için: .venv/bin/market-moe serve"
