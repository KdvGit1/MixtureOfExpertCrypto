import asyncio
import logging
from collections import deque
from datetime import datetime
import ccxt.async_support as ccxt

logger = logging.getLogger("price_engine")

class PriceEngine:
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = ["ETH/USDT"]
        self.symbols = symbols
        self.exchange = None
        self.running = False
        
        # Keep track of latest data and tick history (last 300 prices)
        self.latest_data = {}
        self.price_history = {symbol: deque(maxlen=300) for symbol in self.symbols}
        self.callbacks = []

    async def initialize(self):
        """Initialize CCXT exchange and pre-populate price history with historical candles."""
        logger.info("🔌 Initializing live price engine with Binance futures...")
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'futures'}
        })
        # Warm up cache
        await self.exchange.load_markets()
        
        # Pre-populate history with the last 300 15-minute candles so the chart is instantly beautiful
        logger.info("🕯️ Pre-populating chart history with 300 historical candles (15m)...")
        for symbol in self.symbols:
            try:
                # Fetch 300 15-minute candles
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe='15m', limit=300)
                for candle in ohlcv:
                    # candle format: [timestamp, open, high, low, close, volume]
                    time_str = datetime.fromtimestamp(candle[0] / 1000).strftime('%H:%M:%S')
                    close_price = candle[4]
                    self.price_history[symbol].append({
                        "time": time_str,
                        "price": close_price,
                        "timestamp": candle[0]  # Store opening timestamp in ms
                    })
                logger.info(f"   ✅ Pre-populated {len(ohlcv)} history points for {symbol}")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not pre-populate history for {symbol}: {e}")

    def register_callback(self, callback):
        """Register a callback for when new tick data arrives."""
        self.callbacks.append(callback)

    async def start(self):
        """Start the live price fetching loop."""
        if not self.exchange:
            await self.initialize()
            
        self.running = True
        asyncio.create_task(self._fetch_loop())
        logger.info("⚡ Live price engine started.")

    async def stop(self):
        """Stop the price engine and close connection."""
        self.running = False
        if self.exchange:
            await self.exchange.close()
            logger.info("🔌 Price engine exchange connection closed.")

    async def _fetch_loop(self):
        """Background loop for fetching prices every 3 seconds."""
        while self.running:
            try:
                tasks = [self._fetch_symbol(symbol) for symbol in self.symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check if any errors occurred
                has_updates = False
                for r in results:
                    if isinstance(r, dict) and "error" not in r:
                        has_updates = True
                
                # Trigger callbacks
                if has_updates and self.callbacks:
                    tick_payload = {
                        "type": "tick",
                        "timestamp": datetime.now().isoformat(),
                        "data": self.latest_data
                    }
                    for cb in self.callbacks:
                        try:
                            if asyncio.iscoroutinefunction(cb):
                                await cb(tick_payload)
                            else:
                                cb(tick_payload)
                        except Exception as e:
                            logger.error(f"❌ Callback error: {e}")
                            
            except Exception as e:
                logger.error(f"❌ Error in price fetch loop: {e}")
                
            await asyncio.sleep(3)

    async def _fetch_symbol(self, symbol):
        """Fetch ticker for a single symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            price = ticker.get("last", 0.0)
            high24h = ticker.get("high", 0.0)
            low24h = ticker.get("low", 0.0)
            volume24h = ticker.get("baseVolume", 0.0)
            percentage = ticker.get("percentage", 0.0)
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Save or update history on 15m candle boundary
            import time as pytime
            now_ms = int(pytime.time() * 1000)
            candle_interval_ms = 15 * 60 * 1000
            current_candle_timestamp = now_ms - (now_ms % candle_interval_ms)
            
            if len(self.price_history[symbol]) > 0:
                last_item = self.price_history[symbol][-1]
                if last_item.get("timestamp") == current_candle_timestamp:
                    # Update active live candle close price
                    last_item["price"] = price
                    last_item["time"] = timestamp
                else:
                    # Append new candle
                    self.price_history[symbol].append({
                        "time": timestamp,
                        "price": price,
                        "timestamp": current_candle_timestamp
                    })
            else:
                self.price_history[symbol].append({
                    "time": timestamp,
                    "price": price,
                    "timestamp": current_candle_timestamp
                })
            
            # Update latest
            self.latest_data[symbol] = {
                "price": price,
                "high24h": high24h,
                "low24h": low24h,
                "volume24h": volume24h,
                "change24h": percentage,
                "history": list(self.price_history[symbol]),
                "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return self.latest_data[symbol]
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} ticker: {e}")
            return {"error": str(e)}
