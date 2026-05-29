import os
import sys
import logging
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Setup sys path so we can import from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from live_streamer.price_engine import PriceEngine
from live_streamer.analysis_runner import AnalysisRunner

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_streamer")

app = FastAPI(title="Crypto MoE Streamer API", version="1.0.0")

# Enable CORS for local streaming overlays
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSockets connection pool
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 New client connected. Total clients: {len(self.active_connections)}")
        
        # Send initial cache if available
        if price_engine.latest_data:
            await self.send_personal_message({
                "type": "tick",
                "data": price_engine.latest_data
            }, websocket)
        if analysis_runner.latest_analysis:
            await self.send_personal_message({
                "type": "analysis",
                "data": analysis_runner.latest_analysis
            }, websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"🔌 Client disconnected. Total clients: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception:
            pass

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
                
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Initialize Engines
price_engine = PriceEngine()
analysis_runner = AnalysisRunner(price_engine, interval=60)

# Connect runner/engine callbacks to WS broadcast
async def on_tick_update(payload):
    await manager.broadcast(payload)

async def on_analysis_update(payload):
    await manager.broadcast(payload)

price_engine.register_callback(on_tick_update)
analysis_runner.register_callback(on_analysis_update)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting live streamer backend services...")
    await price_engine.initialize()
    await analysis_runner.initialize()
    
    await price_engine.start()
    await analysis_runner.start()
    logger.info("✅ Live streamer services initialized successfully.")

class ModelPathRequest(BaseModel):
    timeframe: str
    path: str

@app.post("/api/config/models")
async def configure_model(req: ModelPathRequest):
    if os.getenv("PUBLIC_MODE", "false").lower() == "true":
        return {"success": False, "message": "Model configuration is disabled in public mode."}
    if req.timeframe not in ['15m']:
        return {"success": False, "message": "Invalid timeframe. Only '15m' timeframe is active."}
    success, message = await analysis_runner.load_custom_model(req.timeframe, req.path)
    return {"success": success, "message": message}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down live streamer services...")
    await price_engine.stop()
    await analysis_runner.stop()
    logger.info("👋 Live streamer services stopped.")

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep-alive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# HTML overlay root
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        with open(index_path, encoding='utf-8') as f:
            content = f.read()
            if os.getenv("PUBLIC_MODE", "false").lower() == "true":
                content = content.replace("</body>", "<script>window.PUBLIC_MODE = true;</script></body>")
            return content
    return "<h1>Streamer overlay index.html not found! Check live_streamer/static directory.</h1>"

# Mount static files (JS, CSS, audio, graphics)
static_path = Path(__file__).parent / "static"
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

if __name__ == "__main__":
    import uvicorn
    logger.info("🔥 Starting Uvicorn Server on port 8888...")
    uvicorn.run(app, host="127.0.0.1", port=8888)
