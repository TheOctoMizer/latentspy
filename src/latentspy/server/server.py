import asyncio
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="LatentSpy API Server")

# Add CORS Middleware to allow external dashboards to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for the dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DB_PATH = Path(".") / ".latentspy" / "runs" / "runs.db"
UI_PATH = Path(__file__).parent / "ui"

# Mount static files for UI
app.mount("/static", StaticFiles(directory=str(UI_PATH)), name="static")

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML file."""
    return FileResponse(UI_PATH / "index.html")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

async def get_latest_metrics(last_id: int, experiment_id: int) -> list:
    """Fetch all metrics with ID > last_id for a specific experiment."""
    loop = asyncio.get_event_loop()
    def query():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, step, layer_name, metric_name, value, is_validation, timestamp FROM metrics WHERE id > ? AND experiment_id = ? ORDER BY id ASC",
            (last_id, experiment_id)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    return await loop.run_in_executor(None, query)

async def get_latest_alerts(last_id: int, experiment_id: int) -> list:
    """Fetch all alerts with ID > last_id for a specific experiment."""
    loop = asyncio.get_event_loop()
    def query():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, step, layer_name, level, message, timestamp FROM health_alerts WHERE id > ? AND experiment_id = ? ORDER BY id ASC",
            (last_id, experiment_id)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    return await loop.run_in_executor(None, query)

async def get_latest_projections(last_id: int, experiment_id: int) -> list:
    """Fetch all projections with ID > last_id for a specific experiment."""
    loop = asyncio.get_event_loop()
    def query():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, step, layer_name, x, y, z, cluster_id FROM projections WHERE id > ? AND experiment_id = ? ORDER BY id ASC LIMIT 2000",
            (last_id, experiment_id)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    return await loop.run_in_executor(None, query)

async def get_latest_experiment_id() -> int:
    """Get the ID of the most recently created experiment."""
    loop = asyncio.get_event_loop()
    def query():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM experiments ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    return await loop.run_in_executor(None, query)

async def get_all_experiments() -> list:
    """Get all experiments from the database."""
    loop = asyncio.get_event_loop()
    def query():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, created_at FROM experiments ORDER BY created_at DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    return await loop.run_in_executor(None, query)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        last_metric_id = 0
        last_alert_id = 0
        last_projection_id = 0
        current_exp_id = await get_latest_experiment_id()
        last_known_exp_id = current_exp_id
        
        # Send initial experiment info
        if current_exp_id:
            await manager.send_personal_message(
                json.dumps({"type": "new_experiment", "data": {"id": current_exp_id}}),
                websocket
            )
        
        while True:
            # Check for new experiment
            new_exp_id = await get_latest_experiment_id()
            if new_exp_id and new_exp_id != current_exp_id:
                # Send experiment ended message for old experiment
                if current_exp_id:
                    await manager.send_personal_message(
                        json.dumps({"type": "experiment_ended", "data": {"id": current_exp_id}}),
                        websocket
                    )
                
                current_exp_id = new_exp_id
                last_metric_id = 0
                last_alert_id = 0
                last_projection_id = 0
                await manager.send_personal_message(
                    json.dumps({"type": "new_experiment", "data": {"id": current_exp_id}}),
                    websocket
                )

            # Only fetch data if we have an active experiment
            if current_exp_id:
                # Fetch new metrics
                new_metrics = await get_latest_metrics(last_metric_id, current_exp_id)
                if new_metrics:
                    for m in new_metrics:
                        last_metric_id = max(last_metric_id, m["id"])
                    
                    # Send metrics in a batch
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "metrics_batch",
                            "data": new_metrics,
                            "experiment_id": current_exp_id
                        }),
                        websocket
                    )

                # Fetch new alerts
                new_alerts = await get_latest_alerts(last_alert_id, current_exp_id)
                if new_alerts:
                    for a in new_alerts:
                        last_alert_id = max(last_alert_id, a["id"])
                    
                    # Send alerts in a batch
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "alerts_batch",
                            "data": new_alerts,
                            "experiment_id": current_exp_id
                        }),
                        websocket
                    )

                # Fetch new projections
                new_projections = await get_latest_projections(last_projection_id, current_exp_id)
                if new_projections:
                    for p in new_projections:
                        last_projection_id = max(last_projection_id, p["id"])
                    
                    # Send projections in a batch
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "projections_batch",
                            "data": new_projections,
                            "experiment_id": current_exp_id
                        }),
                        websocket
                    )

            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/experiments")
async def get_experiments():
    """API endpoint to fetch all experiments."""
    try:
        experiments = await get_all_experiments()
        return experiments
    except Exception as e:
        return {"error": str(e), "experiments": []}

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
