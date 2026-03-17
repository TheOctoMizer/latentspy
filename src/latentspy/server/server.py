import asyncio
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
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

@app.get("/events")
async def event_stream(request: Request):
    async def event_generator() -> AsyncGenerator:
        last_metric_id = 0
        last_alert_id = 0
        current_exp_id = await get_latest_experiment_id()
        
        while True:
            # Check for client disconnect
            if await request.is_disconnected():
                break

            if not current_exp_id:
                current_exp_id = await get_latest_experiment_id()
                await asyncio.sleep(1)
                continue

            # Check if a new experiment started
            new_exp_id = await get_latest_experiment_id()
            if new_exp_id and new_exp_id != current_exp_id:
                current_exp_id = new_exp_id
                last_metric_id = 0 # Reset for new experiment
                last_alert_id = 0
                yield {
                    "event": "new_experiment",
                    "data": json.dumps({"id": current_exp_id})
                }

            # Fetch new data
            new_metrics = await get_latest_metrics(last_metric_id, current_exp_id)
            for m in new_metrics:
                last_metric_id = max(last_metric_id, m["id"])
                yield {
                    "event": "metric",
                    "data": json.dumps(m)
                }

            new_alerts = await get_latest_alerts(last_alert_id, current_exp_id)
            for a in new_alerts:
                last_alert_id = max(last_alert_id, a["id"])
                yield {
                    "event": "alert",
                    "data": json.dumps(a)
                }

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
