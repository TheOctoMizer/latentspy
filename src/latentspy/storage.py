from collections import defaultdict
from pathlib import Path
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional


class MetricStorage:
    def __init__(self, experiment_name: Optional[str] = None, log_type: str = "db"):
        self.history = defaultdict(lambda: defaultdict(list))
        self.experiment_name = experiment_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_type = log_type.lower()
        
        self.run_storage: Path = Path(".") / ".latentspy" / "runs"
        self.run_storage.mkdir(parents=True, exist_ok=True)
        self.run_database: Path = self.run_storage / "runs.db"
        
        # self.json_path = self.run_storage / f"{self.experiment_name}.json"
        self.json_path = self.run_storage / f"{self.experiment_name}.jsonl"
        self.csv_path = self.run_storage / f"{self.experiment_name}.csv"
        
        if self.log_type == "json" and not self.json_path.exists():
            with open(self.json_path, 'w') as f:
                json.dump({"experiment": self.experiment_name, "created_at": datetime.now().isoformat(), "metrics": []}, f)
        
        if self.log_type == "csv" and not self.csv_path.exists():
            with open(self.csv_path, 'w') as f:
                f.write("step,layer,metric,value,is_validation,timestamp\n")

        self.conn = sqlite3.connect(self.run_database, timeout=5.0)
        self._init_database()

    def _init_database(self):
        """Initialize database tables for storing experiment data."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                step INTEGER NOT NULL,
                layer_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                is_validation BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)

        # NEW: Health Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                step INTEGER NOT NULL,
                layer_name TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_experiment_step ON metrics(experiment_id, step)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_layer_metric ON metrics(layer_name, metric_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_experiment ON health_alerts(experiment_id)")
        
        self.conn.commit()
        self._register_experiment()

    def _register_experiment(self):
        """Register the current experiment in the database, updating the timestamp if it exists."""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        metadata = json.dumps({"created_at": timestamp})
        
        # Using a more compatible approach than ON CONFLICT if version is old
        cursor.execute("SELECT id FROM experiments WHERE name = ?", (self.experiment_name,))
        res = cursor.fetchone()
        
        if res:
            cursor.execute(
                "UPDATE experiments SET created_at = ?, metadata = ? WHERE id = ?",
                (timestamp, metadata, res[0])
            )
        else:
            cursor.execute(
                "INSERT INTO experiments (name, created_at, metadata) VALUES (?, ?, ?)",
                (self.experiment_name, timestamp, metadata)
            )
        self.conn.commit()

    def log_alert(self, step: int, layer_name: str, level: str, message: str):
        """Log a health alert to the database and/or file."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM experiments WHERE name = ?", (self.experiment_name,))
        experiment_id = cursor.fetchone()[0]
        
        cursor.execute(
            "INSERT INTO health_alerts (experiment_id, step, layer_name, level, message) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, step, layer_name, level, message)
        )
        self.conn.commit()

        if self.log_type == "json":
            self._stream_json({"type": "alert", "step": step, "layer": layer_name, "level": level, "message": message})

    def update(self, results: Dict[str, Dict[str, Any]], step: int, is_validation: bool = False):
        """Update storage with new metrics."""
        if not results and step > 0: return 
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM experiments WHERE name = ?", (self.experiment_name,))
        res = cursor.fetchone()
        if not res:
            self._register_experiment()
            cursor.execute("SELECT id FROM experiments WHERE name = ?", (self.experiment_name,))
            res = cursor.fetchone()
        experiment_id = res[0]
        
        timestamp = datetime.now().isoformat()
        
        for layer_name, metrics in results.items():
            for metric_name, value in metrics.items():
                val_f = float(value)
                self.history[layer_name][metric_name].append((step, val_f))
                
                # DB Storage
                cursor.execute(
                    "INSERT INTO metrics (experiment_id, step, layer_name, metric_name, value, is_validation) VALUES (?, ?, ?, ?, ?, ?)",
                    (experiment_id, step, layer_name, metric_name, val_f, is_validation)
                )

                # JSON Streaming
                if self.log_type == "json":
                    self._stream_json({"step": step, "layer": layer_name, "metric": metric_name, "value": val_f, "is_validation": is_validation})
                
                # CSV Streaming
                if self.log_type == "csv":
                    with open(self.csv_path, 'a') as f:
                        f.write(f"{step},{layer_name},{metric_name},{val_f},{is_validation},{timestamp}\n")
        
        self.conn.commit()

    def _stream_json(self, entry: Dict):
        """Append an entry to the JSON file by reading and rewriting (basic implementation)."""
        # try:
        #     with open(self.json_path, 'r+') as f:
        #         data = json.load(f)
        #         if "metrics" not in data: data["metrics"] = []
        #         data["metrics"].append(entry)
        #         f.seek(0)
        #         json.dump(data, f, indent=2)
        #         f.truncate()
        # except (FileNotFoundError, json.JSONDecodeError):
        #     with open(self.json_path, 'w') as f:
        #         json.dump({"experiment": self.experiment_name, "metrics": [entry]}, f)
        with open(self.json_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def get_history(self) -> Dict[str, Dict[str, list]]:
        return dict(self.history)

    def get_experiment_history(self, experiment_name: str = None) -> Dict[str, Dict[str, list]]:
        if experiment_name is None: experiment_name = self.experiment_name
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT step, layer_name, metric_name, value, is_validation
            FROM metrics m JOIN experiments e ON m.experiment_id = e.id
            WHERE e.name = ? ORDER BY step, layer_name, metric_name
        """, (experiment_name,))
        history = defaultdict(lambda: defaultdict(list))
        for step, ln, mn, val, is_v in cursor.fetchall():
            prefix = "val_" if is_v else ""
            history[f"{prefix}{ln}"][mn].append((step, val))
        return dict(history)

    def get_pp_progression(self, experiment_name: str = None, layer_name: str = None) -> list:
        if experiment_name is None: experiment_name = self.experiment_name
        cursor = self.conn.cursor()
        if layer_name:
            cursor.execute("""
                SELECT step, value, is_validation FROM metrics m JOIN experiments e ON m.experiment_id = e.id
                WHERE e.name = ? AND layer_name = ? AND metric_name = 'patchiness' ORDER BY step
            """, (experiment_name, layer_name))
        else:
            cursor.execute("""
                SELECT step, AVG(value), is_validation FROM metrics m JOIN experiments e ON m.experiment_id = e.id
                WHERE e.name = ? AND metric_name = 'patchiness' GROUP BY step, is_validation ORDER BY step
            """, (experiment_name,))
        return cursor.fetchall()

    def list_experiments(self) -> list:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, created_at, metadata FROM experiments ORDER BY created_at DESC")
        return [{'name': n, 'created_at': c, 'metadata': json.loads(m)} for n, c, m in cursor.fetchall()]

    def export_experiment_data(self, experiment_name: str = None, output_path: str = None) -> str:
        if experiment_name is None: experiment_name = self.experiment_name
        if output_path is None: output_path = self.run_storage / f"{experiment_name}_export.json"
        data = {
            'experiment_name': experiment_name,
            'history': self.get_experiment_history(experiment_name),
            'pp_progression': self.get_pp_progression(experiment_name),
            'exported_at': datetime.now().isoformat()
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return str(output_path)


_global_storage = None

def get_storage(experiment_name: Optional[str] = None, log_type: str = "db") -> MetricStorage:
    """Get or create the global storage instance."""
    global _global_storage
    if _global_storage is None:
        _global_storage = MetricStorage(experiment_name, log_type)
    return _global_storage

        