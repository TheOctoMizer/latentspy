from collections import defaultdict
from pathlib import Path
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional


class MetricStorage:
    def __init__(self, experiment_name: Optional[str] = None):
        self.history = defaultdict(lambda: defaultdict(list))
        self.experiment_name = experiment_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_storage: Path = Path(".") / ".latentspy" / "runs"
        self.run_storage.mkdir(parents=True, exist_ok=True)
        self.run_database: Path = self.run_storage / "runs.db"
        self.run_database.touch(exist_ok=True)
        self.conn = sqlite3.connect(self.run_database)
        
        # Initialize database tables
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
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_experiment_step ON metrics(experiment_id, step)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_layer_metric ON metrics(layer_name, metric_name)")
        
        self.conn.commit()
        
        self._register_experiment()

    def _register_experiment(self):
        """Register the current experiment in the database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO experiments (name, metadata) VALUES (?, ?)",
            (self.experiment_name, json.dumps({"created_at": datetime.now().isoformat()}))
        )
        self.conn.commit()

    def update(self, results: Dict[str, Dict[str, Any]], step: int, is_validation: bool = False):
        """Update storage with new metrics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT id FROM experiments WHERE name = ?", (self.experiment_name,))
        experiment_row = cursor.fetchone()
        if not experiment_row:
            self._register_experiment()
            cursor.execute("SELECT id FROM experiments WHERE name = ?", (self.experiment_name,))
            experiment_row = cursor.fetchone()
        
        experiment_id = experiment_row[0]
        
        for layer_name, metrics in results.items():
            for metric_name, value in metrics.items():
                self.history[layer_name][metric_name].append((step, value))
                
                cursor.execute(
                    """
                    INSERT INTO metrics 
                    (experiment_id, step, layer_name, metric_name, value, is_validation) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (experiment_id, step, layer_name, metric_name, float(value), is_validation)
                )
        
        self.conn.commit()

    def get_history(self) -> Dict[str, Dict[str, list]]:
        """Get the in-memory history."""
        return dict(self.history)

    def get_experiment_history(self, experiment_name: str = None) -> Dict[str, Dict[str, list]]:
        """Get full history for a specific experiment from database."""
        if experiment_name is None:
            experiment_name = self.experiment_name
            
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT step, layer_name, metric_name, value, is_validation
            FROM metrics m
            JOIN experiments e ON m.experiment_id = e.id
            WHERE e.name = ?
            ORDER BY step, layer_name, metric_name
        """, (experiment_name,))
        
        history = defaultdict(lambda: defaultdict(list))
        for step, layer_name, metric_name, value, is_validation in cursor.fetchall():
            prefix = "val_" if is_validation else ""
            history[f"{prefix}{layer_name}"][metric_name].append((step, value))
        
        return dict(history)

    def get_pp_progression(self, experiment_name: str = None, layer_name: str = None) -> list:
        """Get PP progression over time for analysis."""
        if experiment_name is None:
            experiment_name = self.experiment_name
            
        cursor = self.conn.cursor()
        
        if layer_name:
            cursor.execute("""
                SELECT step, value, is_validation
                FROM metrics m
                JOIN experiments e ON m.experiment_id = e.id
                WHERE e.name = ? AND layer_name = ? AND metric_name = 'patchiness'
                ORDER BY step
            """, (experiment_name, layer_name))
        else:
            cursor.execute("""
                SELECT step, AVG(value), is_validation
                FROM metrics m
                JOIN experiments e ON m.experiment_id = e.id
                WHERE e.name = ? AND metric_name = 'patchiness'
                GROUP BY step, is_validation
                ORDER BY step
            """, (experiment_name,))
        
        return cursor.fetchall()

    def list_experiments(self) -> list:
        """List all experiments in the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, created_at, metadata
            FROM experiments
            ORDER BY created_at DESC
        """)
        
        experiments = []
        for name, created_at, metadata in cursor.fetchall():
            experiments.append({
                'name': name,
                'created_at': created_at,
                'metadata': json.loads(metadata) if metadata else {}
            })
        
        return experiments

    def compare_experiments(self, experiment_names: list) -> Dict[str, Dict]:
        """Compare PP progression across multiple experiments."""
        comparison = {}
        
        for exp_name in experiment_names:
            pp_data = self.get_pp_progression(exp_name)
            comparison[exp_name] = {
                'pp_data': pp_data,
                'final_pp': pp_data[-1][1] if pp_data else None,
                'pp_change': pp_data[-1][1] - pp_data[0][1] if len(pp_data) > 1 else 0
            }
        
        return comparison

    def export_experiment_data(self, experiment_name: str = None, output_path: str = None) -> str:
        """Export experiment data to JSON file."""
        if experiment_name is None:
            experiment_name = self.experiment_name
            
        if output_path is None:
            output_path = self.run_storage / f"{experiment_name}_export.json"
        
        data = {
            'experiment_name': experiment_name,
            'history': self.get_experiment_history(experiment_name),
            'pp_progression': self.get_pp_progression(experiment_name),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(output_path)

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


store = MetricStorage()

        