from .watch import watch
from .monitor import LatentMonitor
from .metrics import Metric
# from .hooks import register_hooks
# from .storage import MetricStorage

def serve(host: str = "0.0.0.0", port: int = 8000):
    """Start the LatentSpy dashboard server."""
    from .server import start_server
    start_server(host=host, port=port)