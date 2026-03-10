from collections import defaultdict

class MetricStorage:
    def __init__(self):
        self.history = defaultdict(lambda: defaultdict(list))
    def update(self, results, step: int):
        for layer_name, metrics in results.items():
            for metric_name, value in metrics.items():
                self.history[layer_name][metric_name].append((step, value))

    def get_history(self):
        return self.history

store = MetricStorage()

        