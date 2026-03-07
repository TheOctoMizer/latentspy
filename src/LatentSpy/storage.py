from collections import defaultdict

class MetricStorage:
    def __init__(self):
        self.history = defaultdict(lambda: defaultdict(list))
        self.step = 0
    
    def update(self,results):
        for layer_name, metrics in results.item():
            for metric_name, value in metrics.items():
                self.history[layer_name][metric_name].append(value)
        self.step += 1

    def get_history(self):
        return self.history

store = MetricStorage()

        