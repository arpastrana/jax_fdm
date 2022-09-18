from compas.data import Data


# ==========================================================================
# Recorder
# ==========================================================================

class OptimizationRecorder(Data):
    def __init__(self):
        self.history = []

    def record(self, value):
        self.history.append(value)

    def __call__(self, q, *args, **kwargs):
        self.record(q)

    @property
    def data(self):
        data = dict()
        data["history"] = self.history
        return data

    @data.setter
    def data(self, data):
        self.history = data["history"]
