import numpy as np
from sklearn.metrics import confusion_matrix

from modules.utils import metrics2


class ClassificationEvaluator():
    def __init__(self, num_class):
        self.registed_metrics = {}
        # self.register_new_metrics({'Precision': metrics2.precision,
        #                            'Recall': metrics2.recall})
        self.total_evaluation = []
        self.num_class = num_class
        # for m in self.metrics:
        #     if m not in self.registed_metrics:
        #         raise NotImplementedError('The assign evaluate function is not exist.')

    def register_new_metrics(self, new_metrics):
        # TODO: some check maybe
        if not isinstance(new_metrics, dict):
            raise TypeError(f'The new_metrics data type need to be list[dict] not {isinstance(new_metrics)}')
        self.registed_metrics.update(new_metrics)

    def evaluate(self, traget, pred, cm=None):
        # TODO: check inputs? (value? dimension?)
        # TODO: evaluation name?
        # TODO: flatten?
        # TODO: input type, flexible way?
        if cm is None:
            cm = confusion_matrix(traget, pred, labels=np.arange(0, self.num_class))
        one_evaluation = {}
        for m in self.registed_metrics:
            one_evaluation[m] = self.registed_metrics[m](cm)
        self.total_evaluation.append(one_evaluation)
        return one_evaluation

    def get_aggregation(self, agg_func):
        total_aggregation = {}
        for idx, one_eval in enumerate(self.total_evaluation):
            if idx == 0:
                total_aggregation = {m: np.array([], dtype=np.float32) for m in one_eval}
            for metrics in one_eval:
                total_aggregation[metrics] = np.append(total_aggregation[metrics], one_eval[metrics])

        total_aggregation = {metrics: agg_func(total_aggregation[metrics]) for metrics in total_aggregation}
        return total_aggregation