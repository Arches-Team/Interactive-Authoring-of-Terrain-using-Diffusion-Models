
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support


class Metric:

    def reset(self):
        raise NotImplementedError

    def add(self, outputs, targets):
        # outputs and targets (in batches)
        raise NotImplementedError

    def total(self):
        # Return dictionary of metrics
        raise NotImplementedError


class ClassificationMetrics(Metric):

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_targets = []

    def add(self, outputs, targets):
        _, predictions = torch.max(outputs, dim=1)

        for prediction, target in zip(predictions, targets):
            self.all_predictions.append(prediction.item())
            self.all_targets.append(target.item())

        # TODO return something for live updates?

    def total(self):
        data = dict(
            y_true=self.all_targets,
            y_pred=self.all_predictions
        )

        accuracy = accuracy_score(**data)
        balanced_accuracy = balanced_accuracy_score(**data)
        prec, rec, f1, _ = precision_recall_fscore_support(
            **data, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        }
