from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def estimate_quality(y_test: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    result = {}
    result['accuracy'] = accuracy_score(y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred)
    result['per_class_accuracy'] = matrix.diagonal()/matrix.sum(axis=1)

    result['f1'] = f1_score(y_test, y_pred, average = 'micro')
    result['per_class_f1'] = f1_score(y_test, y_pred, average = None)
    return result

