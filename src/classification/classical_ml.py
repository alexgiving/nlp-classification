from enum import Enum

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassicClassificationType(Enum):
    SVC = 'svc',
    DECISION_TREE = 'decision_tree'
    RANDOM_FOREST = 'random_forest'
    K_NEIGHBORS = 'k_neighbors'


class ClassicClassification:
    def __init__(self, classification_type: ClassicClassificationType) -> None:
        self._classification_type = classification_type

        self._classification_map = {
            ClassicClassificationType.SVC: SVC,
            ClassicClassificationType.DECISION_TREE: DecisionTreeClassifier,
            ClassicClassificationType.RANDOM_FOREST: RandomForestClassifier,
            ClassicClassificationType.K_NEIGHBORS: KNeighborsClassifier,
        }
        self._classification = None

    def _init_classification(self, *args, **kwargs) -> None:
        self._classification_configuration_map = {
            ClassicClassificationType.SVC: {
                'gamma': 2,
                'C': 1,
                'random_state': 42,
            },
            ClassicClassificationType.DECISION_TREE: {
                'max_depth': 5,
                'random_state': 42,
            },
            ClassicClassificationType.RANDOM_FOREST: {
                'max_depth': 5,
                'n_estimators': 10,
                'max_features': 1,
                'random_state': 42,
            },
            ClassicClassificationType.K_NEIGHBORS: {
                'n_neighbors': 3,
            },
        }

        classification_class = self._classification_map[self._classification_type]
        classification_parameters = self._classification_configuration_map[self._classification_type]

        self._classification = classification_class(**classification_parameters)

    def fit(self, x: pd.Series, y: pd.Series) -> None:
            if not self._classification:
                self._init_classification()
            self._classification.fit(x, y)

    def predict(self, x_test: pd.Series) -> pd.Series:
        return self._classification.predict(x_test)

