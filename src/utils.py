from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def get_subset(dataset: pd.DataFrame, n_samples: int, random_state: Optional[int] = None) -> pd.DataFrame:
    return dataset.sample(n=n_samples, random_state=random_state)


def dump_dict(dict: Dict[str, Any], indent: int = 2, precision: int = 2) -> str:
    string = ''
    for key, value in dict.items():

        if isinstance(value, np.ndarray):
            value = [round(val, precision) for val in value]
        elif isinstance(value, list):
            value = [round(val, precision) for val in value]
        elif isinstance(value, str):
            pass
        else:
            value = round(value, precision)

        string += ' '* indent + f'{key}: {value}\n'
    return string
