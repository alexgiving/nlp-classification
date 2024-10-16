from pathlib import Path

import pandas as pd


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset_header = ['id', 'entity', 'sentiment', 'content']

    dataset = pd.read_csv(
        dataset_path,
        header=None,
        names=dataset_header
    )
    return dataset
