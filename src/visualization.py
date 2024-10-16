import matplotlib.pyplot as plt
import pandas as pd


def distribution_chart(data: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots()

    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    ax.bar(
        [name for name, _ in data.items()],
        data.values,
        color = bar_colors
        )

    ax.set_ylabel('Number of samples')
    ax.set_title('Distribution of samples')
    return fig
