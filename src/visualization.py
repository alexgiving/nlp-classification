import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

def distribution_chart(data: pd.Series, column_name: str) -> plt.Figure:
    fig, ax = plt.subplots()

    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    ax.barh(
        [name for name, _ in data.items()],
        data.values,
        color = bar_colors
        )

    ax.set_ylabel(f'Number of {column_name}')
    ax.set_title(f'Distribution of {column_name}')
    return fig

def sentiment_distribution_chart(data: pd.DataFrame, entity_col: str, sentiment_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 15))

    sns.countplot(
        data=data,
        y=entity_col,
        hue=sentiment_col,
        order=data[entity_col].value_counts().index,
        palette='viridis',
        ax=ax
    )

    ax.set_title('Sentiment Distribution by Entity')
    ax.set_xlabel('Count')
    ax.set_ylabel('Entity')
    ax.legend(title='Sentiment')

    plt.tight_layout()

    return fig

def generate_sentiment_wordclouds(data: pd.DataFrame,
                                  sentiment_col: str,
                                  text_col: str,
                                  sentiments:list,
                                  background_color: str = 'white',
                                  width: int = 400,
                                  height: int = 200
) -> plt.Figure:

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for sentiment, ax in zip(sentiments, axs.ravel()):
        sentiment_text = " ".join(tweet for tweet in data[data[sentiment_col] == sentiment][text_col])
        wordcloud = WordCloud(background_color=background_color, width=width, height=height).generate(sentiment_text)

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud for {sentiment} Sentiment')

    plt.tight_layout()
    return fig
