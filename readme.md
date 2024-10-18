# NLP Classification

Deadline: `October 8 2024`

## Install dependencies
Python 3.11
>Note Python <= 3.11 is required

```bash
pip install -r requirements.txt
```

## Download dataset

In this work the [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data) dataset is used.

```bash
mkdir -p data/
pushd data/
    kaggle datasets download -d jp797498e/twitter-entity-sentiment-analysis/
    unzip twitter-entity-sentiment-analysis.zip
    rm -rf twitter-entity-sentiment-analysis.zip
popd
```



1. EDA
2. 3 ML, 2 NN
3. Improvement
4. Vectorization analysis
