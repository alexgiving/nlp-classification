from pathlib import Path
import numpy as np
import pandas as pd
import random
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def set_seed(seed_value=123):
    np.random.seed(seed_value)
    random.seed(seed_value)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    dataset_header = ['id', 'entity', 'sentiment', 'content']

    dataset = pd.read_csv(
        dataset_path,
        header=None,
        names=dataset_header
    )
    return dataset


def remove_stop_words(text: str) -> str:
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def remove_special_characters(text: str) -> str:
    clean_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return clean_text


def convert_to_lowercase(text: str) -> str:
    return text.lower()


def remove_extra_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


def removing_digits(text: str) -> str:
    return re.sub(r'\d', '', text.strip())


def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    result_text = []
    for word in text:
        result_text.append(lemmatizer.lemmatize(word))
    return ' '.join(result_text)


def stemming(text: str) -> str:
    stemmer = PorterStemmer()
    text = text.split()
    result_text = []
    for word in text:
        result_text.append(stemmer.stem(word))
    return ' '.join(result_text)


def normalization_text(type_normalization: str, text: str) -> str:
    if type_normalization == 'stemming':
        return stemming(text)
    elif type_normalization == 'lemmatization':
        return lemmatization(text)
    elif type_normalization == 'combo':
        text = lemmatization(text)
        return stemming(text)
    else:
        raise ValueError(f"Unsupported normalization type: {type_normalization}. Avaliable options: ['stemming', 'lemmatization', 'combo'(lemmatization + stemming)]")


def tokenize_word(text: str) -> str:
    return text.split()


def one_hot_vectorize(list_words: np.array) -> np.array:
    unique_words = list(set(list_words))
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    number_words = len(unique_words)

    one_hot_matrix = np.zeros((len(list_words), number_words))

    for i, word in enumerate(list_words):
        one_hot_matrix[i, word_to_index[word]] = 1

    return one_hot_matrix


def word2vec(corpus=list[list[str]], vector_size=100, window=5, min_count=1, workers=4) -> Word2Vec:
    return Word2Vec(sentences=corpus,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=workers
            )


def fasttext(corpus=list[list[str]], vector_size=100, window=5, min_count=1, workers=4) -> FastText:
    return FastText(sentences=corpus,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=workers
            )


def tf_idf(corpus: np.array) -> tuple[np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=19000)
    vector = vectorizer.fit_transform(corpus).toarray()
    return vector


def count_vectorizer(corpus: np.array) -> tuple[np.ndarray, CountVectorizer]:
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(corpus).toarray()
    return vector


def vectorization(type_vec: str, data: pd.Series):
    if type_vec == 'one_hot_vectorize':
        list_words = np.hstack([sentence.split() for sentence in data])
        return one_hot_vectorize(list_words)
    elif type_vec == 'word2vec':
        pass
    elif type_vec == 'fasttext':
        pass
    elif type_vec == 'tf_idf':
        pass
    else:
        return count_vectorizer()
