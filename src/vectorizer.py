from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class VectorizerType(Enum):
    ONE_HOT = 'one_hot'
    W2V = 'w2v'
    TF_IDF = 'tf_idf'


class Vectorizer:
    def __init__(self, vectorizer_type: VectorizerType) -> None:
        self._vectorizer_type = vectorizer_type

        self._vectorizer_map = {
            VectorizerType.ONE_HOT: CountVectorizer,
            VectorizerType.W2V: Word2Vec,
            VectorizerType.TF_IDF: TfidfVectorizer,
        }
        self._vectorizer = None

    def _init_vectorizer(self, *args, **kwargs) -> None:
        self._vectorizer_configuration_map = {
            VectorizerType.ONE_HOT: {
                'binary': True
            },
            VectorizerType.W2V: {
                'sentences': kwargs.get('corpus'),
                'vector_size': 100,
                'window': 5,
                'min_count': 1,
                'workers': 4
            },
            VectorizerType.TF_IDF: {
                'ngram_range': (1, 2),
                'norm': None,
                'smooth_idf': False,
                'stop_words': 'english',
                'use_idf': False
            },
        }

        vectorizer_class = self._vectorizer_map[self._vectorizer_type]
        vectorizer_parameters = self._vectorizer_configuration_map[self._vectorizer_type]

        self._vectorizer = vectorizer_class(**vectorizer_parameters)

    @staticmethod
    def _tokenize(x: pd.Series) -> List[List[str]]:
        return [sentence.split() for sentence in x]

    def fit(self, x: pd.Series) -> None:
        if self._vectorizer_type is VectorizerType.W2V and not self._vectorizer:
            self._init_vectorizer(corpus=self._tokenize(x))
        elif not self._vectorizer:
            self._init_vectorizer()

        if self._vectorizer_type is not VectorizerType.W2V:
            self._vectorizer.fit(x)

    @staticmethod
    def _get_vectors_word2vec(tokens_list: List[str], vector: KeyedVectors, vector_size: int) -> List[List[float]]:
        return [vector[word] if word in vector else np.zeros(vector_size) for word in tokens_list]

    @staticmethod
    def _get_average_vectors_word2vec(vectors: List[List[float]]) -> List[float]:
        return np.divide(np.sum(vectors, axis=0), len(vectors))

    def transform(self, x: pd.Series) -> np.typing.NDArray[np.float32]:
        if self._vectorizer_type is VectorizerType.W2V:
            vector_size = self._vectorizer_configuration_map[VectorizerType.W2V]['vector_size']
            X_train_word2vec = [self._get_vectors_word2vec(sentence, self._vectorizer.wv, vector_size) for sentence in self._tokenize(x)]
            return X_train_word2vec

        return self._vectorizer.transform(x).toarray()