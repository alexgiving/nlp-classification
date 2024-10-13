from pathlib import Path
import numpy as np
import pandas as pd
import random
import re

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
