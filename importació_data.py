import pandas as pd
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from typing import List, Tuple, Callable
import numpy as np

def read_ts_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep = '\t')
    return data

def read_all_ts_data(train_path: str = 'data_ts\sts_cat_train_v1.tsv', 
                  test_path: str = 'data_ts\sts_cat_test_v1.tsv', 
                  val_path: str = 'data_ts\sts_cat_dev_v1.tsv') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = read_ts_data(train_path)
    test = read_ts_data(test_path)
    val = read_ts_data(val_path)

    return train, test, val

def reformat_data(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    input_pairs = [(train["sentence1"][i], train["sentence2"][i], train["avg"][i]) for i in range(len(train))]
    input_pairs_val = [(val["sentence1"][i], val["sentence2"][i], val["avg"][i]) for i in range(len(val))]
    input_pairs_test = [(test["sentence1"][i], test["sentence2"][i], test["avg"][i]) for i in range(len(test))]
    return input_pairs, input_pairs_val, input_pairs_test

def create_corpus(input_pairs: List[Tuple[str, str, float]], input_pairs_val: List[Tuple[str, str, float]], input_pairs_test: List[Tuple[str, str, float]], preprocess: Callable = simple_preprocess) -> List[List[Tuple[int, int]]]:
    all_input_pairs = input_pairs + input_pairs_val + input_pairs_test
    # Preprocesamiento de las oraciones y creación del diccionario
    sentences_1_preproc = [preprocess(sentence_1) for sentence_1, _, _ in all_input_pairs]
    sentences_2_preproc = [preprocess(sentence_2) for _, sentence_2, _ in all_input_pairs]

    sentences_pairs_flattened = sentences_1_preproc + sentences_2_preproc
    diccionario = Dictionary(sentences_pairs_flattened)
    corpus = [diccionario.doc2bow(sent) for sent in sentences_pairs_flattened]
    return corpus, diccionario

def stopwords_cat(path_stopwords: str = 'data_ts/stopwords_cat.txt') -> set:
    stopwords = set()
    with open(path_stopwords, 'r') as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords

def preprocess(sentence: str, stopwords: set = stopwords_cat()) -> List[str]:
    preprocessed = simple_preprocess(sentence)
    preprocessed = [token for token in preprocessed if token not in stopwords]
    return preprocessed

def pair_list_to_x_y(pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    _x, _y = zip(*pair_list)
    _x_1, _x_2 = zip(*_x)
    return (np.row_stack(_x_1), np.row_stack(_x_2)), np.array(_y)