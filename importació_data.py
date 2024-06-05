import pandas as pd
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from typing import List, Tuple, Callable
import numpy as np

def read_ts_data(path: str
                 ) -> pd.DataFrame:
    '''
    Reads a file with tab-separated values and returns a dataframe with the data

    Parameters:
    - path: str, path to the file to be read

    Returns:
    - data: pd.DataFrame, dataframe with the data read from the file
    '''
    data = pd.read_csv(path, sep = '\t')
    return data

def read_all_ts_data(train_path: str = 'data_ts\sts_cat_train_v1.tsv', test_path: str = 'data_ts\sts_cat_test_v1.tsv', val_path: str = 'data_ts\sts_cat_dev_v1.tsv'
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Reads the training, testing and validation files and returns the dataframes with the data

    Parameters:
    - train_path: str, path to the training file
    - test_path: str, path to the testing file
    - val_path: str, path to the validation file

    Returns:
    - train: pd.DataFrame, dataframe with the training data
    - test: pd.DataFrame, dataframe with the testing data
    - val: pd.DataFrame, dataframe with the validation data
    '''
    train = read_ts_data(train_path)
    test = read_ts_data(test_path)
    val = read_ts_data(val_path)

    return train, test, val

def reformat_data(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame
                  ) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    '''
    Reformats the data from the dataframes to a list of tuples in order to be used in the model

    Parameters:
    - train: pd.DataFrame, dataframe with the training data
    - test: pd.DataFrame, dataframe with the testing data
    - val: pd.DataFrame, dataframe with the validation data

    Returns:
    - input_pairs_train: List[Tuple[str, str, float]], list of tuples with the training data
    - input_pairs_val: List[Tuple[str, str, float]], list of tuples with the validation data
    - input_pairs_test: List[Tuple[str, str, float]], list of tuples with the testing data
    '''
    input_pairs_train = [(train["sentence1"][i], train["sentence2"][i], train["avg"][i]) for i in range(len(train))]
    input_pairs_val = [(val["sentence1"][i], val["sentence2"][i], val["avg"][i]) for i in range(len(val))]
    input_pairs_test = [(test["sentence1"][i], test["sentence2"][i], test["avg"][i]) for i in range(len(test))]
    return input_pairs_train, input_pairs_val, input_pairs_test

def create_corpus(input_pairs: List[Tuple[str, str, float]], input_pairs_val: List[Tuple[str, str, float]], input_pairs_test: List[Tuple[str, str, float]], preprocess: Callable = simple_preprocess
                  ) -> Tuple[List[List[Tuple[int, int]]], Dictionary]:
    '''	
    Creates the corpus and the dictionary from the input pairs of sentences

    Parameters:
    - input_pairs: List[Tuple[str, str, float]], list of tuples with the training data
    - input_pairs_val: List[Tuple[str, str, float]], list of tuples with the validation data
    - input_pairs_test: List[Tuple[str, str, float]], list of tuples with the testing data
    - preprocess: Callable, function to preprocess the sentences

    Returns:
    - corpus: List[List[Tuple[int, int]]], list of lists with the corpus
    - diccionario: Dictionary, dictionary with the words of the corpus
    '''
    all_input_pairs = input_pairs + input_pairs_val + input_pairs_test
    # Preprocesamiento de las oraciones y creación del diccionario
    sentences_1_preproc = [preprocess(sentence_1) for sentence_1, _, _ in all_input_pairs]
    sentences_2_preproc = [preprocess(sentence_2) for _, sentence_2, _ in all_input_pairs]

    sentences_pairs_flattened = sentences_1_preproc + sentences_2_preproc
    diccionario = Dictionary(sentences_pairs_flattened)
    corpus = [diccionario.doc2bow(sent) for sent in sentences_pairs_flattened]
    return corpus, diccionario

def stopwords_cat(path_stopwords: str = 'data_ts/stopwords_cat.txt'
                  ) -> set:
    '''
    Reads the stopwords from a file and returns a set with them

    Parameters:
    - path_stopwords: str, path to the file with the stopwords

    Returns:
    - stopwords: set, set with the stopwords
    '''

    stopwords = set()
    with open(path_stopwords, 'r', encoding = 'utf-8') as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords

def preprocess(sentence: str, stopwords: set = stopwords_cat()
               ) -> List[str]:
    '''
    Preprocesses a sentence by removing the stopwords

    Parameters:
    - sentence: str, sentence to be preprocessed
    - stopwords: set, set with the stopwords

    Returns:
    - preprocessed: List[str], list with the preprocessed words
    '''
    preprocessed = simple_preprocess(sentence)
    preprocessed = [token for token in preprocessed if token not in stopwords]
    return preprocessed

def pair_list_to_x_y(pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    '''
    Transforms a list of pairs of vectors and labels to a tuple with the vectors and a numpy array with the labels

    Parameters:
    - pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]], list of tuples with the pairs of vectors and labels

    Returns:
    - x_: Tuple[np.ndarray, np.ndarray], tuple with the vectors
    - y_: np.ndarray, numpy array with the labels
    '''
    _x, _y = zip(*pair_list)
    _x_1, _x_2 = zip(*_x)
    return (np.row_stack(_x_1), np.row_stack(_x_2)), np.array(_y)

def flattened_corpus_count(corpus: List[List[Tuple[int, int]]]
                           ) -> dict:
    '''
    Flattens the corpus and counts the number of times each word appears

    Parameters:
    - corpus: List[List[Tuple[int, int]]], list of lists with the corpus

    Returns:
    - corpus_aplanado: dict, dictionary with the words and the number of times they appear
    '''
    corpus_aplanado = {}
    for sentence in corpus:
        for word in sentence:
            if word[0] not in corpus_aplanado:
                corpus_aplanado[word[0]] = word[1]
            else:
                corpus_aplanado[word[0]] += word[1]
    return corpus_aplanado