from gensim.corpora import Dictionary
from typing import List, Tuple, Callable
from importació_data import preprocess
import numpy as np

def sentence_one_hot(sentence: str, dict: Dictionary, keys:List = None, preprocess: Callable = preprocess
                     ) -> np.ndarray:
    '''
    Creates a one-hot encoding of a sentence using a gensim Dictionary

    Parameters:
    - sentence: str, sentence to be encoded
    - dict: Dictionary, gensim Dictionary object

    Returns:
    - one_hot: np.ndarray, one-hot encoding of the sentence
    '''
    preprocessed = preprocess(sentence)
    sentence_indexed = [dict.doc2idx([word])[0] for word in preprocessed]
    if keys is None:
        return np.array([1 if i in sentence_indexed else 0 for i in dict.keys()])
    else:
        return np.array([1 if i in sentence_indexed else 0 for i in keys])

def map_one_hot(sentence_pairs: List[Tuple[str, str, float]], dictionary:Dictionary, keys: List = None
                ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    '''
    Reformats a list of sentence pairs to a list of pairs of one-hot encoded sentences

    Parameters:
    - sentence_pairs: List[Tuple[str, str, float]], list of tuples with the training data
    - dictionary: Dictionary, gensim Dictionary object

    Returns:
    - pairs_vectors: List[Tuple[Tuple[np.ndarray, np.ndarray], float]], list of tuples with the one-hot encoded sentences and the similarity score
    '''
    pairs_vectors = []
    for (s1, s2, score) in sentence_pairs:
        s1_onehot = sentence_one_hot(s1, dictionary, keys)
        s2_onehot = sentence_one_hot(s2, dictionary, keys)
        pairs_vectors.append(((s1_onehot, s2_onehot), score))
    return pairs_vectors