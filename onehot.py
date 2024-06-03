from gensim.corpora import Dictionary
from typing import List, Tuple, Callable
from importació_data import preprocess
import numpy as np

def sentence_one_hot(dict: Dictionary, sentence: str, preprocess: Callable = preprocess) -> np.ndarray:
    preprocessed = preprocess(sentence)
    sentence_indexed = [dict.doc2idx([word])[0] for word in preprocessed]
    return np.array([1 if i in sentence_indexed else 0 for i in range(len(dict))])

def map_one_hot(sentence_pairs: List[Tuple[str, str, float]], dictionary:Dictionary = None) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    if dictionary is None:
            raise ValueError("Dictionary is required for one-hot encoding")
    pairs_vectors = []
    for (s1, s2, score) in sentence_pairs:
        s1_onehot = sentence_one_hot(dictionary, s1)
        s2_onehot = sentence_one_hot(dictionary, s2)
        pairs_vectors.append(((s1_onehot, s2_onehot), score))
    return pairs_vectors