from gensim.corpora import Dictionary
from typing import List, Tuple, Callable
from importació_data import preprocess

def sentence_one_hot(dict: Dictionary, sentence: str, preprocess: Callable = preprocess) -> List[Tuple[int, int]]:
    preprocessed = preprocess(sentence)
    sentence_indexed = dict.doc2bow(preprocessed)
    return [0 if word not in sentence_indexed else 1 for word in range(len(dict))]

