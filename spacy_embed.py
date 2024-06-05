from typing import List, Tuple, Callable
import numpy as np


def map_spacy_embed(sentence_pairs: List[Tuple[str, str, float]], model_spacy: Callable
                ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    '''
    Reformats a list of sentence pairs to a list of pairs of spacy embeddings

    Parameters:
    - sentence_pairs: List[Tuple[str, str, float]], list of tuples with the training data
    - model_spacy: Callable, spacy model object

    Returns:
    - pairs_vectors: List[Tuple[Tuple[np.ndarray, np.ndarray], float]], list of tuples with the spacy embeddings and the similarity score
    '''
    pairs_vectors = []
    for (s1, s2, score) in sentence_pairs:
        s1_embed = model_spacy(s1).vector
        s2_embed = model_spacy(s2).vector
        pairs_vectors.append(((s1_embed, s2_embed), score))
    return pairs_vectors