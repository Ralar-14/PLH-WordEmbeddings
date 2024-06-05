import numpy as np
from typing import List, Tuple, Callable

def get_roberta_embedding(text: str, roberta_model: Callable) -> np.ndarray:
    '''
    Get the RoBERTa embedding of a text.
    
    Parameters:
    - text: String. Text to embed.
    
    Returns:
    - embedding: Numpy array. RoBERTa embedding of the text.
    '''
    
    return np.mean(roberta_model(text)._.trf_data.last_hidden_layer_state.data[:-1], axis=0)

def map_roberta_embed(
    sentence_pairs: List[Tuple[str, str, float]],
    roberta_model: Callable
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    '''
    Map sentence pairs to vector pairs.
    
    Parameters:
    - sentence_pairs: List of tuples of strings and floats. Each tuple represents a pair of sentences and a similarity score.
    - roberta_model: Callable. RoBERTa model from the transformers library.
    
    Returns:
    - pares_vectores: List of tuples of tuples of numpy arrays and floats. Each tuple represents a pair of vectors and a similarity score.'''
    pares_vectores = []
    for (sentence_1, sentence_2, similitud) in sentence_pairs:
        vector1 = get_roberta_embedding(sentence_1, roberta_model)
        vector2 = get_roberta_embedding(sentence_2, roberta_model)
        pares_vectores.append(((vector1, vector2), similitud))
    return pares_vectores