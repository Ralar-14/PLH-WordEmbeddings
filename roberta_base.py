import numpy as np
from typing import List, Tuple, Callable

def get_roberta_embedding(text: str, roberta_model: Callable) -> np.ndarray:
    return np.mean(roberta_model(text)._.trf_data.last_hidden_layer_state.data[:-1], axis=0)

def map_roberta_embed(
    sentence_pairs: List[Tuple[str, str, float]],
    roberta_model: Callable
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    # Mapeo de los pares de oraciones a pares de vectores
    pares_vectores = []
    for (sentence_1, sentence_2, similitud) in sentence_pairs:
        vector1 = get_roberta_embedding(sentence_1, roberta_model)
        vector2 = get_roberta_embedding(sentence_2, roberta_model)
        # Añadir a la lista
        pares_vectores.append(((vector1, vector2), similitud))
    return pares_vectores