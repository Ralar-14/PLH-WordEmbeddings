
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from typing import List, Tuple, Callable
import numpy as np
from gensim.utils import simple_preprocess
from gensim.models.fasttext import FastTextKeyedVectors



def map_tf_idf(sentence_preproc: List[str], embedding_model: FastTextKeyedVectors, dictionary: Dictionary, tf_idf_model: TfidfModel) -> Tuple[List[np.ndarray], List[float]]:
    bow = dictionary.doc2bow(sentence_preproc)
    tf_idf = tf_idf_model[bow]
    vectors, weights = [], []
    for word_index, weight in tf_idf:
        word = dictionary.get(word_index)
        if word in embedding_model:
            vectors.append(embedding_model[word])
            weights.append(weight)
    return vectors, weights

def map_pairs_w2v(
        sentence_pairs: List[Tuple[str, str, float]],
        embedding_model: FastTextKeyedVectors,
        dictionary: Dictionary = None,
        tf_idf_model: TfidfModel = None,
        preprocess: Callable[[str], List[str]] = simple_preprocess,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    # Mapeo de los pares de oraciones a pares de vectores
    pares_vectores = []
    for (sentence_1, sentence_2, similitud) in sentence_pairs:
        sentence_1_preproc = preprocess(sentence_1)
        sentence_2_preproc = preprocess(sentence_2)
        # Si usamos TF-IDF
        if tf_idf_model is not None:
            # Cálculo del promedio ponderado por TF-IDF de los word embeddings
            vectors1, weights1 = map_tf_idf(sentence_1_preproc, dictionary=dictionary, tf_idf_model=tf_idf_model, )
            vectors2, weights2 = map_tf_idf(sentence_2_preproc, dictionary=dictionary, tf_idf_model=tf_idf_model, )
            vector1 = np.average(vectors1, weights=weights1, axis=0, )
            vector2 = np.average(vectors2, weights=weights2, axis=0, )
        else:
            # Cálculo del promedio de los word embeddings
            vectors1 = [embedding_model[word] for word in sentence_1_preproc if word in embedding_model]
            vectors2 = [embedding_model[word] for word in sentence_2_preproc if word in embedding_model]
            vector1 = np.mean(vectors1, axis=0)
            vector2 = np.mean(vectors2, axis=0)
        # Añadir a la lista
        pares_vectores.append(((vector1, vector2), similitud))
    return pares_vectores