import tensorflow as tf
from typing import List, Tuple, Optional
import numpy as np
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

def map_word_embeddings(
        sentence: str,
        sequence_len: int,
        fixed_dictionary: Optional[Dictionary] = None,
        wv_model: Optional[tf.keras.layers.Embedding] = None
) -> np.ndarray:
    """
    Map to word-embedding indices
    :param sentence:
    :param sequence_len:
    :param fixed_dictionary:
    :return:
    """
    sentence_preproc = simple_preprocess(sentence)[:sequence_len]
    _vectors = np.zeros(sequence_len, dtype=np.int32)
    index = 0
    for word in sentence_preproc:
        if fixed_dictionary is not None:
            if word in fixed_dictionary.token2id:
                # Sumo 1 porque el valor 0 está reservado a padding
                _vectors[index] = fixed_dictionary.token2id[word] + 1
                index += 1
        else:
            if word in wv_model.key_to_index:
                _vectors[index] = wv_model.key_to_index[word] + 1
                index += 1
    return _vectors

def map_w2v_trainable(
        wv_model: tf.keras.layers.Embedding,
    sentence_pairs: List[Tuple[str, str, float]],
    sequence_len: int,
    fixed_dictionary: Optional[Dictionary] = None
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    '''
    Mapea los pares de oraciones a pares de vectores
    
    Parameters:
    - wv_model: Modelo de embeddings entrenable
    - sentence_pairs: Lista de pares de oraciones
    - sequence_len: Longitud de las secuencias
    - fixed_dictionary: Diccionario fijo de palabras
    
    Returns:
    - pares_vectores: Lista de pares de vectores
    '''
    # Mapeo de los pares de oraciones a pares de vectores
    pares_vectores = []
    for i, (sentence_1, sentence_2, similitud) in enumerate(sentence_pairs):
        vector1 = map_word_embeddings(sentence_1, sequence_len, fixed_dictionary, wv_model)
        vector2 = map_word_embeddings(sentence_2, sequence_len, fixed_dictionary, wv_model)
        # Añadir a la lista
        pares_vectores.append(((vector1, vector2), similitud))
    return pares_vectores

def pair_list_to_x_y(pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    '''
    Convierte una lista de pares de vectores a un par de arrays de vectores y un array de etiquetas
    
    Parameters:
    - pair_list: Lista de pares de vectores
    
    Returns:
    - x: Par de arrays de vectores
    - y: Array de etiquetas
    '''
    _x, _y = zip(*pair_list)
    _x_1, _x_2 = zip(*_x)
    return (np.row_stack(_x_1), np.row_stack(_x_2)), np.array(_y)

# Aquestes classes són necessàries per a poder fer servir les funcions de TensorFlow.

class MyLayer_mask(tf.keras.layers.Layer):
    def call(self, x):
        return tf.not_equal(x, 0)
    
class MyLayer_exp(tf.keras.layers.Layer):
    def call(self, x):
        return tf.exp(x)
    
class MyLayer_cast(tf.keras.layers.Layer):
    def call(self, x):
        return tf.cast(x, tf.float32)
    
class MyLayer_reduce_sum(tf.keras.layers.Layer):
    def call(self, x, _keepdims=True):
        return tf.reduce_sum(x, axis=1, keepdims=_keepdims)
    
def model_2(
    input_length: int,
    dictionary_size: int = 1000,
    embedding_size: int = 16,
    learning_rate: float = 1e-3,
    pretrained_weights: Optional[np.ndarray] = None,
    trainable: bool = False,
    use_cosine: bool = False,
) -> tf.keras.Model:
    # Inputs
    input_1 = tf.keras.Input((input_length,), dtype=tf.int32)
    input_2 = tf.keras.Input((input_length,), dtype=tf.int32)

    # Embedding Layer
    if pretrained_weights is None:
        embedding = tf.keras.layers.Embedding(
            dictionary_size, embedding_size, input_length=input_length, mask_zero=True
        )
    else:
        dictionary_size = pretrained_weights.shape[0]
        embedding_size = pretrained_weights.shape[1]
        initializer = tf.keras.initializers.Constant(pretrained_weights)
        embedding = tf.keras.layers.Embedding(
            dictionary_size,
            embedding_size,
            input_length=input_length,
            mask_zero=True,
            embeddings_initializer=initializer,
            trainable=trainable,
        )

    # Embed the inputs
    embedded_1 = embedding(input_1)
    embedded_2 = embedding(input_2)
    # Pass through the embedding layer
    _input_mask_1, _input_mask_2 = MyLayer_mask()(input_1), MyLayer_mask()(input_2)

    # Attention Mechanism
    attention_mlp = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    # Apply attention to each embedding
    attention_weights_1 = attention_mlp(embedded_1)  
    attention_weights_2 = attention_mlp(embedded_2) 
    # Mask the attention weights
    attention_weights_1 = MyLayer_exp()(attention_weights_1) * MyLayer_cast()(_input_mask_1[:, :, None])
    attention_weights_2 = MyLayer_exp()(attention_weights_2) * MyLayer_cast()(_input_mask_2[:, :, None])
    # Normalize attention weights
    attention_weights_1 = attention_weights_1 / MyLayer_reduce_sum()(attention_weights_1)
    attention_weights_2 = attention_weights_2 / MyLayer_reduce_sum()(attention_weights_2)
    # Compute context vectors
    projected_1 = MyLayer_reduce_sum()(embedded_1 * attention_weights_1, _keepdims=False) 
    projected_2 = MyLayer_reduce_sum()(embedded_2 * attention_weights_2, _keepdims=False) 
    
    if use_cosine:
        # Compute the cosine distance using a Lambda layer
        def cosine_distance(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))
        output = tf.keras.layers.Lambda(cosine_distance)([projected_1, projected_2])
    else:
         # Compute the cosine distance using a Lambda layer
        def normalized_product(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return x1_normalized * x2_normalized
    
        output = tf.keras.layers.Lambda(normalized_product)([projected_1, projected_2])
        output = tf.keras.layers.Dropout(0.1)(output)
        output = tf.keras.layers.Dense(
            16,
            activation="relu",
        )(output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )(output)
        
        output = tf.keras.layers.Lambda(lambda x: x * 5)(output)
    # Model Definition
    model = tf.keras.Model(inputs=(input_1, input_2), outputs=output)
    model.compile(
        loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model
