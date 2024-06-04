import tensorflow as tf
from scipy.stats import pearsonr
import numpy as np
from typing import Tuple

def build_and_compile_model_better(embedding_size: int = 300, learning_rate: float = 1e-3) -> tf.keras.Model:
    # Capa de entrada para los pares de vectores
    input_1 = tf.keras.Input(shape=(embedding_size,))
    input_2 = tf.keras.Input(shape=(embedding_size,))

    # Hidden layer
    first_projection = tf.keras.layers.Dense(
        embedding_size,
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    projected_1 =  first_projection(input_1)
    projected_2 = first_projection(input_2)
    
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
    
    # Define output
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model

def compute_pearson(model: tf.keras.Model, x_: Tuple[np.ndarray, np.ndarray], y_: np.ndarray) -> float:
    # Obtener las predicciones del modelo para los datos de prueba. En este ejemplo vamos a utilizar el corpus de training.
    y_pred = model.predict(x_)
    # Calcular la correlación de Pearson entre las predicciones y los datos de prueba
    correlation, _ = pearsonr(y_pred.flatten(), y_.flatten())
    return correlation
