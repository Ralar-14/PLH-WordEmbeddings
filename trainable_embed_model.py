import tensorflow as tf

def model_2(
    input_length: int = MAX_LEN,
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
    _input_mask_1, _input_mask_2 = tf.not_equal(input_1, 0), tf.not_equal(input_2, 0)

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
    attention_weights_1 = tf.exp(attention_weights_1) * tf.cast(_input_mask_1[:, :, None], tf.float32)
    attention_weights_2 = tf.exp(attention_weights_2) * tf.cast(_input_mask_2[:, :, None], tf.float32)
    # Normalize attention weights
    attention_weights_1 = attention_weights_1 / tf.reduce_sum(attention_weights_1, axis=1, keepdims=True)
    attention_weights_2 = attention_weights_2 / tf.reduce_sum(attention_weights_2, axis=1, keepdims=True)
    # Compute context vectors
    projected_1 = tf.reduce_sum(embedded_1 * attention_weights_1, axis=1) 
    projected_2 = tf.reduce_sum(embedded_2 * attention_weights_2, axis=1) 
    
    
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