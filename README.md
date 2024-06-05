Executant el notebook.ipnyb es poden veure els resultats de l'execució dels diferents models i la comparació entre ells. En tot el procés utilitzem diferents funcions que no estan descrites en el notebook però que es poden trobar en els següents arxius definides amb docstrings:

importació_data.py:
    - read_ts_data
    - read_all_ts_data
    - reformat_data
    - create_corpus
    - stopwords_cat
    - preprocess
    - pair_list_to_x_y
    - flattened_corpus_count

model_bàsic.py:
    - build_and_compile_model_better
    - compute_pearson

onehot:
    - sentence_one_hot
    - map_one_hot

word2vec_tf_idf.py:
    - map_tf_idf
    - map_pairs_w2v

spacy_embed.py:
    - map_spacy_embed

roberta_base.py:
    - get_roberta_embedding
    - map_roberta_embed

roberta_fine_tuned.py:
    - prepare_roberta_ft
    - x_y_split_roberta_ft
    - compute_pearson_roberta_ft

trainable_embed_model:
    - map_word_embeddings
    - map_w2v_trainable
    - pair_list_to_x_y
    - model_2
