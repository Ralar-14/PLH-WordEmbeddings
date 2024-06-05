from typing import Tuple, List, Callable
from scipy.stats import pearsonr
from scipy.special import logit

def prepare_roberta_ft(sentence_pairs: List[Tuple[str,str]], tokenizer: Callable):
    '''
    Prepare sentence pairs for RoBERTa fine-tuned model.
    
    Parameters:
    - sentence_pairs: List of tuples of strings. Each tuple represents a pair of sentences.
    - tokenizer: Tokenizer object from the transformers library.
    
    Returns:
    - sentence_pairs_prep: List of strings. Each string is a pair of sentences prepared for the model.
    '''

    sentence_pairs_prep = []
    for s1, s2 in sentence_pairs:
        sentence_pairs_prep.append(f"{tokenizer.cls_token} {s1}{tokenizer.sep_token}{tokenizer.sep_token} {s2}{tokenizer.sep_token}")
    return sentence_pairs_prep

def x_y_split_roberta_ft(data: List[Tuple[str, str, float]]) -> Tuple[List[Tuple[str, str]], List[float]]:
    '''
    Split the data into x and y.
    
    Parameters:
    - data: List of tuples of strings and floats. Each tuple represents a pair of sentences and a score.
    
    Returns:
    - x_: List of tuples of strings. Each tuple represents a pair of sentences.
    - y_: List of floats. Each float is a score.
    '''
    x_ = [(s1, s2) for s1, s2, _ in data]
    y_ = [score for _, _, score in data]
    return x_, y_

def compute_pearson_roberta_ft(predictions: List[dict['label':str, 'score':float]], y: List[float]) -> float:
    '''
    Compute the Pearson correlation between the predictions and the true scores.
    
    Parameters:
    - predictions: List of dictionaries. Each dictionary contains a label and a score.
    - y: List of floats. Each float is a score.
    
    Returns:
    - correlation: Float. Pearson correlation between the predictions and the true scores.
    '''
    y_pred = [logit(item['score']) for item in predictions]
    correlation, _ = pearsonr(y_pred, y)
    return correlation