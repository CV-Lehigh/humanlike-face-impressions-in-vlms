import os
import json
import pandas as pd
import math

orig_scores_path = f'subjective_attribute_datasets/original_scores'
file = 'omi__original_scores.csv' # cfd__original_scores__Likert_1_to_7.csv, uaf__original_scores__Likert_1_to_9.csv, omi__original_scores.csv

def cfd_normalize(score):
    return score/7

def uaf_normalize(score):
    return score/9

def omi_normalize(score):
    return score/100

def reorder_tuples(tuples_list, order_list):
    """
    Reorders a list of tuples based on the order of the first element 
    in another list.

    Args:
        tuples_list: A list of tuples.
        order_list: A list specifying the desired order of the first 
          elements of the tuples.

    Returns:
        A new list of tuples, reordered according to order_list.
    """
    value_index_map = {value: index for index, value in enumerate(order_list)}
    return sorted(tuples_list, key=lambda x: value_index_map.get(x[0], float('inf')))

def rank_labels(labels, scores):
    """Ranks labels based on their scores in descending order.

    Args:
        labels: A list of labels.
        scores: A list of scores corresponding to the labels.

    Returns:
        A list of tuples, where each tuple contains a label and its rank,
        sorted by score in descending order.
    """

    # Create a list of (label, score) pairs
    label_score_pairs = list(zip(labels, scores))

    # Sort the pairs based on scores in descending order
    ranked_labels = sorted(label_score_pairs, key=lambda x: x[1], reverse=True)
    # Add rank to each label
    ranked_labels_with_rank = [(label, rank + 1) for rank, (label, _) in enumerate(ranked_labels)]
    
    reordered_tuples = reorder_tuples(ranked_labels_with_rank, labels)
    ranks = [tuple[1] for tuple in reordered_tuples]
    
    return ranks


orig_scores = pd.read_csv(f'{orig_scores_path}/{file}')
scores_attributes = []

if file == 'cfd__original_scores__Likert_1_to_7.csv':
    labels = ['attractive', 'feminine', 'masculine', 'happy', 'sad', 'trustworthy', 'dominance']
    for index, row in orig_scores.iterrows():
        if not math.isnan(row['Dominant']):
            image_name = row['Model']
            attractive = cfd_normalize(row['Attractive'])
            feminine = cfd_normalize(row['Feminine'])
            masculine = cfd_normalize(row['Masculine'])
            happy = cfd_normalize(row['Happy'])
            sad = cfd_normalize(row['Sad'])
            trustworthy = cfd_normalize(row['Trustworthy'])
            dominance = cfd_normalize(row['Dominant'])
            scores = [attractive, feminine, masculine, happy, sad, trustworthy, dominance]
            ranks = rank_labels(labels, scores)
            scores_ranks = {'image_name': image_name, 'attractive': (scores[0], ranks[0]), 'feminine': (scores[1], ranks[1]), 'masculine': (scores[2], ranks[2]), 'happy': (scores[3], ranks[3]), 'sad': (scores[4], ranks[4]), 'trustworthy': (scores[5], ranks[5]), 'dominance': (scores[6], ranks[6])}
            scores_attributes.append(scores_ranks)
    with open(f"{orig_scores_path}/cfd__norm_0_1__ranks.json", "w") as f:
        json.dump(scores_attributes, f)
        
elif file == 'omi__original_scores.csv':
    labels = ['attractive', 'happy', 'trustworthy', 'dominance']
    for index, row in orig_scores.iterrows():
        image_name = int(row['stimulus'])
        attractive = omi_normalize(row['attractive'])
        happy = omi_normalize(row['happy'])
        trustworthy = omi_normalize(row['trustworthy'])
        dominance = omi_normalize(row['dominant'])
        scores = [attractive, happy, trustworthy, dominance]
        ranks = rank_labels(labels, scores)
        scores_ranks = {'image_name': image_name, 'attractive': (scores[0], ranks[0]), 'happy': (scores[1], ranks[1]), 'trustworthy': (scores[2], ranks[2]), 'dominance': (scores[3], ranks[3])}
        scores_attributes.append(scores_ranks)
    with open(f"{orig_scores_path}/omi__norm_0_1__ranks.json", "w") as f:
        json.dump(scores_attributes, f)
        
elif file == 'uaf__original_scores__Likert_1_to_9.csv':
    labels = ['attractive', 'unattractive', 'happy', 'unhappy', 'trustworthy', 'untrustworthy']
    for index, row in orig_scores.iterrows():
        image_name = row['Filename']
        attractive = uaf_normalize(row['attractive'])
        unattractive = uaf_normalize(row['unattractive'])
        happy = uaf_normalize(row['happy'])
        unhappy = uaf_normalize(row['unhappy'])
        trustworthy = uaf_normalize(row['trustworthy'])
        untrustworthy = uaf_normalize(row['untrustworthy'])
        scores = [attractive, unattractive, happy, unhappy, trustworthy, untrustworthy]
        scores_ranks = {'image_name': image_name, 'attractive': scores[0], 'unattractive': scores[1], 'happy': scores[2], 'unhappy': scores[3], 'trustworthy': scores[4], 'untrustworthy': scores[5]}
        scores_attributes.append(scores_ranks)
    with open(f"{orig_scores_path}/uaf__norm_0_1.json", "w") as f:
        json.dump(scores_attributes, f)
            