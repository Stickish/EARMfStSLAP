from importlib_metadata import unique_everseen
import numpy as np
import pickle as pkl
from collections import defaultdict

def compute_nn_radius_rules(single_supports, embedding_space, embedding_map, r=2, save_path=None):
    beta = r/10
    rules = defaultdict(dict)
    unique_articles = list(single_supports.keys())
    
    for article in unique_articles:
        embedding = embedding_map[article]
        distances = np.linalg.norm(embedding_space - embedding, axis=1)
        # Set distance to self to inf
        min_idx = np.argmin(distances)
        distances[min_idx] = np.infty
        rules[article] = {}
        
        while np.min(distances) < r:
            min_idx = np.argmin(distances)
            article_2 = unique_articles[min_idx]
            rules[article][article_2] = beta/(distances[min_idx] + beta) # similarity
            distances[min_idx] = np.infty
            
    
    if type(save_path) == str:
        try:
            rule_file = open(save_path, "wb")
            pkl.dump(rules, rule_file)
            rule_file.close()
        except FileNotFoundError as e:
            print(f"File doesnt exist, unable to save. Error: {e}")
            print(f'Most likely caused by the directory not exxisting beforehand!\n')

    return rules


def compute_nn_rules(single_supports, embedding_space, embedding_map, k=5, r=None, save_path=None):              
    """
    INPUT:
    =========
    single_supports: A pandas dataframe containing the supports of the unique articles
    embedding_space: An numpy array with shape (#unique_articles, embedding_dim) of all embeddings
    embedding_map: A dictionary mapping articles to their embedding
    k: How many rules to create for each article
    r: Radius to be within to count as a rule
    save_path: Path for which the rules should be saved to
    
    OUTPUT:
    ========
    rules: a dictionary containg the rules generated, looks like [article_1][article_2] = (similarity, support)

    """
    # print(f'r = {r} which is None?')
    # TODO: Try to load embeddings, tbh kinda pointless saves a couple of seconds

    beta = 0.01
    if r is not None:
        beta = r/10
    rules = defaultdict(dict)
    unique_articles = list(single_supports.keys())
    for article in unique_articles:
        embedding = embedding_map[article]
        distances = np.linalg.norm(embedding_space - embedding, axis=1)
        # Set distance to itself to inf
        min_idx = np.argmin(distances)
        distances[min_idx] = np.infty
        
        rules[article] = {}
        # Find the k closest articles and add to rule dict
        if r is None:
            for _ in range(k):
                min_idx = np.argmin(distances)
                article_2 = unique_articles[min_idx]
                rules[article][article_2] = beta/(distances[min_idx] + beta)
                distances[min_idx] = np.infty
        
        elif r > 0:
            for _ in range(k):
                min_idx = np.argmin(distances)
                if distances[min_idx] < r:
                    article_2 = unique_articles[min_idx]
                    rules[article][article_2] = beta/(distances[min_idx] + beta)
                    distances[min_idx] = np.infty
                else:
                    break
        
    # TODO: Pickle the rules more useful saves like 4 seconds
    if type(save_path) == str:
        try:
            rule_file = open(save_path, "wb")
            pkl.dump(rules, rule_file)
            rule_file.close()
        except FileNotFoundError as e:
            print(f"File doesnt exist, unable to save. Error: {e}")
            print(f'Most likely caused by the directory not exxisting beforehand!\n')

    return rules


def get_rules(filepath):
    """
    Gets rules from file. TODO: Maybe this should be a file in its own as it is used for all types of rules

    Input:
    ============================================================
    filepath: Path to the file containing the rules
    
    Returns:
    ============================================================
    rules: A dictionary of rules
    """

    rule_file = open(filepath, "rb")
    rules = pkl.load(rule_file)
    rule_file.close()

    return rules

