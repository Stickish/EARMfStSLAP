from collections import defaultdict
import itertools
from operator import is_ # ???
import pickle
from matplotlib.pyplot import close
import numpy as np
from embeddings import get_embeddings


def compute_distance(a1, a2, embeddings, word_weights=None, is_sparse=False):
    """
    Computes the distance between two articles in the embedding space provided

    Input:
    ============================================================
    a1, a2: The articles between which the distance should be calculated
    embeddings: A dictionary containing embeddings for all articles
    word_weights: List of scaling factors for each embedding dimension
    is_sparse: Boolean flag marking if the embeddings are given in sparse format or not
    
    Returns:
    ============================================================
    The distance between the embeddings of the two articles
    """
    try:
        e1 = embeddings[a1]
    except:
        e1 = a1
    
    try:
        e2 = embeddings[a2]
    except:
        e2 = a2

    
    # print(a1, e1)
    # print(a2, e2)
    

    dist = 0.
    if is_sparse:
        if word_weights is None:
            word_weights = defaultdict(lambda: 1)
        i = 0
        j = 0
        try:
            while True:
                if e1[i] == e2[j]:
                    i += 1
                    j += 1

                elif e1[i] < e2[j]:
                    dist += word_weights[e1[i]]
                    i += 1

                elif e1[i] > e2[j]:
                    dist += word_weights[e2[j]]
                    j += 1


                if i >= len(e1) and j >= len(e2): 
                    return dist

                if i >= len(e1):
                    for jj in range(j, len(e2)):
                        dist += word_weights[e2[jj]]
                    return dist

                if j >= len(e2):
                    for ii in range(i, len(e1)):
                        dist += word_weights[e1[ii]]
                    return dist
        except Exception as e:
            raise e
            print(a1, ' & ', a2)
            return max(len(e1), len(e2))

    else:
        if word_weights is None:
            return np.linalg.norm(np.array(e1) - np.array(e2))
        else:
            # Är det inte konstigt att vikta orden här?, vi viktar embeddingsen inte bow, eller använder du den här funktionen där? ## Högst oklart
            return np.linalg.norm(np.multiply(np.array(e1) - np.array(e2), word_weights))


def compute_similarity(dist, beta=0.0001):
    """
    Computes the similarity between two articles based on the distance between them

    Input:
    ============================================================
    dist: Distance between two articles
    beta: Hyperparameter regulating the size of the output
    
    Returns:
    ============================================================
    A similarity measure between the embeddings of the two articles, ranging from 0 (infinitely separated) an 1 (located in the same point)
    """
    return beta / (dist + beta)


def sparse_subtract(a, b):
    """
    Subtracts two sparse vectors. Assumes the vectors are sorted by index. 
    A positive index means there is a 1 at that index, and a negative index means there is a -1 in the position abs(index)
    Input:
    =========================================
    a, b: Sparse vectors to be subtracted
    Returns:
    =========================================
    result: a - b in sparse format
    """
    result = []
    i = 0
    j = 0
    while True:
        if a[i] == b[j]:
            i += 1
            j += 1

        elif np.abs(a[i]) < np.abs(b[j]):
            result.append(a[i])
            i += 1

        elif np.abs(a[i]) > np.abs(b[j]):
            result.append(-b[j])
            j += 1

        if i >= len(a) and j >= len(b): 
            return result

        if i >= len(a):
            for jj in range(j, len(b)):
                result.append(-b[jj])
            return result

        if j >= len(b):
            for ii in range(i, len(a)):
                result.append(a[ii])
            return result


def sparse_add(a, b):
    """
    Adds two sparse vectors. Assumes the vectors are sorted by index. 
    A positive index means there is a 1 at that index, and a negative index means there is a -1 in the position abs(index)
    Input:
    =========================================
    a, b: Sparse vectors to be added
    Returns:
    =========================================
    result: a + b in sparse format
    """
    result = []
    i = 0
    j = 0
    while True:
        if a[i] == b[j]:
            result.append(a[i])
            result.append(b[j])
            i += 1
            j += 1

        elif a[i] == -b[j]:
            i += 1
            j += 1

        elif np.abs(a[i]) < np.abs(b[j]):
            result.append(a[i])
            i += 1

        elif np.abs(a[i]) > np.abs(b[j]):
            result.append(b[j])
            j += 1

        if i >= len(a) and j >= len(b): 
            return result

        if i >= len(a):
            for jj in range(j, len(b)):
                result.append(b[jj])
            return result

        if j >= len(b):
            for ii in range(i, len(a)):
                result.append(a[ii])
            return result


def find_k_most_similar(target, all_items, embeddings, beta, k=2, word_weights=None, is_sparse=False):
    """
    Finds the k items that are the most similar to the target item in the embedding space provided

    Input:
    ============================================================
    target: The article for which similar articles should be found
    all_items: A list of all items to search within
    embeddings: A dictionary containing embeddings for all articles
    k: The number of most similar articles to be found
    word_weights: List of scaling factors for each embedding dimension
    is_sparse: Boolean flag marking if the embeddings are given in sparse format or not
    
    Returns:
    ============================================================
    closest_items: A list over the k articles that are most similar to the target article
    similarities: A list of similarities, with the similarity on each index corresponding to the similarity between the target article and the item at that index in closest_items
    """
    closest_items = [0 for _ in range(k)]
    min_distances = [np.infty for _ in range(k)]
    
    try:
        items = list(all_items.keys())
    except:
        items = all_items

    for item in items:
        if item == target:
            continue

        # Maintains a sorted list of the closest items with their respective distances
        dist = compute_distance(item, target, embeddings, word_weights=word_weights, is_sparse=is_sparse)
        for i, min_distance in enumerate(min_distances):
            if dist < min_distance:
                min_distances.insert(i, dist)
                closest_items.insert(i, item)
                min_distances.pop()
                closest_items.pop()
                break
    
    similarities = []
    for dist in min_distances:
        similarities.append(compute_similarity(dist, beta))

    return closest_items, similarities

def find_similar_items(target, all_items, embeddings, beta, k=2, r=1, word_weights=None, is_sparse=False):
    """
    Finds the k items that are the most similar to the target item in the embedding space provided

    Input:
    ============================================================
    target: The article for which similar articles should be found
    all_items: A list of all items to search within
    embeddings: A dictionary containing embeddings for all articles
    k: The number of most similar articles to be found
    word_weights: List of scaling factors for each embedding dimension
    is_sparse: Boolean flag marking if the embeddings are given in sparse format or not
    
    Returns:
    ============================================================
    closest_items: A list over the k articles that are most similar to the target article
    similarities: A list of similarities, with the similarity on each index corresponding to the similarity between the target article and the item at that index in closest_items
    """
    closest_items = []
    min_distances = [np.infty for _ in range(k)]
    
    try:
        items = list(all_items.keys())
    except:
        items = all_items

    for item in items:
        if item == target:
            continue

        # Maintains a sorted list of the closest items with their respective distances
        dist = compute_distance(item, target, embeddings, word_weights=word_weights, is_sparse=is_sparse)
        if dist < r:
            if len(closest_items) == 0:
                min_distances[0] = dist
                closest_items.append(item)
            else:
                for i, min_distance in enumerate(min_distances):
                    if dist < min_distance:
                        min_distances.insert(i, dist)
                        closest_items.insert(i, item)
                        min_distances.pop()
                        if len(closest_items) < k:
                            closest_items.pop()
                        break

    
    similarities = []
    for dist in min_distances[:len(closest_items)]:
        similarities.append(compute_similarity(dist, beta))

    return closest_items, similarities


def find_articles_within_radius(target, all_items, embeddings, beta, radius=2, word_weights=None, is_sparse=False):
    """
    Finds all items that have a maximum distance of radius to the target item in the embedding space provided

    Input:
    ============================================================
    target: The article for which similar articles should be found
    all_items: A list of all items to search within
    embeddings: A dictionary containing embeddings for all articles
    radius: The maximum distance for an article to be considered close to the target article
    word_weights: List of scaling factors for each embedding dimension
    is_sparse: Boolean flag marking if the embeddings are given in sparse format or not
    
    Returns:
    ============================================================
    close_items: A list over all articles that are within the radius of the target article in the embedding space
    similarities: A list of similarities, with the similarity on each index corresponding to the similarity between the target article and the item at that index in closest_items
    """
    close_items = []
    distances = []
    
    try:
        items = list(all_items.keys())
    except:
        items = all_items

    for item in items:
        if item == target:
            continue

        # Maintains a sorted list of the close items with their respective distances
        dist = compute_distance(item, target, embeddings, word_weights=word_weights, is_sparse=is_sparse)
        if dist <= radius:
            distances.append(dist)
            close_items.append(item)
    
    similarities = []
    for item, dist in zip(close_items, distances):
        similarities.append(compute_similarity(dist, beta))

    return close_items, similarities


def find_parallel_rules(antecedent, consequent, possible_antecedents, embeddings, embedding_to_article, word_to_articles, is_sparse=False):
    """
    Finds all items that have a maximum distance of radius to the target item in the embedding space provided

    Input:
    ============================================================
    antecedent, consequent: Articles in the original rule 
    all_items: A list of all items to search within
    embeddings: A dictionary containing embeddings for all articles
    word_weights: List of scaling factors for each embedding dimension
    is_sparse: Boolean flag marking if the embeddings are given in sparse format or not
    
    Returns:
    ============================================================
    rules: List of tuples, each representing a rule
    """
    rules = []
    antecedent = embeddings[antecedent]
    consequent = embeddings[consequent]
    
    
    if is_sparse:
        rule = sparse_subtract(consequent, antecedent)
        if len(rule) == 0:
            return []

        for r in rule:
            if r < 0:  # What happens for index 0???
                possible_antecedents = set.intersection(possible_antecedents, word_to_articles[-r])
        for new_antecedent in possible_antecedents:
            try:
                embedding = sparse_add(rule, embeddings[new_antecedent]) # = consequent - antecedent + ia
            except:
                print(antecedent, consequent)
            
            # if embedding == [638, 1202, 1577, 1831, 2485, 2992, 3178]:
            #     print('iphone _8_ skärm glas och display svart')
            #     print(embedding_to_article[])
            try:
                new_consequent = embedding_to_article[str(embedding)]
                rules.append((new_antecedent, new_consequent))
            except:
                pass

    # else:
    #     for ia, ic in itertools.permutations(all_items, 2):
    #         fake_item = antecedent - consequent + ic
    #         distance = compute_distance(fake_item, ia, embeddings, word_weights=word_weights, is_sparse=is_sparse)
    #         if distance < 1: # == 0?
    #             rules.append((ia, ic, compute_similarity(distance)))

    return rules

            
def compute_ear_rules(rule_dict,
                         unique_items,
                         word_weights=None, 
                         k=None, radius=None, 
                         embeddings=None,
                         embeddings_path=None, 
                         save_path=None, 
                         is_sparse=False,
                         beta=0.0001,
                         parallel_rules=False,
                         parallel_weight=1):
    """
    For a given set of rules, create new rules between articles that are close to those already in a rule.

    Input:
    ============================================================
    rule_dict: A dictionary of rules obtained from Association rule mining
    word_weights: List of scaling factors for each embedding dimension
    k: The number of rules to be created for each other rule. Exactly one of k and radius should be provided TODO: make this obligatory
    radius: The maximum distance for an article to be considered close to the target article. Exactly one of k and radius should be provided TODO: make this obligatory
    embeddings: Dictionary mapping an article to it's embedding
    embeddings_path: Path to a file containing a saved embedding mapping
    save_path: Path to a pickle file in which the rules should be stored
    is_sparse: Boolean flag marking if the embeddings are given in sparse format or not
    
    Returns:
    ============================================================
    new_rules: A dictionary of all rules created
    """

    

    # Load embeddings
    if type(embeddings_path) == str:
        try:
            embeddings = get_embeddings(embeddings_path)
        except FileNotFoundError as e:
            print(f"File doesnt exist. Error: {e}")
            if embeddings == None:
                return None  
    

    new_rules = defaultdict(defaultdict(list).copy)

    if k is None and radius is None:
        return new_rules

    antecedents = set(rule_dict.keys())

    consequents = set()
    for rules in rule_dict.values():
        for consequent in rules.keys():
            consequents.add(consequent)

    unique_items_with_rules = set.union(antecedents, consequents)
    
    # closest_items_dict = {}
    closest_items_dict = defaultdict(list)

    for item in unique_items_with_rules:
        closest_items, similarities = find_similar_items(item, unique_items, embeddings, beta, k=k, r=radius, word_weights=word_weights, is_sparse=is_sparse)
        for item2, sim in zip(closest_items, similarities):
            closest_items_dict[item].append((item2, sim))


        # if k is not None:
        #     for item in unique_items_with_rules:
        #         closest_items, similarities = find_k_most_similar(item, unique_items, embeddings, beta, k=k, word_weights=word_weights, is_sparse=is_sparse)
        #         for item2, sim in zip(closest_items, similarities):
        #             closest_items_dict[item].append((item2, sim))
        #         # closest_items_dict[item] = zip(closest_items, similarities)
        # elif radius is not None:
        #     for item in unique_items_with_rules:
        #         closest_items, similarities = find_articles_within_radius(item, unique_items, embeddings, beta, radius=radius, word_weights=word_weights, is_sparse=is_sparse)
        #         closest_items_dict[item] = zip(closest_items, similarities)

    if parallel_rules:
        embedding_to_article = {}
        for item in unique_items:
            embedding_to_article[str(embeddings[item])] = item

        word_to_articles = defaultdict(set)
        for item in unique_items:
            for word in embeddings[item]:
                word_to_articles[word].add(item)


    for antecedent, rules in rule_dict.items():
        for consequent, weight in rules.items():

            if parallel_rules:
                for a, c in find_parallel_rules(antecedent, consequent, set(unique_items), embeddings, embedding_to_article=embedding_to_article, word_to_articles=word_to_articles, is_sparse=is_sparse):
                    new_rules[a][c].append(weight*parallel_weight) 

            
            # Create rules from the k nearest neighbors of the antecedent to the consequent
            for item, similarity in closest_items_dict[antecedent]:
                new_rules[item][consequent].append(weight*similarity)

            # Create rules from the antecedent to the k nearest neighbors of the consequent
            for item, similarity in closest_items_dict[consequent]:
                new_rules[antecedent][item].append(weight*similarity)

            

            # for item1, similarity1 in closest_items_dict[antecedent]:
            #     for item2, similarity2 in closest_items_dict[consequent]:
            #         new_rules[item1][item2].append(weight*(1-similarity1)*(1-similarity2))

                
    # Save the newly created rules to file if desired
    if type(save_path) == str:
        try:
            rule_file = open(save_path, "wb")
            pickle.dump(new_rules, rule_file)
            rule_file.close()
        except FileNotFoundError as e:
            print(f"File doesnt exist, unable to save. Error: {e}")

    return new_rules







