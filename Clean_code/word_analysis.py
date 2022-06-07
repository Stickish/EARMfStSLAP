import itertools
import numpy as np
from collections import defaultdict

from sympy import nth_power_roots_poly

def calculate_word_scores(articles, orders, vocabulary, delta=2, gamma=1/2):
    """
    Computes a number of different measures of word importance based on the order data

    Input:
    ============================================================
    articles: List of all unique articles in the data set
    orders: Pandas Dataframe consisting of orders
    vocabulary: List of all unique words
    delta, gamma: Hyperparameters contolling the behaviour of the scores
    
    Returns:
    ============================================================
    The distance between the embeddings of the two articles
    """
    word_counts = defaultdict(int)
    word_pair_counts = defaultdict(int)
    n_articles = 0
    for _, row in orders.iterrows():
        order = row['Articles']
        n_articles += len(order)
        for article in order:
            for word in article.split():
                word_counts[word] += 1

        for article1, article2 in itertools.combinations(order, 2):
            for word1 in article1.split():
                for word2 in article2.split():
                    word_pair_counts[word1 + ' & ' + word2] += 1
                    word_pair_counts[word2 + ' & ' + word1] += 1


    word_list = [word for article in articles for word in article.split()]
    unique_words = np.unique(word_list) 

    word_pair_scores = defaultdict(float)

    max_count = np.max(np.fromiter(word_counts.values(), dtype=int))

    for i, word1 in enumerate(unique_words):
        
        for j in range(i, unique_words.shape[0]):
            word2 = unique_words[j]
            
            # if word_counts[word1] == 0 and word_counts[word2] == 0:
            #     word_pair_scores[word1 + ' & ' + word2] = 1 / max_count # Should probably be higher

            # elif word_pair_counts[word1 + ' & ' + word2] == 0:
            #     if word1 == word2:
            #        word_pair_scores[word1 + ' & ' + word2] = 1 / (word_counts[word1] + word_counts[word2])**gamma
            #     else:
            #         word_pair_scores[word1 + ' & ' + word2] = 1 / (word_counts[word1] + word_counts[word2])**delta
            #     word_pair_scores[word1 + ' & ' + word2] = 1 / max_count # Should probably be higher
            #     # word_pair_scores[word1 + ' & ' + word2] = 1 / (word_counts[word1] + word_counts[word2])
            if word_pair_counts[word1 + ' & ' + word2] > 0:
                if word1 == word2:
                    word_pair_scores[word1 + ' & ' + word2] = word_pair_counts[word1 + ' & ' + word2] / (word_counts[word1] + word_counts[word2])**gamma
                else:
                    word_pair_scores[word1 + ' & ' + word2] = word_pair_counts[word1 + ' & ' + word2] / (word_counts[word1] + word_counts[word2])**delta
                

    word_scores = defaultdict(float)
    single_word_scores = defaultdict(float)
    for words, score in word_pair_scores.items():
        word1, word2 = words.split(' & ')
        word_scores[word1] += score
        word_scores[word2] += score
        if word1 == word2:
            single_word_scores[word1] += score


    # print(single_word_scores['huawei'])
    # print(single_word_scores['skal'])
    # print(single_word_scores['transparent'])
    # print(single_word_scores)

    # word_pair_scores_sorted = [(k,v) for k, v in sorted(word_pair_scores.items(), key=lambda item: item[1], reverse=True)]
    # word_frequencies_sorted = [(k,v) for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)]
    # word_scores_sorted = [(k,v) for k, v in sorted(word_scores.items(), key=lambda item: item[1], reverse=True)]

    score_list = [word_scores[word] for word in vocabulary]
    single_score_list = [single_word_scores[word] for word in vocabulary]
    frequency_list = [word_counts[word]/n_articles for word in vocabulary] # not quite right, could give frequencies over 1...

    return score_list, single_score_list, frequency_list