import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

#### Redundant? ####
def get_arm_data(orders, test_articles):
    """
    Pre-processes the data to be used for creating association rules

    Input:
    ============================================================
    orders: A pandas DataFrame with one column containing order ids and one containing the corresponding lists of articles
    test_articles: A list over unique articles in the test set.

    Output:
    ============================================================
    order_df: A pandas DataFrame where each row corresponds to a one-hot encoded order, and with each column representing an article
    single_supports: A pandas Series storing the frequency (support) of each article
    """

    order_list = []
    for _, row in orders.iterrows():
        order = row[1]
        order_list.append(order)

    order_te = TransactionEncoder()
    order_one_hot = order_te.fit(order_list).transform(order_list)
    order_df = pd.DataFrame(order_one_hot, columns=order_te.columns_)

    # Calculate frequency (support) of each item in the train set 
    single_supports = order_df.sum(axis=0)/order_df.shape[0]

    # for article in single_supports.index:
    #     single_supports[article] += np.random.normal(loc=1e-6, scale=1e-8)

    support_articles = list(single_supports.index)
    min_support = single_supports.min()

    # Adds supports for the test articles that did not appear in the input orders, equel to the minimum support of articles that did appear in the unput orders
    for a in test_articles:
        if a not in support_articles:
            single_supports.loc[a] = min_support / 2 # + np.random.normal(loc=1e-6, scale=1e-8)
            order_df[a] = False

    single_supports = single_supports.sort_values(ascending=False)
    order_df = order_df.sort_index(axis=1, key=lambda x: single_supports[x], ascending=False)

    return order_df, single_supports


def generate_itemsets(transactions, min_support=0.0001):
    """
    Generates all itemsets occuring more frequently than a certain threshold

    Input:
    ============================================================
    transactions: A pandas DataFrame with each row corresponding to an order
    min_support: The support threshold for an itemset to be kept

    Returns:
    ============================================================
    itemsets: All itemsets occuring more frequently than the min_support threshold
    itemsets_size_2: All itemsets of size 2 occuring more frequently than the min_support threshold
    """

    # Extract all itemsets occuring with a higher frequency than min_support 
    itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)

    # Filter the itemsets so that only sets of exavtly two items are kept
    itemsets['size'] = itemsets['itemsets'].apply(lambda x: len(x))
    itemsets_size_2 = itemsets.loc[itemsets['size'] <= 2]


    return itemsets, itemsets_size_2


def compute_ar_rules(itemsets, metric='confidence', min_threshold=0.1, save_path=None):
    """
    Generates all association rules within the input itemsets with a confidence above a certain threshold

    Input:
    ============================================================
    itemsets: A pandas DataFrame with each row consisting of an itemset
    metric: The metric to be used for deciding which rules should be kept
    min_threshold: The threshold  of the given metric for a rule to be kept
    save_path: Path to a pickle file in which the rules should be stored
    
    Returns:
    ============================================================
    rule_dict: A python dictionary mapping an article to a list of rules from that article
    """

    rule_df = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    # Convert the rules from a pandas DataFrame to a python dictionary
    rule_dict = defaultdict(defaultdict(float).copy)
    for _, rule in rule_df.iterrows():
        rule_dict[list(rule['antecedents'])[0]][list(rule['consequents'])[0]] = rule['lift'] - 1


    # Save the file if desired
    if type(save_path) == str:
        try:
            rule_file = open(save_path, "wb")
            pickle.dump(rule_dict, rule_file)
            rule_file.close()
        except FileNotFoundError as e:
            print(f"File doesnt exist, unable to save. Error: {e}")

    return rule_dict