import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


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

    support_articles = list(single_supports.index)
    min_support = single_supports.min()
    for a in test_articles:
        if a not in support_articles:
            single_supports.loc[a] = min_support

    return order_df, single_supports