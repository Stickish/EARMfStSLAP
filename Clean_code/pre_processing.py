### Imports
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import pickle
from collections import defaultdict
from joblib import Parallel, delayed


from sklearn.feature_extraction.text import CountVectorizer

from pp_functions import get_preprocessing_functions

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10,10)
color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'] 


                         

def preprocess_data(data, save_path=None, verbose=False):
    """
    Applies the functions specified in the file pp_functions.py on all article names
    Input:
    ====================================
    data: pandas Dataframe consising of the data that the pre-processing should be applied on.
    save_path: Path to a csv file where the pre-processed data should be saved
    verbose: Boolean flag marking if a selection of the pre-processed data should be shown

    Returns:
    ====================================
    data: The data after pre-processing
    """
    preprocessing_functions = get_preprocessing_functions()
                 
    for fcn in preprocessing_functions:
        data['ArticleName'] = data['ArticleName'].apply(fcn) 

    if verbose:
        data.head()
    
    if save_path is not None:
        data.to_csv(save_path, encoding='utf-8-sig')

    return data


def unique_orders(dataset, n_orders=1000):
    """
    Requires that dataset includes OrderId from SQL_query
    Input:
    ============================================================
    dataset: A dataset at least containing OrderId
    n_orders: Number of orders to consider. TODO: Actually implement this

    Output:
    ============================================================
    orders: pd DataFrame on the form [OrderId] = <article_1 ... article_n>, n=number of articles in order
    """
    unique_ids = dataset['OrderId'].unique()

    orders = pd.DataFrame(columns=['OrderId', 'Articles'])


    for index, id in enumerate(unique_ids):
        order = dataset[dataset['OrderId'] == id]
        articles = list(order['ArticleName'])
        orders.loc[index] = [id, articles]

    return orders


def cross_validation_split(data_path, 
                           k=4, 
                           val_prop=0.25, 
                           n_orders=20000):
    """
    Splits k*n_orders orders into k n_orders with val_prop validation/test data

    INPUTS:
    ================================
    data_path: Path to csv file from sqlQuery
    k: How many folds to create
    val_prop: How much of the data that should be for validation
    n_orders: How many orders should be in each fold

    RETURNS:
    ================================

    """
    data = pd.read_csv(data_path, sep=';')
    order_ids = list(data['OrderId'].unique())
    num_orders_in_data = len(order_ids)
    if k*n_orders > num_orders_in_data:
        n_orders = num_orders_in_data // k

    order_ids = order_ids[:k*n_orders]
    folds = []
    for i in range(k):
        val_split = int(n_orders * ((i+1) - val_prop))-1
        #print(val_split)
        first_train_order_id = order_ids[i*n_orders]
        last_train_order_id = order_ids[val_split]
        last_val_order_id = order_ids[(i+1) * n_orders - 1]

        first_train_order_idx = data.index[data['OrderId'] == first_train_order_id].tolist()[0]
        last_train_order_idx = data.index[data['OrderId'] == last_train_order_id].tolist()[-1]
        last_val_order_idx = data.index[data['OrderId'] == last_val_order_id].tolist()[-1]

        train_data = data.iloc[first_train_order_idx:last_train_order_idx+1]
        val_data = data.iloc[last_train_order_idx+1:last_val_order_idx+1]
        folds.append([train_data, val_data])

    return folds


def train_val_test_split(data=None, data_path=None, val_prop=0.1, test_prop=0.2, n_orders=1000):
    """
    Splits the orders into train, val and test sets

    Inputs:
    =====================================
    data_path: Path to a csv file with each row containing OrderId and one article on each row
    val_prop, test_prop: Proportion of the data to be put into the validation and test sets. The rest is put into the training set
    n_orders: The total number of orders to be considered
    
    Returns:
    =====================================
    The input data split into train, validation and test sets
    """
    if data_path is not None:
        data = pd.read_csv(data_path, sep=';')

    orders = list(data['OrderId'].unique())
    num_orders_in_data = len(orders) # 8898
    if n_orders == 0 or n_orders > num_orders_in_data:
        n_orders = num_orders_in_data

    orders = orders[:n_orders]

    # Compute the indices in the list of unique orders where it should be split
    val_split = int(n_orders * (1 - val_prop - test_prop))
    test_split = int(n_orders * (1 - test_prop))

    # Translate the indices above to order ids where the data set should be split (as the data set may contain several rows for each order)
    last_train_order_id = orders[val_split - 1]
    first_test_order_id = orders[test_split]
    last_test_order_id = orders[-1]

    # Compute the indices where the data set should be split
    last_train_order_index = data.index[data['OrderId'] == last_train_order_id].tolist()[-1]
    first_test_order_index = data.index[data['OrderId'] == first_test_order_id].tolist()[0]
    last_test_order_index = data.index[data['OrderId'] == last_test_order_id].tolist()[-1]


    # Split the data accordingly
    train_data = data.iloc[ : last_train_order_index + 1]
    val_data = data.iloc[last_train_order_index+1 : first_test_order_index]
    test_data = data.iloc[first_test_order_index : last_test_order_index + 1]
    
    return train_data, val_data, test_data



class Vectorizer:
    """
    TODO: Describe class. 
    """
    def __init__(self, unique_articles):
        self.unique_articles = unique_articles
        self.cv = CountVectorizer(ngram_range=(1, 1),
                                  analyzer='word',
                                  lowercase=False).fit(unique_articles)
        
        self.vocabulary_size = len(self.cv.vocabulary_.keys()) # num unique words
        self.num_words_max = self.article_max_length()
        self.map = self.get_str_to_array_map() # map for fast transform of articles
        
    
    def fast_transform(self, article):
        """
        Function for transforming articles to BoW representations faster,
        uses a dictionary with key article_name and value article_BoW
        Input:
        ================================
        article: the name of the article to transform

        Output:
        ================================ 
        y: the BoW representation of the article name
        """
        assert type(article) == str

        try:
            y = self.map[article]
        except KeyError:
            y = self.transform(article)
            self.map[article] = y # Update str_to_array mapping 
        return y

    def transform(self, article):
        """
        Wrapper function for transforming a single article into a BoW
        Input:
        ================================
        article: the article to transform
        
        Output:
        ================================
        y: the BoW representation of the article
        """
        assert type(article) == str
        
        y = self.cv.transform([article]).toarray()
        y[np.where(y > 1)] = 1

        return y


    def transform_dataset(self, dataset):
        """
        Wrapper function for transforming an entire dataset of articles
        Input:
        ================================
        dataset: list of article names

        Output:
        ================================
        X: array where row i = article i in dataset
        """
        X = self.cv.transform(dataset).toarray()
        X[np.where(X > 1)] = 1
        
        return X


    def article_max_length(self):
        """
        Helper function for determining the maximum number of words in the unique articles, used when converting articles to OBoWs
        TODO: Probably deprecated, remove
        """
        max_len=0
        
        for a in self.unique_articles:
            s = a.split()
            a_len = len(s)
            
            if a_len > max_len:
                max_len = a_len
        
        return 12 # Insane fix
        # return max_len


    def get_str_to_array_map(self):
        """
        Helper function for creating a map between articles and BoWs.
        Converts every article in self.unique articles
        """

        str_to_array_map = {} 

        for article in self.unique_articles:
            bow = self.transform(article)
            str_to_array_map[article] = bow

        return str_to_array_map
        

    def vectorize_data(self,
                       orders,
                       num_negative_samples=1):
    
        """
        Function for creating the training data for the network which uses a form of triplet loss

        Input:
        =====================
        orders: dataframe with columns OrderId and OrderList ex. row[1337] = [banana, electric guitar]
        num_negative_samples: The number of negative samples to be generated for each of the individuals in a positive pair
        =====================
        
        Ouput:
        =====================
        X: array with shape (n,3,m) where n=total combinations of 2 within the orders, 3=number of samples per pair (A, P , N), m=vocabulary size
        """
        X = []
        for _, row in orders.iterrows():
            order = row[1]
            order_vec = [self.fast_transform(a) for a in order] # Vectorize the order ## Might be worth to use transform_dataset here
            
            combinations = itertools.combinations(order_vec, 2) # Get all possible unique combinations of size 2
            for set_2 in combinations:
                anchor = set_2[0]
                positive = set_2[1]
                
                # Negative sampling
                for _ in range(num_negative_samples):
                   
                    negative = str(np.random.choice(self.unique_articles))
                    # While loop to ensure negative sampling outside of 
                    neg_in_order = negative in order
                    while neg_in_order:
                        negative = str(np.random.choice(self.unique_articles))
                        neg_in_order = negative in order
                    negative = self.fast_transform(negative)
                    X.append([anchor, positive, negative])
                    X.append([positive, anchor, negative]) # Does this makes sense? Are we adding extra versions ## Makes sense

        # Convert to float32 arrays
        X = np.asarray(X)
        # Reshapes since the X above have the shape (n,3,1,m)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[3]))
        return X


    
    def get_articles_for_embedding(self,
                                   unique_test_articles):
        """
        Function for returning all unique articles in the entire data, both names and BoWs
        Input:
        =====================
        unique_test_articles: Unique articles in the test data, since training and validation are already known
        
        Ouput:
        =====================
        article_names: numpy array of article names
        article_vecs: numpy array of BoWs corresponding to the article names
        """
        unique_articles_full = list(dict.fromkeys(list(self.unique_articles) + unique_test_articles))
        article_vecs = self.transform_dataset(unique_articles_full)
    
        return unique_articles_full, article_vecs

    
    def get_sparse_bow_embeddings(self, 
                                  unique_test_articles, 
                                  save_path=None):
        """
        Vectorizer has this already.
        """
        unique_articles, article_vecs = self.get_articles_for_embedding(unique_test_articles)
        # article_vecs = np.reshape(article_vecs, (-1, article_vecs.shape[2]))

        name_to_bow = defaultdict(list)
        for article, vec in zip(unique_articles, article_vecs):
            for index, bit in enumerate(vec):
                if bit == 1: 
                    name_to_bow[article].append(index)


        # Save the rules to file if desired
        if type(save_path) == str:
            try:
                bow_file = open(save_path, "wb")
                pickle.dump(name_to_bow, bow_file)
                bow_file.close()
            except FileNotFoundError as e:
                print(f"File doesnt exist, unable to save. Error: {e}")

        return name_to_bow

    

