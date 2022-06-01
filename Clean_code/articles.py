from collections import defaultdict

class Article:
    """
    Class for representing an article 

    Class variables:
    ============================================================
    name: A string containing the name of the article
    support: The support of the article in the training set
    min_size, max_size: The min and max size required/allowed for storing the article
    shelves: A set of shelves in which the article is stored
    rules: A python dictionary containing all rules from this article to all other articles
    size_penalty: A scaling factor that penalizes big articles when an article should be chosen to place
    """
    def __init__(self, name, support, shelf = None, rules = None, index = None):
        self.name = name
        self.support = support
        self.shelf = shelf
        self.index = index

        if rules is None:
            self.rules = defaultdict(float)
        else:
            self.rules = rules

        self.is_placed = False
        self.score = self.support 

    def reset_variables(self):
        """
        Reset the class variables for a new round of optimization
        """

        self.shelf = None 
        self.is_placed = False
        self.score = self.support


    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name



def create_articles(train_supports, ar_rules=None, ear_rules=None, nn_rules=None, ar_weight=1, ear_weight=1, nn_weight=1):
    """
    
    Input:
    =========================================
    train_supports: Pandas Series of articles and their supports
    ar_rules, ear_rules, nn_rules: Rule dictionaries
    ar_weight, ear_weight, nn_weight: Weights for scaling the influence given to each rule type

    Returns:
    =========================================
    train_articles: A list of instances of the Article class with weights set accordingly
    """
    train_articles = []
    for name, support in train_supports.items():
        article = Article(name=name, support=support)
        train_articles.append(article)

    train_articles.sort(key=lambda x: x.support, reverse=True)
        
    name_to_article = {}
    for article in train_articles:
        name_to_article[article.name] = article

    for i, article in enumerate(train_articles):
        article.index = i

        if ar_rules is not None and ar_weight != 0:
            for consequent, weight in ar_rules[article.name].items():
                article_2 = name_to_article[consequent]
                article.rules[article_2] += ar_weight * weight 
                article_2.rules[article] += ar_weight * weight

        if ear_rules is not None and ear_weight != 0:
            for consequent, rules in ear_rules[article.name].items():
                article_2 = name_to_article[consequent]
                for weight in rules:
                    article.rules[article_2] += ear_weight * weight
                    article_2.rules[article] += ear_weight * weight

        if nn_rules is not None and nn_weight != 0:
            for consequent, similarity in nn_rules[article.name].items():
                article_2 = name_to_article[consequent]
                article.rules[article_2] += nn_weight * similarity
                article_2.rules[article] += nn_weight * similarity

                
    return train_articles