import math
from random import random
import numpy as np



# Redundant? TODO: Remove
def binary_search(articles, target, lo, hi):
    """
    
    """
    if lo > hi:
        return -1

    mid = (hi + lo) // 2

    if articles[mid] == target:
        return mid
    
    elif articles[mid].score > target.score:
        return binary_search(articles, target, lo, mid-1)

    elif articles[mid].score < target.score:
        return binary_search(articles, target, mid+1, hi)

    else:
        
        while articles[mid].score == target.score:
            if articles[mid] == target:
                return mid
            else: 
                mid -= 1
            
        mid = (hi + lo) // 2
        while articles[mid].score == target.score:
            if articles[mid] == target:
                return mid
            else: 
                mid += 1
        
        return -1
    

    



class Greedy:
    """
    Class variables:
    ==========================
    layout: An instance of the Layout class representing the warehouse layout
    articles: list of instances of the Article class
    distance_weight, rule_weight: Determines how much impact should be given to the distance to depot and the distance between articles within rules
    rule_weight_for_article_scores: Determines how much weight should be given to rules when deciding which article to place next (compared to support)
    penalty_weight: Determines how much penalty is given on for placing an article on the same shelf as articles it is not correlated to. 
    """
    def __init__(self, layout, articles, distance_weight=1, rule_weight=1, rule_weight_for_article_scores=1, penalty_weight=1, hp_tuning=False):
        self.layout = layout

        self.shelves = self.layout.shelves

        self.available_shelves = set(layout.closest_shelves)

        self.n_shelves = len(self.shelves)  # m
        self.n_articles = len(articles) # n

        self.articles = articles
        

        self.articles.sort(key=lambda a: a.score)
        
        self.unplaced_articles = self.articles.copy()
        self.n_placed_articles = 0

        self.distance_weight = distance_weight
        self.rule_weight = rule_weight
        self.rule_weight_for_article_scores = rule_weight_for_article_scores
        self.penalty_weight = penalty_weight
        self.hp_tuning = hp_tuning

        self.reset_status()


    def reset_status(self):
        """
        Resets the shelves and articles for a new round of optimization
        """
        for shelf in self.shelves:
            shelf.n_articles = 0
            shelf.articles = set()
        
        for article in self.articles:
            article.reset_variables()



    def update_article_scores(self, placed_article):
        """
        Updates the scores of all unplaced articles after an article has been placed
        
        Input:
        ==========================
        placed_article: The article that was recently placed
        """

        placed_article.is_placed = True
        self.n_placed_articles += 1

        if len(self.unplaced_articles) <= 1:
            return

        # Update score
        for article, weight in placed_article.rules.items():
            article.score += weight * self.rule_weight_for_article_scores
        
        # Resort the list of unplaced articles
        self.unplaced_articles.sort(key=lambda article: article.score)


    # O(m*log(k)) sort of...
    def find_optimal_shelf_for_article(self, article):
        """
        Determines what shelf an article should be placed at

        Input:
        ====================
        article: The article that should be placed

        Returns:
        ====================
        best_shelf: The shelf that was chosen for placing the input article
        """
        min_score = math.inf
        best_shelf = None
        # Loop through all shelves, calculate a score for each shelf, and pick the lowest scoring one
        for shelf in self.available_shelves:
            dist_to_depot = self.layout.distances[self.layout.depot_location][shelf.pick_location]
            depot_score = dist_to_depot * (article.support + 1e-10)

            rule_score = 0
            for other_article, weight in article.rules.items():
                if other_article.is_placed:
                    dist = self.layout.distances[other_article.shelf.pick_location][shelf.pick_location]

                    rule_score += dist * weight

            penalty_term = 0
            if self.n_placed_articles < len(self.unplaced_articles) * 0.9:
                for other_article in shelf.articles:
                    if other_article not in article.rules.keys():
                        penalty_term += self.penalty_weight * article.support * other_article.support 

        
            total_score = depot_score * self.distance_weight + rule_score * self.rule_weight + penalty_term
            if total_score < min_score:
                min_score = total_score
                best_shelf = shelf


        return best_shelf


    # O(n*log(n)*m*log(k)) sort of...
    def optimize_locations(self):
        """
        Runs the greedy algorithm and places one article at a time until all articles are placed.

        Returns:
        ====================
        solution_matrix: Visualization of how the articles are distributed in the warehouse
        0: For consistency with the other classes for optimizing article placements
        solution: A dictionary mapping article names to articles. 
        """

        solution_matrix = np.zeros_like(self.layout.layout_matrix)
        
        if self.hp_tuning:
            for article in self.unplaced_articles:
                article.score = 0
    


        for _ in range(self.n_articles):
            next_article = self.unplaced_articles.pop()


            shelf = self.find_optimal_shelf_for_article(next_article)
            try:
                shelf.n_articles += 1
            except AttributeError:
                raise Exception('All articles doesn\'t fit in the warehouse. Try increasing layout size.')


            if shelf.is_full():
                self.available_shelves.remove(shelf)
                for next_shelf in shelf.connected_shelves:
                    if not next_shelf.is_full():
                        self.available_shelves.add(next_shelf)

                
            next_article.shelf = shelf
            shelf.articles.add(next_article)

            solution_matrix[shelf.location] += 1

            self.update_article_scores(next_article)

        solution = {}
        for article in self.articles:
            solution[article.name] = article
        
 

        return solution_matrix, 0, solution









