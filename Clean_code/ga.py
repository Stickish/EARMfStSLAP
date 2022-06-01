from ctypes import pointer
from wsgiref.validate import WriteWrapper
import numpy as np
import random
from evaluation import evaluate_solution, evaluate_solution_for_greedy
from tqdm import tqdm
import copy
from joblib import Parallel, delayed
import time
from matplotlib import pyplot as plt

class Individual:
    def __init__(self, index, placements):
        self.index = index
        self.placements = placements
    
    def copy(self):
        # return Individual(self.index, self.placements.copy())
        return copy.deepcopy(self)

    def flip_indices(self, i, j):
        self.placements[i], self.placements[j] = self.placements[j], self.placements[i]
        return self # Shouldn't need this one


class GeneticModel:
    def __init__(self, layout, solution=None, warm_start=False, articles=None, rules=None, distance_weight=1, rule_weight=1, population_size=100, n_iter=100, crossover_rate=1, mutation_rate=0.1, k_selection=3, orders=None, crossover='crossover_improved', fitness='fitness_random', parallel=False, verbose=0):
        self.layout = layout
        self.n_spaces = len(articles)
        self.solution = solution
        self.warm_start = warm_start
        self.n_articles = len(articles)
        self.articles = articles
        self.rules = rules
        self.distance_weight = distance_weight
        self.rule_weight = rule_weight
        self.population_size = population_size
        self.n_iter = n_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.k_selection = k_selection
        self.orders = orders
        self.parallel = parallel
        self.verbose = verbose



        crossover_mapping = {
            'crossover_height': self.crossover_height
        }

        fitness_mapping = {
            'fitness_distance_and_rules': self.fitness_distance_and_rules,
            'fitness_order_picking': self.fitness_order_picking
        }

        self.crossover = crossover_mapping[crossover]
        self.fitness = fitness_mapping[fitness]

    
    # Fitness function for placing common articles near the depot and also placing articles with rules close to each other
    def fitness_distance_and_rules(self, individual):
        fitness_score = 0
        for article_index, shelf in enumerate(individual.placements):
            fitness_score += self.distance_weight * self.articles[article_index].support * self.layout.distances[self.layout.depot_location][self.layout.shelves[shelf].pick_location]
            for other_article, weight in self.articles[article_index].rules.items():
                j = individual.placements[other_article.index]
                fitness_score += self.rule_weight * weight * self.layout.distances[self.layout.shelves[shelf].pick_location][self.layout.shelves[j].pick_location]
        return fitness_score

    def fitness_order_picking(self, individual):
        return evaluate_solution(individual.placements, self.orders, self.layout, batch_size=1)


    # Function for selection of the strongest individuals
    def selection(self, population, scores, k=3):
        candidates = random.sample(range(self.population_size), k)
        selected_i = candidates[0]
        for i in candidates[1:]:
            if scores[i] < scores[selected_i]: # A lower score is better
                selected_i = i
        return population[selected_i].copy()


    def adjust_article_placements(self, individual):
        for i, old_shelf_index in reversed(list(enumerate(individual.placements))): 
            while self.article_counts[individual.index, old_shelf_index] > self.layout.shelves[old_shelf_index].height: # TODO: if instead of while?
                for new_shelf_index, shelf in enumerate(self.layout.shelves): # TODO: Make sure shelves are sorted by distance to depot
                    if self.article_counts[individual.index, new_shelf_index] < shelf.height:
                        self.article_counts[individual.index, old_shelf_index] -= 1
                        individual.placements[i] = new_shelf_index
                        self.article_counts[individual.index, new_shelf_index] += 1
                        break
                    
        return individual.copy()


    def crossover_height(self, p1, p2):
        crossover_points = random.sample(range(self.n_articles), self.crossover_rate)
        c1 = p1.copy()
        c2 = p2.copy()
        
        for point in crossover_points:

            self.article_counts[c1.index, c1.placements[point]] -= 1
            self.article_counts[c2.index, c2.placements[point]] -= 1

            c1.placements[point] = p2.placements.copy()[point]
            c2.placements[point] = p1.placements.copy()[point]

            self.article_counts[c1.index, c1.placements[point]] += 1
            self.article_counts[c2.index, c2.placements[point]] += 1

        c1 = self.adjust_article_placements(c1)
        c2 = self.adjust_article_placements(c2)

        return (c1.copy(), c2.copy())
        


    # Function for performing mutations by randomly flipping two indices
    def mutation(self, individual):
        for i in range(len(individual.placements)):
            if np.random.rand() < self.mutation_rate:
                flip_index = np.random.randint(0, len(individual.placements))
                individual.flip_indices(i, flip_index)

        return individual
        
        
    # Main function for the genetic algorithm
    def genetic_algorithm(self):

        
        possible_values = []
        for i, shelf in enumerate(self.layout.shelves):
            for _ in range(shelf.height):
                possible_values.append(i) 

        # If warm start is desired, we initialize the placements so that frequent articles are placed near the depot
        if self.warm_start: 
            random_fraction = 0.95
            if self.solution is None:
                # possible_values.sort(key=lambda i: self.layout.distances[self.layout.shelves[i].pick_location][self.layout.depot_location])

                # Make most individuals randomized
                population = [Individual(i, random.sample(possible_values, self.n_articles)) for i in range(int(self.population_size*random_fraction))]

                # But add a few good solutions, where articles are placed in order by frequency
                for i in range(int(self.population_size*random_fraction), self.population_size):
                    population.append(Individual(i, possible_values[:self.n_articles]))

                # print([self.layout.distances[self.layout.depot_location][self.layout.shelves[index].pick_location] for index in population[-1].placements])
                
                print('Warm start by frequency')
                
            else:

                # Make most individuals randomized
                population = [Individual(i, random.sample(possible_values, self.n_articles)) for i in range(int(self.population_size*random_fraction))]

                # But add a few good solutions, where articles are placed in order by frequency
                for i in range(int(self.population_size*random_fraction), self.population_size):
                    population.append(Individual(i, self.solution.copy()))

                print('Warm start using custom solution')

        # If warm start is not desired, randomize the placements
        else:
            population = [Individual(i, random.sample(possible_values, self.n_articles)) for i in range(self.population_size)]


            self.article_counts = np.zeros((self.population_size, len(self.layout.shelves)))
            for individual in population:
                for shelf_index in individual.placements:
                    self.article_counts[individual.index, shelf_index] += 1

    

        # Initialize the best individual as the first one
        scores = [self.fitness(individual) for individual in population]
        best_individual, best_score = population[0], scores[0]

        for i in range(self.population_size):
                if scores[i] < best_score:
                    best_individual, best_score = population[i], scores[i]
                
        best_scores = [best_score]

    
        
        if self.verbose == 1:
            iterator = tqdm(range(self.n_iter))
        else:
            iterator = range(self.n_iter)

        for generation in iterator:

            # Score all individuals

            scores = [self.fitness(individual) for individual in population]


            # Look for the best scoring individual
            for i in range(self.population_size):
                if scores[i] < best_score:
                    best_individual, best_score = population[i], scores[i]

            best_scores.append(best_score)

            if self.verbose == 2 and generation % 10 == 9:
                print(f'Best score at generation {generation+1}: {best_score}')

            # Perform selection to pick a number of individuals who will breed
            selected_individuals = [self.selection(population, scores, k=self.k_selection) for _ in range(self.population_size)]

            self.article_counts = np.zeros((self.population_size, len(self.layout.shelves)))
            for i, individual in enumerate(selected_individuals):
                individual.index = i
                for shelf_index in individual.placements:
                    self.article_counts[individual.index, shelf_index] += 1


            # Produce a new generation by crossing parents in pairs and mutating the offsprings
            children = []
            for i in range(0, self.population_size, 2):
                p1, p2 = selected_individuals[i], selected_individuals[i+1]
                c1, c2 = self.crossover(p1, p2)
                # c1, c2 = p1, p2
                children.append(self.mutation(c1).copy())
                children.append(self.mutation(c2).copy())
                

            # Update the population
            population = children

        return best_individual.placements, best_scores


    def optimize_locations(self):
        self.solution_matrix = np.zeros_like(self.layout.layout_matrix)
        best_state, best_scores = self.genetic_algorithm()


        # Update the solution matrix
        for article_index, shelf_index in enumerate(best_state):
            shelf = self.layout.shelves[shelf_index]
            self.solution_matrix[shelf.location] += 1 
            self.articles[article_index].shelf = shelf

        solution = {}
        for article in self.articles:
            solution[article.name] = article

        return self.solution_matrix, best_scores, solution
     