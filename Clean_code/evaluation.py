import enum
import numpy as np
from tqdm import tqdm
from python_tsp.exact import solve_tsp_dynamic_programming

def solve_tsp(solution, items, layout):
    """
    Solves the TSP corresponding to picking a number of items from the proposed solution.
    
    Input:
    ============================================================
    solution: A list representing the locations where each article is stored
    items: A list of items to be picked
    layout: An instance of the layout class that represents the warehouse layout

    Returns:
    ============================================================
    distance: The shortest distance required to pick all items in the items list
    """
    # Generate a distance matrix containing the distances between each pair of articles that should be visited
    distance_matrix = np.zeros((len(items)+1, len(items)+1))
    for i, item_1 in enumerate(items):
        for j, item_2 in enumerate(items):
            if i == j:
                continue
            distance_matrix[i,j] = layout.distances[layout.shelves[solution[item_1][0]].pick_location][layout.shelves[solution[item_2][0]].pick_location] # TODO: Remove all stuff related to articles being placed at several shelves
        distance_matrix[i,-1] = layout.distances[layout.shelves[solution[item_1][0]].pick_location][layout.depot_location]
        distance_matrix[-1,i] = layout.distances[layout.shelves[solution[item_1][0]].pick_location][layout.depot_location]

    _, distance = solve_tsp_dynamic_programming(distance_matrix)
    return distance





def solve_tsp_for_genetic(solution, items, layout):
    """
    Solves the TSP corresponding to picking a number of items from the proposed solution.
    
    Input:
    ============================================================
    solution: A list representing the locations where each article is stored
    items: A list of items to be picked
    layout: An instance of the layout class that represents the warehouse layout

    Returns:
    ============================================================
    distance: The shortest distance required to pick all items in the items list
    """
    # Generate a distance matrix containing the distances between each pair of articles that should be visited
    distance_matrix = np.zeros((len(items)+1, len(items)+1))
    for i, item_1 in enumerate(items):
        for j, item_2 in enumerate(items):
            if i == j:
                continue
            distance_matrix[i,j] = layout.distances[layout.shelves[solution[item_1]].pick_location][layout.shelves[solution[item_2]].pick_location]   
        distance_matrix[i,-1] = layout.distances[layout.shelves[solution[item_1]].pick_location][layout.depot_location]
        distance_matrix[-1,i] = layout.distances[layout.shelves[solution[item_1]].pick_location][layout.depot_location]

    _, distance = solve_tsp_dynamic_programming(distance_matrix)
    return distance


def evaluate_solution(solution, test_df, layout, batch_size=1, verbose=False):
    """
    Calculates the average distance required to pick all orders for a given solution

    Input:
    ============================================================
    solution: The solution to be evaluated. A list representing the locations where each article is stored
    test_df: A pandas DataFrame containing a number of test orders
    layout: An instance of the layout class that represents the warehouse layout
    batch_size: The minimum number of items to be picked in one batch. 
    
    Returns:
    ============================================================
    average_distance_travelled: The avarage distance travelled to pick one order
    """
    distance_travelled = 0

    items = set()
    current_batch = set()
    if verbose:
        iterator = tqdm(test_df.iterrows())
    else:
        iterator = test_df.iterrows()
    for _, row in iterator:


        # Extract a list of items in the order
        for i, item in enumerate(row):
            if item:
                items.add(i)

        
        n_items = len(items)

        # Build up batch until significantly large
        if n_items < batch_size: 
            current_batch = items.copy()

        # When no more items fit in the batch, solve the tsp that corresponds to picking all orders in the batch
        else:
            if len(current_batch) > 0: 
                dist = solve_tsp_for_genetic(solution, current_batch, layout)
                distance_travelled += dist

            # Start a new batch with the articles that did not fit in the previous batch 
            items = set()
            for i, item in enumerate(row):
                if item:
                    items.add(i)
            current_batch = items.copy()
    
    dist = solve_tsp_for_genetic(solution, current_batch, layout)
    distance_travelled += dist
    # Calculate the avarage distance for picking all orders 
    average_distance_travelled = distance_travelled/(test_df.shape[0])
    return average_distance_travelled


def solve_tsp_for_greedy(solution, items, layout):
    """
    TODO: Merge with solve_tsp
    """
    # print(items)
    # return 0
    distance_matrix = np.zeros((len(items)+1, len(items)+1))
    for i, item_1 in enumerate(items):
        for j, item_2 in enumerate(items):
            if i == j:
                continue

            distance_matrix[i,j] = layout.distances[solution[item_1].shelf.pick_location][solution[item_2].shelf.pick_location] # TODO: Remove all stuff related to articles being placed at several shelves
        distance_matrix[i,-1] = layout.distances[solution[item_1].shelf.pick_location][layout.depot_location]
        distance_matrix[-1,i] = layout.distances[solution[item_1].shelf.pick_location][layout.depot_location]

    try:
        _, distance = solve_tsp_dynamic_programming(distance_matrix)
        return distance
    except:
        print('failed to run solve_tsp')
        return 0


def solve_tsp_for_greedy_no_depot(solution, items, layout):
    """
    TODO: Merge with solve_tsp
    """

    distance_matrix = np.zeros((len(items), len(items)))
    for i, item_1 in enumerate(items):
        for j, item_2 in enumerate(items):
            if i == j:
                continue

            distance_matrix[i,j] = layout.distances[solution[item_1].shelf.pick_location][solution[item_2].shelf.pick_location] # TODO: Remove all stuff related to articles being placed at several shelves
        # distance_matrix[i,-1] = layout.distances[solution[item_1].shelf.pick_location][layout.depot_location]
        # distance_matrix[-1,i] = layout.distances[solution[item_1].shelf.pick_location][layout.depot_location]
    try:
        _, distance = solve_tsp_dynamic_programming(distance_matrix)
        return distance
    except:
        return 0



def evaluate_solution_for_greedy(solution, test_df, layout, batch_size=1, verbose=False):
    """
    TODO: Merge with evaluate_solution
    """

    distance_travelled = 0

    items = set()
    current_batch = set()
    
    if verbose:
        iterator = tqdm(test_df.iterrows())
    
    else:
        iterator = test_df.iterrows()
    
    for _, row in iterator:
        # Extract a list of items in the order
        for item, item_in_order in row.iteritems():
            if item_in_order:
                items.add(item)

        # return 0
        n_items = len(items)
        # Build up batch until significantly large
        if n_items < batch_size: 
            current_batch = list(items)
        # When no more items fit in the batch, solve the tsp that corresponds to picking all orders in the batch
        else:
            # while len(current_batch) > 10:
            #     dist = solve_tsp_for_greedy(solution, current_batch[:10], layout)
            #     distance_travelled += dist
            #     current_batch = current_batch[10:]

            if len(current_batch) > 0: 
                dist = solve_tsp_for_greedy(solution, current_batch, layout)
                distance_travelled += dist
            # Start a new batch with the articles that did not fit in the previous batch 
            items = set()
            for item, item_in_order in row.iteritems():
                if item_in_order:
                    items.add(item)
            current_batch = list(items)

    dist = solve_tsp_for_greedy(solution, current_batch, layout)
    distance_travelled += dist
    # Calculate the avarage distance for picking all orders 
    average_distance_travelled = distance_travelled/(test_df.shape[0])
    return average_distance_travelled


def evaluate_solution_for_greedy_no_depot(solution, test_df, layout, batch_size=1, verbose=False):
    """
    TODO: Merge with evaluate_solution
    """

    distance_travelled = 0

    items = set()
    current_batch = set()
    if verbose:
        iterator = tqdm(test_df.iterrows())
    else:
        iterator = test_df.iterrows()
    for _, row in iterator:


        # Extract a list of items in the order
        for item, item_in_order in row.iteritems():
            if item_in_order:
                items.add(item)

        
        n_items = len(items)

        # Build up batch until significantly large
        if n_items < batch_size: 
            current_batch = list(items)

        # When no more items fit in the batch, solve the tsp that corresponds to picking all orders in the batch
        else:
            if len(current_batch) > 0: 
                while len(current_batch) > 10:
                    dist = solve_tsp_for_greedy_no_depot(solution, current_batch[:10], layout)
                    distance_travelled += dist
                    current_batch = current_batch[10:]

                dist = solve_tsp_for_greedy_no_depot(solution, current_batch, layout)
                distance_travelled += dist

            # Start a new batch with the articles that did not fit in the previous batch 
            items = set()
            for item, item_in_order in row.iteritems():
                if item_in_order:
                    items.add(item)
            current_batch = list(items)
    
    dist = solve_tsp_for_greedy_no_depot(solution, current_batch, layout)
    distance_travelled += dist
    # Calculate the avarage distance for picking all orders 
    average_distance_travelled = distance_travelled/(test_df.shape[0])
    return average_distance_travelled


