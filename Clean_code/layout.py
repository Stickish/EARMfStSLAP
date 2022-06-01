
import numpy as np
import math
from collections import defaultdict
import itertools
from joblib import Parallel, delayed
import time


class ShelfIndex:
    """
    Iterator for giving shelves appropriate indices
    """
    def __init__(self):
        self.value = -1

    def next(self):
        self.value += 1
        return self.value


class Shelf:
    """
    Class representing a single shelf

    Class variables:
    ============================================================
    location: Coordinates on the form (x, y)
    height: Number of levels, where each level may contain only one article type
    size: The size of one level of the shelf
    shelf_row: An instance of the ShelfRow class where this Shelf is located  
    n_articles: The number of spots in this shelf currently occupied by an article
    pick_location: The aisle point from which articles on this shelf is picked
    """ 
    def __init__(self, location=None, height=1, n_articles=0, index=None):
        self.location = location
        self.x = location[0]
        self.y = location[1]
        self.height = height
        self.n_articles = n_articles
        self.pick_location = None
        self.index = index
        self.connected_shelves = []
        self.articles = set()

    def __hash__(self):
        return hash(self.location)

    def __eq__(self, other):
        """
        Two shelves are defined to be equal if their positions are the same
        """
        return self.location == other.location

    def is_full(self):
        if self.height < self.n_articles:
            raise Exception(f'Shelf at {self.location} is overfull')
        else:
            return self.height == self.n_articles



class ShelfRow:
    """
    Class representing a row of shelves. 

    Class Variables: TODO: Be consistent in whether class variables that are not arguments of init are described here or not
    ============================================================
    shelves: A list of instances of the Shelf class
    aisle_directions: A list of allowed picking directions
    pick_locations: A set of aisle points where articles from this shelf row are picked TODO: Make this only be one single location?
    x_min, x_max, y_min, y_max: Coordinates for the extreme points of the shelf row  
    start_point, end_points: Extreme_points of the shelf row
    vertical, horizontal: Alignment of the shelf row TODO: are these really necessary? 
    length: length of the shelf row
    """

    def __init__(self, shelves=[]):
        self.shelves = shelves
        self.n_shelves = len(shelves)

        self.aisle_direction = None

        self.x_min = math.inf
        self.x_max = 0
        self.y_min = math.inf
        self.y_max = 0

        self.start_point = (self.x_min, self.y_min)
        self.end_point = (self.x_max, self.y_max)

        self.vertical = False
        self.horizontal = False

        self.length = 0

    def is_empty(self):
        """
        Checks if the ShelfRow is empty (does not contain any shelves)
        """
        return len(self.shelves) == 0
        

    def add(self, shelf):
        """
        Adds a shelf to the list of shelves
        """
        self.shelves.append(shelf)
        self.n_shelves += 1


    def remove(self, shelf):
        """
        Removes a shelf from the list of shelves
        """
        self.shelves.remove(shelf)
        self.n_shelves -= 1



    # def set_dimensions(self):
    #     """
    #     Sets the dimensions and alignment of the shelf row
    #     """

    #     # Search for start and end points of the row
    #     for shelf in self.shelves:
    #         x, y = shelf.x, shelf.y
    #         if x < self.x_min:
    #             self.x_min = x
    #         if x > self.x_max:
    #             self.x_max = x
    #         if y < self.y_min:
    #             self.y_min = y
    #         if y > self.y_max:
    #             self.y_max = y


    #     self.start_point = (self.x_min, self.y_min)
    #     self.end_point = (self.x_max, self.y_max)

    #     # Check if vertical or horizontal
    #     if self.x_min == self.x_max:
    #         self.vertical = True
    #     if self.y_min == self.y_max:
    #         self.horizontal = True

    #     # Set row length
    #     self.length = max(self.x_max - self.x_min, self.y_max - self.y_min)


    def set_pick_locations(self, layout):
        """
        Set the points from which articles in each shelf of this row can be picked

        Input:
        ============================================================
        layout: An instance of the Layout class

        Returns:
        ============================================================
        pick_locations: A list of all points from which articles in this shelf row can be picked 
        """
    
        # If start point has a free tile above it assume the entire row can be picked from above
        
        if layout.layout_matrix[self.shelves[0].x + 1, self.shelves[0].y] == -1:
            self.aisle_direction = (1, 0)
        
        # If start point has a free tile below it the enitre row can be picked from below
        elif layout.layout_matrix[self.shelves[0].x - 1, self.shelves[0].y] == -1:
            self.aisle_direction = (-1, 0)           

        # If start point has a free tile to the left, the entire row can be picked from the left
        elif layout.layout_matrix[self.shelves[0].x, self.shelves[0].y - 1] == -1:
            self.aisle_direction = (0, -1)
        
        # If start point has a free tile to the right, the entire row can be picked from the right
        elif layout.layout_matrix[self.shelves[0].x, self.shelves[0].y + 1] == -1:
            self.aisle_direction = (0, 1)           
        
        # There must exist some direction to pick from
        if self.aisle_direction is None:
            raise Exception(f'Shelf at {self.shelves[0].location} not accessible')


        # Check for valid pick locations. TODO: multiple directions doesn't really work...
    
        j_range = layout.bin_size
        for i in range(0, self.n_shelves, layout.bin_size):
            mid_index = i + min(layout.bin_size, (self.n_shelves - i)) // 2

            pick_location = (self.shelves[mid_index].x + self.aisle_direction[0], self.shelves[mid_index].y + self.aisle_direction[1])
            j_range = min(layout.bin_size, self.n_shelves - i)

            self.shelves[mid_index].pick_location = pick_location

            for j in range(j_range):
                if i + j >= self.n_shelves: 
                    break

                if i+j != mid_index:
                    self.shelves[mid_index].height += self.shelves[i+j].height

                    
        self.shelves = [shelf for shelf in self.shelves if shelf.pick_location is not None]

        # Sort shelves by distance to depot
        # if self.vertical:
        #     self.shelves = sorted(self.shelves, key=lambda item: item.location[0])

        # elif self.horizontal:
        #     self.shelves = sorted(self.shelves, key=lambda item: item.location[1])

        # return self.pick_locations


class Layout:
    """
    Class for representing a warehouse layout

    Class variables:
    ============================================================
    shape: The shape of the layout_matrix
    depot_location: Location of the start and end position of the pickers
    shelves: A list of all shelves in the warehouse 
    shelf_rows: A list of all shelf_rows in the warehouse
    aisle_locations: A list of all positions in the layout corresponding to an aisle tile
    bin_size: The amount of shelves to be grouped together to a single picking place (pick_location)
    distances: A dictionary containing the distances between each pair of pick_locations (including the distance to depot)
    layout_matrix: A matrix representation of the layout. Structured as followed:
                    depot = -10
                    walkways = -2
                    aisles = -1
                    shelves = 0
    pick_locations: A list of all picking places in the warehouse
    """


    def __init__(self, shape, depot_location, initial_shelves, shelf_rows, aisle_locations, bin_size=5):
        self.shape = shape
        self.depot_location = depot_location
        self.initial_shelves = initial_shelves # ugly but convenient
        self.shelf_rows = shelf_rows
        self.aisle_locations = aisle_locations
        self.bin_size = bin_size

        self.distances = defaultdict(dict)

        self.create_layout_matrix()

        self.set_pick_locations()

        self.shelves = []
        for shelf_row in self.shelf_rows:
            for shelf in shelf_row.shelves:
                self.shelves.append(shelf)

        self.generate_distance_graph()

        self.set_shelf_order()

        self.find_shelves_closest_to_depot()


    def create_layout_matrix(self):
        """
        Generates a layout matrix based on the shelf and aisle locations
        """
        self.layout_matrix = np.zeros(self.shape) - 2
        self.layout_matrix[self.depot_location] = -10

        for aisle_location in self.aisle_locations:
            self.layout_matrix[aisle_location] = -1

        for shelf in self.initial_shelves:
            shelf_location = shelf.location
            self.layout_matrix[shelf_location] = 0


    def set_pick_locations(self):
        """
        Sets pick locations for all shelves 
        """
        for shelf_row in self.shelf_rows:
            # shelf_row.set_dimensions()
            shelf_row.set_pick_locations(self)
            
        for row1, row2 in itertools.combinations(self.shelf_rows, 2):
            for shelf1 in row1.shelves:
                for shelf2 in row2.shelves:
                    if shelf1.pick_location == shelf2.pick_location:
                        shelf1.height += shelf2.height
                        row2.remove(shelf2)
                        if row2.is_empty():
                            self.shelf_rows.remove(row2)


                    

    def generate_distance_graph(self):
        """
        Generates distances between each combination of pick_locations and/or depot and stores them in the class variable called distances
        """

        class AstarNode:
            """
            Class for representing a node while performing astar search
            
            Class variables:
            ============================================================
            position: Coordinates on the form (x, y)
            parent: Parent node 
            g: Cost from start node to this node
            h: Value of the heuristic function for this node:
            f: g+h 
            """

            def __init__(self, position, parent=None):
                self.position = position
                self.parent = parent
                self.g = self.h = self.f = 0

            def __eq__(self, other):
                """
                Two nodes are defined to be equal if their positions are the same
                """
                return self.position == other.position

        def heuristic(node1, node2):
            """
            Heuristic function (Manhattan distance) for use within the astar search

            Input:
            ============================================================
            Two instances of the AstarNode class

            Returns:
            ============================================================
            The manhattan distance between the two nodes
            """
            return np.abs(node1.position[0] - node2.position[0]) + np.abs(node1.position[1] - node2.position[1])


        def astar_search(start_position, end_position):
            """
            Calculates the shortest path from start_position to end_position, without crossing any shelf tiles

            Input:
            ============================================================
            Two locations in the layout (start_prosition and end_position)
            
            Returns:
            ============================================================
            The length of the shortest path between the two points
            """
            grid = self.layout_matrix
            start = AstarNode(start_position)
            end = AstarNode(end_position)

            open = []
            closed = []

            open.append(start)

            while(len(open) > 0):
                current = open[0]
                current_i = 0
                for i, node in enumerate(open):
                    if node.f < current.f:
                        current = node
                        current_i = i
                
                open.pop(current_i)
                closed.append(current)

                # Check if goal is found
                if current == end:
                    path = []
                    while current is not None:
                        path.append(current.position)
                        current = current.parent
                    return len(path) - 1

                # If the current node is a picking point, only check left/right
                if grid[current.position] == -1:
                    for move in [1, -1]:
                        new_position = (current.position[0], current.position[1] + move)

                        # Check if new position is within the grid
                        if 0 <= new_position[0] < grid.shape[0] and 0 <= new_position[1] < grid.shape[1]:

                            # Check if new position is a valid tile (not a shelf)
                            if grid[new_position] < 0:
                                new_node = AstarNode(new_position, current)

                                # Check if new node is already visited
                                if new_node in closed:
                                    continue

                                # Calculate and set scores for new node
                                new_node.g = current.g + 1 
                                new_node.h = heuristic(new_node, end)
                                new_node.f = new_node.g + new_node.h

                                # Check if new node is already in queue
                                if new_node in open:
                                    continue

                                open.append(new_node)
                else:
                    # Test all possible moves (right, left, down, up)
                    for move in [(0,1), (0,-1), (1,0), (-1,0)]:
                        new_position = (current.position[0] + move[0], current.position[1] + move[1])

                        # Check if new position is within the grid
                        if 0 <= new_position[0] < grid.shape[0] and 0 <= new_position[1] < grid.shape[1]:

                            # Check if new position is a valid tile (not a shelf)
                            if grid[new_position] < 0:
                                new_node = AstarNode(new_position, current)

                                # Check if new node is already visited
                                if new_node in closed:
                                    continue
                                
                                # Calculate and set scores for new node
                                new_node.g = current.g + 1 
                                new_node.h = heuristic(new_node, end)
                                new_node.f = new_node.g + new_node.h

                                # Check if new node is already in queue
                                if new_node in open:
                                    continue

                                open.append(new_node)
    
        self.pick_locations = []
        for shelf in self.shelves:
            self.pick_locations.append(shelf.pick_location)
        
        self.n_pick_locations = len(self.pick_locations)
        self.depot_distances = np.zeros((self.n_pick_locations))
        self.distance_matrix = np.zeros((self.n_pick_locations, self.n_pick_locations))

        sti = time.time()
        
        def compute_distances(n,pick_locs, i):
            start = pick_locs[i]
            distances = []
            for j in range(i+1, n):
                end = pick_locs[j]
                dist = astar_search(start, end) # + np.random.normal(loc=1e-2, scale=1e-4)
                distances.append(dist)
            
            return distances

        distances = Parallel(n_jobs=4)(delayed(compute_distances)(self.n_pick_locations, self.pick_locations, i) for i in range(self.n_pick_locations))
        
        for i, l in enumerate(distances):
            # distances are only one way so we flip indices to populate entire matrix
            self.distance_matrix[i,i+1:] = np.asarray(l)
            self.distance_matrix[i+1:,i] = np.asarray(l)

            for k, dist in enumerate(l):
                j = k + i + 1
                self.distances[self.pick_locations[i]][self.pick_locations[j]] = dist
                self.distances[self.pick_locations[j]][self.pick_locations[i]] = dist

        def compute_depot_distance(depot_loc, loc):
            dist = astar_search(depot_loc, loc) # + np.random.normal(loc=1e-2, scale=1e-4)
            return dist

        depot_distances = Parallel(n_jobs=4)(delayed(compute_depot_distance)(self.depot_location, self.pick_locations[i]) for i in range(self.n_pick_locations))
        
        for i in range(len(depot_distances)):
            p2 = self.pick_locations[i]
            self.distances[self.depot_location][p2] = depot_distances[i]
            self.distances[p2][self.depot_location] = depot_distances[i]
            self.distances[p2][p2] = 0

        self.shelves.sort(key=lambda x: self.distances[self.depot_location][x.pick_location])

        shelf_index = ShelfIndex()
        for shelf in self.shelves:
            shelf.index = shelf_index.next()


    # Doesn't work for irregular layouts!
    def set_shelf_order(self):
        horizontally_aligned_shelves = defaultdict(list)
        vertically_aligned_shelves = defaultdict(list)

        for shelf in self.shelves:
            horizontally_aligned_shelves[shelf.pick_location[0]].append(shelf)
            vertically_aligned_shelves[shelf.pick_location[1]].append(shelf)
        
        for x, shelves in horizontally_aligned_shelves.items():
            shelves_sorted = sorted(shelves, key=lambda item: item.pick_location[1])
            horizontally_aligned_shelves[x] = shelves_sorted
            for i in range(len(shelves_sorted) - 1):
                shelves_sorted[i].connected_shelves.append(shelves_sorted[i+1])
                shelves_sorted[i+1].connected_shelves.append(shelves_sorted[i])

        for y, shelves in vertically_aligned_shelves.items():
            shelves_sorted = sorted(shelves, key=lambda item: item.pick_location[0])
            horizontally_aligned_shelves[y] = shelves_sorted
            for i in range(len(shelves_sorted) - 1):
                shelves_sorted[i].connected_shelves.append(shelves_sorted[i+1])
                shelves_sorted[i+1].connected_shelves.append(shelves_sorted[i])


    def find_shelves_closest_to_depot(self):
        """
        Looks for the shelves that are the closest to depot, to be used for starting shelves when running the greedy algorithm. 
        """
        min_dist = np.infty
        closest_shelves = []
        for shelf in self.shelves:
            dist = self.distances[shelf.pick_location][self.depot_location]
            if dist == min_dist:
                closest_shelves.append(shelf)            
            elif dist < min_dist:
                closest_shelves = [shelf]
                min_dist = dist
            
        self.closest_shelves = closest_shelves

                
        


