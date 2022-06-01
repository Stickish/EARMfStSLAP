class Warehouse:
    """
    Class for representing a warehouse. TODO: Merge layout and warehouse into a single class

    Class variables:
    ============================================================
    layout: An instance of the layout class
    model: A model for solving the problem. Currently works for LinearProgramming, AbcModel, GeneticModel and Greedy
    """
    def __init__(self, layout, model):
        self.layout = layout
        self.model = model
    

    def optimize_locations(self):
        return self.model.optimize_locations()