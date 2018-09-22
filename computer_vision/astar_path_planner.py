import math

from astar.astar import AStar

class AstarPathPlanner(AStar):
    def __init__(self, map_image):
        self.map = map_image


    def heuristic_cost_estimate(self, current, goal):
        (x1, y1) = current
        (x2, y2) = goal
        return math.hypot(x2 - x1, y2 - y1)


    def distance_between(self, n1, n2):
        neighbor_x, neighbor_y = n2
        neighbor_height = self.map[neighbor_y, neighbor_x] # TODO: verify order
        return 1 + neighbor_height


    def neighbors(self, node):
        curr_x, curr_y = node
        def is_free(x, y):
            if 0 <= x < self.map.shape[1] and 0 <= y < self.map.shape[0]:
                if self.map[y, x] != 1: # TODO: verify order
                    return True
            return False
        return [(x, y) for (x, y) in [(curr_x, curr_y - 1), (curr_x, curr_y + 1), (curr_x - 1, curr_y), (curr_x + 1, curr_y)] if is_free(x, y)]