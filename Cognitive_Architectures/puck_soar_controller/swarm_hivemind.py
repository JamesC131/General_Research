import math

class SwarmHivemind:
    
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.map = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.targets = {}  # robot_id → (x, y)
        self.exploration_radius = 2.0
        self.growth_rate = 0.01

    def update_position(self, robot_id, x, y):
        gx, gy = int(x), int(y)

        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            self.map[gx][gy] += 1

    def find_frontier(self):
        best = None
        best_score = -1

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                score = self.map[x][y]

                # prefer unvisited + edge expansion
                if score < best_score or best is None:
                    best = (x, y)
                    best_score = score

        return best

    def assign_target(self, robot_id, current_pos):
        target = self.find_frontier()
        self.targets[robot_id] = target
        return target
    
    def distributed_search(self, robot_id, current_pos):
        # Example of a more organized search pattern (e.g., spiral)
        x, y = current_pos
        angle = math.atan2(y - self.grid_size / 2, x - self.grid_size / 2)
        radius = math.hypot(x - self.grid_size / 2, y - self.grid_size / 2)

        # Move in a spiral pattern
        radius += self.growth_rate
        new_x = self.grid_size / 2 + radius * math.cos(angle)
        new_y = self.grid_size / 2 + radius * math.sin(angle)

        return int(new_x), int(new_y)