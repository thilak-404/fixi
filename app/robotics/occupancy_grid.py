"""
2D Occupancy Grid Map representation for Fixi Robotics Module.
Represents discrete spatial cells for obstacle mapping, safety inflation,
and spatial grounding of people detected by the vision module.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class CellType:
    EMPTY = 0
    OBSTACLE = 1
    ROBOT = 2
    PERSON = 3
    TARGET = 4
    INFLATED = 5


class OccupancyGrid:
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        resolution: float = 0.25,  # 0.25 meters per cell (5m x 5m total area)
        origin_x: float = 0.0,
        origin_y: float = 0.0
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        # 0: empty, 1: obstacle, 5: inflated obstacle
        self.grid = np.zeros((height, width), dtype=np.int32)
        
        # Keep track of static and dynamic obstacle coordinates (col, row)
        self.static_obstacles: set[Tuple[int, int]] = set()
        self.dynamic_obstacles: set[Tuple[int, int]] = set()
        self.person_cells: set[Tuple[int, int]] = set()

        # Initialize default academic room layout with boundaries and a few obstacles
        self._init_default_environment()

    def _init_default_environment(self):
        """Builds perimeter walls and typical indoor furniture obstacles."""
        # Perimeter boundaries
        for c in range(self.width):
            self.add_obstacle(c, 0, is_dynamic=False)
            self.add_obstacle(c, self.height - 1, is_dynamic=False)
        for r in range(self.height):
            self.add_obstacle(0, r, is_dynamic=False)
            self.add_obstacle(self.width - 1, r, is_dynamic=False)

        # Center obstacle (e.g. coffee table / partition)
        for c in range(7, 13):
            self.add_obstacle(c, 10, is_dynamic=False)
            
        # Side obstacle (e.g. bookshelf)
        for r in range(4, 8):
            self.add_obstacle(4, r, is_dynamic=False)

        self.recompute_inflation()

    def is_valid_cell(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def is_traversable(self, col: int, row: int, allow_inflated: bool = False) -> bool:
        if not self.is_valid_cell(col, row):
            return False
        val = self.grid[row, col]
        if val == CellType.OBSTACLE:
            return False
        if val == CellType.INFLATED and not allow_inflated:
            return False
        return True

    def add_obstacle(self, col: int, row: int, is_dynamic: bool = True):
        if self.is_valid_cell(col, row):
            if is_dynamic:
                self.dynamic_obstacles.add((col, row))
            else:
                self.static_obstacles.add((col, row))
            self.grid[row, col] = CellType.OBSTACLE

    def remove_obstacle(self, col: int, row: int):
        if (col, row) in self.dynamic_obstacles:
            self.dynamic_obstacles.remove((col, row))
        if (col, row) in self.static_obstacles:
            self.static_obstacles.remove((col, row))
        if self.is_valid_cell(col, row):
            self.grid[row, col] = CellType.EMPTY
        self.recompute_inflation()

    def clear_dynamic_obstacles(self):
        for col, row in list(self.dynamic_obstacles):
            self.grid[row, col] = CellType.EMPTY
        self.dynamic_obstacles.clear()
        self.recompute_inflation()

    def update_person_position(self, col: int, row: int):
        """Updates the registered vision person location on the grid."""
        self.person_cells.clear()
        if self.is_valid_cell(col, row):
            self.person_cells.add((col, row))

    def recompute_inflation(self, clearance: int = 1):
        """
        Inflates obstacles by `clearance` cells to create a safety margin for the robot.
        Does not overwrite primary OBSTACLE cells.
        """
        # Reset grid from base obstacles
        self.grid.fill(CellType.EMPTY)
        all_obs = self.static_obstacles | self.dynamic_obstacles
        for c, r in all_obs:
            if self.is_valid_cell(c, r):
                self.grid[r, c] = CellType.OBSTACLE

        # Compute inflation zones
        if clearance > 0:
            for c, r in all_obs:
                for dc in range(-clearance, clearance + 1):
                    for dr in range(-clearance, clearance + 1):
                        nc, nr = c + dc, r + dr
                        if self.is_valid_cell(nc, nr) and self.grid[nr, nc] == CellType.EMPTY:
                            self.grid[nr, nc] = CellType.INFLATED

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Converts metric coordinates (x, y) to grid coordinates (col, row)."""
        col = int(np.clip(round((x - self.origin_x) / self.resolution), 0, self.width - 1))
        row = int(np.clip(round((y - self.origin_y) / self.resolution), 0, self.height - 1))
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Converts grid coordinates (col, row) to metric world center (x, y)."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return round(x, 2), round(y, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes grid state for WebSockets and UI rendering."""
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "grid": self.grid.tolist(),
            "static_obstacles": [list(o) for o in self.static_obstacles],
            "dynamic_obstacles": [list(o) for o in self.dynamic_obstacles],
            "person_cells": [list(p) for p in self.person_cells]
        }
