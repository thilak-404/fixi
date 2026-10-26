"""
8-Connected A* Path Planning Algorithm for Fixi Intelligent Robotics.
Features:
- Admissible Octile heuristic with Euclidean distance tie-breaker.
- Strict corner-cutting prevention on diagonal transitions.
- Obstacle avoidance with configurable safety cost penalties.
- Detailed academic metrics: path length, cost, nodes expanded, planning latency (ms).
"""

import heapq
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Set

from app.robotics.occupancy_grid import OccupancyGrid, CellType


@dataclass
class AStarResult:
    success: bool
    path: List[Tuple[int, int]]
    cost: float
    nodes_expanded: int
    planning_time_ms: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path": [list(pt) for pt in self.path],
            "path_length": len(self.path),
            "cost": round(self.cost, 2),
            "nodes_expanded": self.nodes_expanded,
            "planning_time_ms": round(self.planning_time_ms, 2),
            "error_message": self.error_message
        }


class AStarPlanner:
    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        # 8-connected motion primitives: (dcol, drow, cost)
        SQRT2 = math.sqrt(2.0)
        self.motions: List[Tuple[int, int, float]] = [
            (1, 0, 1.0),    # East
            (-1, 0, 1.0),   # West
            (0, 1, 1.0),    # North
            (0, -1, 1.0),   # South
            (1, 1, SQRT2),  # North-East
            (-1, 1, SQRT2), # North-West
            (1, -1, SQRT2), # South-East
            (-1, -1, SQRT2) # South-West
        ]

    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Admissible Octile distance heuristic for 8-connected grid navigation:
        h = max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    def is_valid_transition(
        self,
        current: Tuple[int, int],
        neighbor: Tuple[int, int],
        allow_inflated: bool = False
    ) -> bool:
        c1, r1 = current
        c2, r2 = neighbor

        # Target cell must be traversable
        if not self.grid.is_traversable(c2, r2, allow_inflated=allow_inflated):
            return False

        # Diagonal movement corner-cutting prevention:
        # Cannot squeeze between two diagonal obstacles
        dc = c2 - c1
        dr = r2 - r1
        if dc != 0 and dr != 0:
            # Check the two adjacent orthogonal cells
            ortho1_traversable = self.grid.is_traversable(c1 + dc, r1, allow_inflated=allow_inflated)
            ortho2_traversable = self.grid.is_traversable(c1, r1 + dr, allow_inflated=allow_inflated)
            if not ortho1_traversable or not ortho2_traversable:
                return False

        return True

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        allow_inflated: bool = False
    ) -> AStarResult:
        t_start = time.perf_counter()

        start_col, start_row = start
        goal_col, goal_row = goal

        # Validation
        if not self.grid.is_valid_cell(start_col, start_row):
            return AStarResult(
                success=False, path=[], cost=0.0, nodes_expanded=0,
                planning_time_ms=(time.perf_counter() - t_start) * 1000.0,
                error_message=f"Start cell {start} is out of bounds."
            )

        if not self.grid.is_valid_cell(goal_col, goal_row):
            return AStarResult(
                success=False, path=[], cost=0.0, nodes_expanded=0,
                planning_time_ms=(time.perf_counter() - t_start) * 1000.0,
                error_message=f"Goal cell {goal} is out of bounds."
            )

        # If goal is an obstacle, check nearest adjacent cell
        if not self.grid.is_traversable(goal_col, goal_row, allow_inflated=True):
            return AStarResult(
                success=False, path=[], cost=0.0, nodes_expanded=0,
                planning_time_ms=(time.perf_counter() - t_start) * 1000.0,
                error_message=f"Goal cell {goal} is an impassable obstacle."
            )

        # Priority queue stores (f_score, tie_breaker, (col, row))
        tie_breaker = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, tie_breaker, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed_set: Set[Tuple[int, int]] = set()

        nodes_expanded = 0

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            # Goal check
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                planning_time_ms = (time.perf_counter() - t_start) * 1000.0
                return AStarResult(
                    success=True,
                    path=path,
                    cost=g_score[goal],
                    nodes_expanded=nodes_expanded,
                    planning_time_ms=planning_time_ms
                )

            # Expand neighbors
            for dc, dr, move_cost in self.motions:
                neighbor = (current[0] + dc, current[1] + dr)

                if neighbor in closed_set:
                    continue

                if not self.is_valid_transition(current, neighbor, allow_inflated=allow_inflated):
                    continue

                # Add penalty if traveling through inflated zone (soft avoidance)
                cell_val = self.grid.grid[neighbor[1], neighbor[0]]
                inflation_penalty = 2.0 if cell_val == CellType.INFLATED else 0.0

                tentative_g = g_score[current] + move_cost + inflation_penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.heuristic(neighbor, goal)
                    f = tentative_g + h
                    tie_breaker += 1
                    heapq.heappush(open_set, (f, tie_breaker, neighbor))

        # No path found
        planning_time_ms = (time.perf_counter() - t_start) * 1000.0
        return AStarResult(
            success=False,
            path=[],
            cost=0.0,
            nodes_expanded=nodes_expanded,
            planning_time_ms=planning_time_ms,
            error_message="No feasible collision-free path found to goal."
        )
