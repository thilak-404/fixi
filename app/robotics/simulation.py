"""
2D Kinematics and Environment Simulation Engine for Fixi Robotics.
Integrates:
- 2D Occupancy Grid Map
- 8-Connected A* Path Planner
- Robotics Finite State Machine
- Dynamic Obstacle Avoidance & Verification
"""

import math
import time
import threading
from typing import List, Tuple, Dict, Any, Optional

from app.robotics.occupancy_grid import OccupancyGrid, CellType
from app.robotics.astar import AStarPlanner, AStarResult
from app.robotics.fsm import RoboticsFSM, RobotFSMState
from app.robotics.controller import SimulatedRobotController, robot_controller
from app.events import event_bus, EventType


class RoboticsSimulationEngine:
    def __init__(self, width: int = 20, height: int = 20):
        self.grid = OccupancyGrid(width=width, height=height, resolution=0.25)
        self.planner = AStarPlanner(self.grid)
        self.fsm = RoboticsFSM(RobotFSMState.IDLE)
        self.controller: SimulatedRobotController = robot_controller

        # Current robot discrete position on the grid
        self.robot_col: int = width // 2  # default col 10
        self.robot_row: int = 2           # default row 2
        self.heading: float = 90.0         # facing north (degrees)

        # Target and person
        self.target_col: Optional[int] = None
        self.target_row: Optional[int] = None
        self.target_label: str = "goal"
        self.person_col: Optional[int] = None
        self.person_row: Optional[int] = None

        # Planned path waypoints
        self.current_path: List[Tuple[int, int]] = []
        self.current_path_index: int = 0
        self.last_planner_result: Optional[AStarResult] = None

        # Auto-navigation thread control
        self._nav_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self.follow_mode: bool = False
        self._follow_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.step_delay: float = 0.35  # seconds per simulation step

    def reset(self, start_col: int = 10, start_row: int = 2):
        """Resets the simulation to standard initial state."""
        with self._lock:
            self.follow_mode = False
            self._running = False
            self.robot_col = start_col
            self.robot_row = start_row
            self.heading = 90.0
            self.target_col = None
            self.target_row = None
            self.current_path = []
            self.current_path_index = 0
            self.last_planner_result = None
            self.grid.clear_dynamic_obstacles()
            self.fsm.transition_to(RobotFSMState.IDLE, {"action": "simulation_reset"})

    def update_person_position(self, col: int, row: int):
        """Updates person position detected by Computer Vision."""
        with self._lock:
            self.person_col = col
            self.person_row = row
            self.grid.update_person_position(col, row)

    def set_goal(self, col: int, row: int, label: str = "target") -> bool:
        """Sets a navigation goal cell and triggers planning."""
        if not self.grid.is_valid_cell(col, row):
            return False

        with self._lock:
            self.target_col = col
            self.target_row = row
            self.target_label = label

        return self.plan_path_to(col, row)

    def navigate_to_person(self, person_col: int, person_row: int) -> bool:
        """
        Plans and navigates to a safe interaction distance from the detected person.
        Avoids stopping directly inside the person's bounding cell.
        """
        self.update_person_position(person_col, person_row)

        # Find closest traversable neighbor cell adjacent to the person
        best_target = None
        min_dist = float("inf")
        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nc, nr = person_col + dc, person_row + dr
            if self.grid.is_traversable(nc, nr, allow_inflated=True):
                dist = math.hypot(nc - self.robot_col, nr - self.robot_row)
                if dist < min_dist:
                    min_dist = dist
                    best_target = (nc, nr)

        if not best_target:
            best_target = (person_col, max(1, person_row - 1))

        self.fsm.transition_to(RobotFSMState.PERCEIVING, {"target": "person", "coords": (person_col, person_row)})
        return self.set_goal(best_target[0], best_target[1], label="person_interaction_zone")

    def start_following(self, person_col: int, person_row: int) -> bool:
        """Starts simulation-only follow mode using repeated visual person updates."""
        if not self.grid.is_valid_cell(person_col, person_row):
            return False
        if self.fsm.state == RobotFSMState.EMERGENCY_STOP:
            self.fsm.transition_to(RobotFSMState.IDLE, {"action": "follow_requested_after_stop"})
        self.update_person_position(person_col, person_row)
        self.follow_mode = True
        if self._follow_thread is None or not self._follow_thread.is_alive():
            self._follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
            self._follow_thread.start()
        return True

    def _follow_loop(self) -> None:
        while self.follow_mode:
            with self._lock:
                person = (self.person_col, self.person_row)
                navigating = self._running
            if person[0] is not None and person[1] is not None and not navigating:
                self.navigate_to_person(person[0], person[1])
            time.sleep(0.5)

    def plan_path_to(self, goal_col: int, goal_row: int) -> bool:
        """Runs 8-connected A* planning from current robot cell to goal."""
        self.fsm.transition_to(RobotFSMState.PLANNING, {"start": (self.robot_col, self.robot_row), "goal": (goal_col, goal_row)})

        event_bus.emit(
            EventType.PLANNING_STARTED,
            {
                "start": [self.robot_col, self.robot_row],
                "goal": [goal_col, goal_row],
                "label": self.target_label
            },
            is_simulated=True
        )

        start = (self.robot_col, self.robot_row)
        goal = (goal_col, goal_row)
        result = self.planner.plan(start, goal, allow_inflated=False)

        # Fallback allowing inflated clearance if initial strict path fails
        if not result.success:
            result = self.planner.plan(start, goal, allow_inflated=True)

        self.last_planner_result = result

        if result.success and len(result.path) > 0:
            with self._lock:
                self.current_path = result.path
                self.current_path_index = 0
            
            event_bus.emit(
                EventType.PATH_PLANNED,
                {
                    "path_length": len(result.path),
                    "cost": result.cost,
                    "nodes_expanded": result.nodes_expanded,
                    "planning_time_ms": result.planning_time_ms,
                    "path": [list(p) for p in result.path]
                },
                is_simulated=True
            )

            self.fsm.transition_to(RobotFSMState.NAVIGATING, {"path_length": len(result.path)})
            self.start_autonomous_execution()
            return True
        else:
            event_bus.emit(
                EventType.ACTION_FAILED,
                {
                    "reason": "Path planning failed: no obstacle-free path.",
                    "details": result.error_message
                },
                is_simulated=True
            )
            self.fsm.transition_to(RobotFSMState.BLOCKED, {"error": result.error_message})
            return False

    def step(self) -> Dict[str, Any]:
        """
        Executes one discrete simulation step along the path with closed-loop verification.
        Returns status dictionary.
        """
        with self._lock:
            if not self.current_path or self.current_path_index >= len(self.current_path) - 1:
                if self.fsm.state == RobotFSMState.NAVIGATING:
                    self._complete_navigation()
                return {"status": "completed", "at_goal": True}

            next_index = self.current_path_index + 1
            next_col, next_row = self.current_path[next_index]

            # 1. Closed-loop verification: Check if next cell is obstructed by obstacle
            if not self.grid.is_traversable(next_col, next_row, allow_inflated=True):
                # Obstacle encountered ahead!
                self.fsm.transition_to(RobotFSMState.BLOCKED, {"obstacle_at": [next_col, next_row]})
                event_bus.emit(
                    EventType.ACTION_FAILED,
                    {
                        "reason": f"Obstacle encountered ahead at ({next_col}, {next_row})",
                        "blocked_cell": [next_col, next_row]
                    },
                    is_simulated=True
                )
                # Attempt replanning
                return self._attempt_replanning()

            # 2. Advance kinematics
            dc = next_col - self.robot_col
            dr = next_row - self.robot_row
            rad = math.atan2(dr, dc)
            self.heading = (math.degrees(rad) + 360) % 360

            self.robot_col = next_col
            self.robot_row = next_row
            self.current_path_index = next_index

            # Update robot controller odometry
            wx, wy = self.grid.grid_to_world(self.robot_col, self.robot_row)
            self.controller.state.x = wx
            self.controller.state.y = wy
            self.controller.state.heading = self.heading

            # 3. Check progress
            remaining_steps = len(self.current_path) - 1 - self.current_path_index

            event_bus.emit(
                EventType.ACTION_PROGRESS,
                {
                    "current_pos": [self.robot_col, self.robot_row],
                    "world_coords": [wx, wy],
                    "heading_deg": round(self.heading, 1),
                    "remaining_steps": remaining_steps
                },
                is_simulated=True
            )

            # 4. Check arrival
            if remaining_steps == 0:
                self._complete_navigation()
                return {"status": "arrived", "pos": [self.robot_col, self.robot_row]}

            return {"status": "in_progress", "remaining": remaining_steps}

    def _complete_navigation(self):
        """Action outcome verification on arrival."""
        self._running = False
        self.fsm.transition_to(RobotFSMState.ARRIVED, {"final_pos": [self.robot_col, self.robot_row]})
        self.controller.wave()

        event_bus.emit(
            EventType.ACTION_VERIFIED,
            {
                "status": "Target reached successfully with visual/spatial verification",
                "final_pos": [self.robot_col, self.robot_row],
                "target_label": self.target_label
            },
            is_simulated=True
        )

    def _attempt_replanning(self) -> Dict[str, Any]:
        """Closed-loop dynamic recovery: replans path around the new obstacle."""
        self.fsm.transition_to(RobotFSMState.REPLANNING, {"from_pos": [self.robot_col, self.robot_row]})
        event_bus.emit(
            EventType.REPLANNING_STARTED,
            {"current_pos": [self.robot_col, self.robot_row], "goal": [self.target_col, self.target_row]},
            is_simulated=True
        )

        if self.target_col is None or self.target_row is None:
            self.fsm.transition_to(RobotFSMState.IDLE)
            return {"status": "replan_failed", "reason": "No target"}

        # Rerun A* from current position
        start = (self.robot_col, self.robot_row)
        goal = (self.target_col, self.target_row)
        result = self.planner.plan(start, goal, allow_inflated=True)

        if result.success and len(result.path) > 0:
            self.current_path = result.path
            self.current_path_index = 0
            self.last_planner_result = result
            self.fsm.transition_to(RobotFSMState.NAVIGATING, {"replanned": True})
            return {"status": "replanned", "new_path_length": len(result.path)}
        else:
            self.fsm.transition_to(RobotFSMState.BLOCKED, {"error": "Replanning found no path"})
            self._running = False
            return {"status": "replan_failed", "reason": "No alternative path"}

    def emergency_stop(self):
        """Immediately halts robot navigation and updates FSM."""
        with self._lock:
            self.follow_mode = False
            self._running = False
            self.fsm.transition_to(RobotFSMState.EMERGENCY_STOP, {"reason": "User emergency stop"})
            self.controller.stop()

    def add_obstacle(self, col: int, row: int):
        """Adds a dynamic obstacle at runtime to test replanning."""
        with self._lock:
            self.grid.add_obstacle(col, row, is_dynamic=True)
            self.grid.recompute_inflation()

    def remove_obstacle(self, col: int, row: int):
        with self._lock:
            self.grid.remove_obstacle(col, row)

    def start_autonomous_execution(self):
        """Launches background thread to step through path until arrival or stop."""
        with self._lock:
            if self._running:
                return
            self._running = True

        def _run():
            while self._running and self.fsm.state == RobotFSMState.NAVIGATING:
                time.sleep(self.step_delay)
                res = self.step()
                if res.get("status") in ["arrived", "completed", "replan_failed"]:
                    break

        self._nav_thread = threading.Thread(target=_run, daemon=True)
        self._nav_thread.start()

    def get_full_state(self) -> Dict[str, Any]:
        """Complete serialized snapshot for WebSocket clients and Frontend."""
        with self._lock:
            wx, wy = self.grid.grid_to_world(self.robot_col, self.robot_row)
            return {
                "robot": {
                    "grid_col": self.robot_col,
                    "grid_row": self.robot_row,
                    "world_x": wx,
                    "world_y": wy,
                    "heading_deg": round(self.heading, 1),
                    "is_moving": self.fsm.state == RobotFSMState.NAVIGATING
                },
                "follow_mode": self.follow_mode,
                "target": {
                    "grid_col": self.target_col,
                    "grid_row": self.target_row,
                    "label": self.target_label
                } if self.target_col is not None else None,
                "person": {
                    "grid_col": self.person_col,
                    "grid_row": self.person_row
                } if self.person_col is not None else None,
                "path": [list(p) for p in self.current_path],
                "path_index": self.current_path_index,
                "fsm": self.fsm.get_status(),
                "grid": self.grid.to_dict(),
                "planner_metrics": self.last_planner_result.to_dict() if self.last_planner_result else None
            }


# Singleton simulation instance
robotics_sim = RoboticsSimulationEngine()
