"""
Unit and Integration Tests for Fixi Intelligent Robotics & Closed-Loop Planning:
1. Occupancy Grid representation and obstacle inflation.
2. 8-Connected A* Search, Octile heuristic, and diagonal corner-cutting prevention.
3. Robotics FSM state transitions and safety stop.
4. Dynamic obstacle detection, collision prevention, and dynamic replanning.
5. Voice/Text intent classification (robotics navigation vs memory conversation).
6. Academic benchmarks execution.
"""

import pytest
from app.robotics.occupancy_grid import OccupancyGrid, CellType
from app.robotics.astar import AStarPlanner
from app.robotics.fsm import RoboticsFSM, RobotFSMState
from app.robotics.simulation import RoboticsSimulationEngine
from app.robotics.benchmark import run_robotics_benchmarks
from app.ai.intent import IntentClassifier, IntentCategory, RoboticsActionType


def test_occupancy_grid_and_safety_inflation():
    grid = OccupancyGrid(width=15, height=15, resolution=0.25)
    grid.static_obstacles.clear()
    grid.clear_dynamic_obstacles()

    # Add single obstacle at center (7, 7)
    grid.add_obstacle(7, 7, is_dynamic=True)
    grid.recompute_inflation(clearance=1)

    # Obstacle itself must be non-traversable
    assert not grid.is_traversable(7, 7, allow_inflated=True)
    assert not grid.is_traversable(7, 7, allow_inflated=False)

    # Adjacent cells (e.g. 7, 8) must be marked as INFLATED
    assert grid.grid[8, 7] == CellType.INFLATED
    # With allow_inflated=False, inflated cells are blocked for safety
    assert not grid.is_traversable(7, 8, allow_inflated=False)
    # With allow_inflated=True, soft traversability is allowed
    assert grid.is_traversable(7, 8, allow_inflated=True)

    # Distant cell (2, 2) is completely EMPTY
    assert grid.grid[2, 2] == CellType.EMPTY
    assert grid.is_traversable(2, 2, allow_inflated=False)


def test_astar_corner_cutting_prevention():
    grid = OccupancyGrid(width=10, height=10)
    grid.static_obstacles.clear()
    grid.clear_dynamic_obstacles()

    # Create diagonal obstacle pinch point:
    # Obstacle at (4, 5) and (5, 4)
    # A path from (4, 4) to (5, 5) MUST NOT squeeze through diagonally!
    grid.add_obstacle(4, 5, is_dynamic=False)
    grid.add_obstacle(5, 4, is_dynamic=False)
    grid.recompute_inflation(clearance=0)

    planner = AStarPlanner(grid)
    res = planner.plan((4, 4), (5, 5), allow_inflated=True)

    assert res.success is True
    # The path must route around, not jump diagonally directly across the pinch point
    for i in range(len(res.path) - 1):
        c1, r1 = res.path[i]
        c2, r2 = res.path[i+1]
        assert not ((c1 == 4 and r1 == 4 and c2 == 5 and r2 == 5) or
                    (c1 == 5 and r1 == 5 and c2 == 4 and r2 == 4))


def test_fsm_state_machine_transitions():
    fsm = RoboticsFSM(RobotFSMState.IDLE)
    assert fsm.state == RobotFSMState.IDLE

    fsm.transition_to(RobotFSMState.PERCEIVING, {"target": "user"})
    assert fsm.state == RobotFSMState.PERCEIVING

    fsm.transition_to(RobotFSMState.PLANNING)
    assert fsm.state == RobotFSMState.PLANNING

    fsm.transition_to(RobotFSMState.NAVIGATING)
    assert fsm.state == RobotFSMState.NAVIGATING

    # Emergency stop can be triggered at any point
    fsm.transition_to(RobotFSMState.EMERGENCY_STOP)
    assert fsm.state == RobotFSMState.EMERGENCY_STOP

    # Return to IDLE
    fsm.transition_to(RobotFSMState.IDLE)
    assert fsm.state == RobotFSMState.IDLE


def test_closed_loop_dynamic_replanning_on_obstacle():
    sim = RoboticsSimulationEngine(width=15, height=15)
    sim.reset(start_col=2, start_row=2)
    sim.grid.static_obstacles.clear()
    sim.grid.recompute_inflation(clearance=0)

    # Set goal to (10, 2) (straight horizontal line)
    success = sim.set_goal(10, 2)
    assert success is True
    assert sim.fsm.state == RobotFSMState.NAVIGATING
    assert len(sim.current_path) > 0

    # Execute first step successfully
    step1 = sim.step()
    assert step1["status"] == "in_progress"
    assert sim.robot_col == 3
    assert sim.robot_row == 2

    # Now inject a dynamic obstacle right in the robot's upcoming path at (4, 2)
    sim.add_obstacle(4, 2)

    # Advance next step: should detect collision ahead, transition to BLOCKED -> REPLANNING -> new path!
    step2 = sim.step()
    assert step2["status"] == "replanned"
    assert sim.fsm.state == RobotFSMState.NAVIGATING
    # Next waypoint in replanned path must detour around (4, 2)
    next_waypoint = sim.current_path[1]
    assert next_waypoint != (4, 2)


def test_intent_classification_robotics_and_memory():
    classifier = IntentClassifier()

    # 1. Navigation towards person
    cat, act, _ = classifier.classify("Move toward me please")
    assert cat == IntentCategory.ROBOTICS_COMMAND
    assert act == RoboticsActionType.MOVE_TOWARD_PERSON

    cat, act, _ = classifier.classify("come closer")
    assert cat == IntentCategory.ROBOTICS_COMMAND
    assert act == RoboticsActionType.MOVE_TOWARD_PERSON

    cat, act, _ = classifier.classify("Fixi, follow me")
    assert cat == IntentCategory.ROBOTICS_COMMAND
    assert act == RoboticsActionType.FOLLOW_PERSON

    # 2. Safety stop
    cat, act, _ = classifier.classify("Emergency stop immediately!")
    assert cat == IntentCategory.ROBOTICS_COMMAND
    assert act == RoboticsActionType.EMERGENCY_STOP

    cat, act, _ = classifier.classify("stop")
    assert cat == IntentCategory.ROBOTICS_COMMAND
    assert act == RoboticsActionType.EMERGENCY_STOP

    # 3. Wave gesture
    cat, act, _ = classifier.classify("wave at me")
    assert cat == IntentCategory.ROBOTICS_COMMAND
    assert act == RoboticsActionType.WAVE_GESTURE

    # 4. Conversational / Memory
    cat, act, _ = classifier.classify("What is my name?")
    assert cat == IntentCategory.CONVERSATIONAL
    assert act is None

    cat, act, _ = classifier.classify("I like playing football and tennis")
    assert cat == IntentCategory.CONVERSATIONAL
    assert act is None


def test_academic_benchmarks_execution():
    bench = run_robotics_benchmarks()
    assert "summary" in bench
    assert "scenarios" in bench
    assert len(bench["scenarios"]) == 3
    for sc in bench["scenarios"]:
        assert sc["success"] is True
        assert sc["nodes_expanded"] > 0
        assert sc["planning_time_ms"] >= 0.0
        assert sc["path_length"] > 0


def test_follow_can_restart_after_emergency_stop():
    from app.robotics.simulation import RoboticsSimulationEngine

    sim = RoboticsSimulationEngine()
    sim.emergency_stop()

    assert sim.start_following(10, 8) is True
    assert sim.follow_mode is True
    sim.emergency_stop()
