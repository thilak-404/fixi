"""
Tests for Fixi Robotics Integration Layer (Section 20).
"""

from app.robotics.controller import SimulatedRobotController


def test_robot_controller_movements():
    robot = SimulatedRobotController()
    assert robot.state.x == 0.0
    assert robot.state.y == 0.0

    robot.move_forward(speed=2.0, duration=1.0)
    assert robot.state.x == 2.0
    assert robot.state.y == 0.0

    robot.turn_left(90.0)
    assert robot.state.heading == 90.0

    robot.move_forward(speed=1.0, duration=2.0)
    assert abs(robot.state.y - 2.0) < 0.01

    robot.stop()
    assert robot.state.is_moving is False


def test_robot_look_at_person():
    robot = SimulatedRobotController()
    # Person on the right side of frame (cx = 500 in 640x480 frame)
    bbox = (450, 100, 100, 150)
    robot.look_at_person(bbox, frame_size=(640, 480))

    # Should pan right (positive pan angle)
    assert robot.state.head_pan > 0


def test_robot_wave_gesture():
    robot = SimulatedRobotController()
    ok = robot.wave()
    assert ok is True
    assert robot.state.last_action == "wave"
    assert len(robot.action_history) > 0
