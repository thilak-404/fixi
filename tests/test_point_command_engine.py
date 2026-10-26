from app.robotics.commands import PointCommandParser
from app.robotics.point_command_engine import PointCommandEngine, PointCommandState
from app.robotics.pointing import GestureClass, GestureObservation
from app.robotics.simulation import RoboticsSimulationEngine
from app.robotics.target_selection import ObjectDetection


def stable_point():
    return GestureObservation(GestureClass.POINTING, 0.9, (1.0, 0.0), (0.0, 0.0), 4, 0.95)


def test_engine_requires_clarification_for_multiple_targets():
    engine = PointCommandEngine(RoboticsSimulationEngine())
    status = engine.observe(stable_point(), [
        ObjectDetection("a", "cup", 0.9, (90, -10, 20, 20)),
        ObjectDetection("b", "bottle", 0.9, (90, 10, 20, 20)),
    ])

    assert status.state == PointCommandState.WAITING_FOR_CLARIFICATION
    assert status.target_id is None


def test_engine_plans_only_after_authorized_command():
    engine = PointCommandEngine(RoboticsSimulationEngine())
    engine.observe(stable_point(), [ObjectDetection("b", "bottle", 0.95, (100, -10, 20, 20))])

    denied = engine.accept_command(PointCommandParser().parse("Go there", "user_001"), "user_001", 0.5, 1, True)
    assert denied.state == PointCommandState.ERROR

    accepted = engine.accept_command(PointCommandParser().parse("Go there", "user_001"), "user_001", 0.95, 1, True)
    assert accepted.state == PointCommandState.EXECUTING


def test_stop_is_immediate():
    engine = PointCommandEngine(RoboticsSimulationEngine())
    status = engine.accept_command(PointCommandParser().parse("Stop"), None, 0, 2, False)

    assert status.state == PointCommandState.STOPPED
