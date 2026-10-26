from app.robotics.pointing import GestureClass, GestureObservation
from app.robotics.target_selection import ObjectDetection, PointingTargetSelector


def pointing():
    return GestureObservation(GestureClass.POINTING, 0.9, (1.0, 0.0), (0.0, 0.0), 4, 0.95)


def test_selects_object_on_pointing_ray():
    selector = PointingTargetSelector(max_line_distance=30, ambiguity_margin=0.05)
    result = selector.select(pointing(), [ObjectDetection("bottle-1", "bottle", 0.9, (80, -10, 20, 20))])

    assert result.status == "selected"
    assert result.selected.detection.label == "bottle"


def test_rejects_ambiguous_candidates():
    selector = PointingTargetSelector(max_line_distance=100, ambiguity_margin=0.2)
    objects = [
        ObjectDetection("a", "cup", 0.9, (90, -10, 20, 20)),
        ObjectDetection("b", "bottle", 0.9, (90, 10, 20, 20)),
    ]

    result = selector.select(pointing(), objects)

    assert result.status == "ambiguous"
    assert result.selected is None
    assert len(result.candidates) == 2


def test_non_pointing_gesture_cannot_select():
    selector = PointingTargetSelector()
    gesture = GestureObservation(GestureClass.OPEN, 0.9, (1.0, 0.0), (0.0, 0.0), 4, 0.95)

    result = selector.select(gesture, [ObjectDetection("a", "cup", 0.9, (80, -10, 20, 20))])

    assert result.status == "none"


def test_unstable_pointing_gesture_cannot_select():
    selector = PointingTargetSelector()
    gesture = GestureObservation(GestureClass.POINTING, 0.49, (1.0, 0.0), (0.0, 0.0), 1, 0.95)

    result = selector.select(gesture, [ObjectDetection("a", "cup", 0.9, (80, -10, 20, 20))])

    assert result.status == "none"
