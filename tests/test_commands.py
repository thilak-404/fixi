from app.robotics.commands import CommandIntent, PointCommandParser, authorize_point_command


def test_supported_commands_are_structured():
    parser = PointCommandParser()

    command = parser.parse("Fixi, go to that bottle", user_id="user_001")
    assert command.intent == CommandIntent.GO
    assert command.target_reference == "bottle"
    assert command.user_id == "user_001"

    assert parser.parse("What am I pointing at?").intent == CommandIntent.IDENTIFY_TARGET
    assert parser.parse("Come back to the starting point").intent == CommandIntent.RETURN_HOME
    assert parser.parse("Stop").intent == CommandIntent.STOP


def test_movement_requires_authorized_identity():
    parser = PointCommandParser()
    command = parser.parse("Go there", user_id="user_001")

    allowed, _ = authorize_point_command(command, "user_001", 0.9, 1, True)
    denied, reason = authorize_point_command(command, "user_001", 0.5, 1, True)

    assert allowed
    assert not denied
    assert "confidence" in reason


def test_stop_bypasses_identity_gate():
    command = PointCommandParser().parse("Stop")

    allowed, _ = authorize_point_command(command, None, 0.0, 3, False)

    assert allowed
