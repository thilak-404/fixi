"""
Unit tests for Memory Vault variable verification, non-empty validation,
template formatting, and natural fallbacks for missing facts.
"""

import pytest
from app.ai.response import safe_get_fact, safe_format_template, resolve_personal_memory_query


def test_safe_get_fact_validation():
    # 1. Valid non-empty fact
    facts = {"outdoor_game": "Football", "sport": "Tennis"}
    assert safe_get_fact(facts, "outdoor_game") == "Football"
    assert safe_get_fact(facts, "sport") == "Tennis"

    # 2. Empty string, whitespace, and None
    empty_facts = {
        "outdoor_game": "",
        "sport": "   ",
        "hobby": None,
        "placeholder": "{outdoor_game}"
    }
    assert safe_get_fact(empty_facts, "outdoor_game") is None
    assert safe_get_fact(empty_facts, "sport") is None
    assert safe_get_fact(empty_facts, "hobby") is None
    assert safe_get_fact(empty_facts, "placeholder") is None

    # 3. Fallback across multiple keys
    multi_facts = {"outdoor_game": "  ", "sport": "Cricket"}
    assert safe_get_fact(multi_facts, "outdoor_game", "sport") == "Cricket"


def test_safe_format_template_fallbacks():
    # 1. Normal interpolation with valid variable
    res = safe_format_template("You love {outdoor_game}, {user_name}!", {"outdoor_game": "Cricket", "user_name": "Arun"})
    assert res == "You love Cricket, Arun!"
    assert "  " not in res

    # 2. Missing {outdoor_game} must fallback to natural 'your favorite sport' without leaving a blank space
    res2 = safe_format_template("You love {outdoor_game}, {user_name}!", {"user_name": "Arun"})
    assert res2 == "You love your favorite sport, Arun!"
    assert "  " not in res2
    assert "{outdoor_game}" not in res2

    # 3. Empty string {outdoor_game} must NOT leave empty blank space
    res3 = safe_format_template("You love {outdoor_game}, {user_name}!", {"outdoor_game": "   ", "user_name": "Arun"})
    assert res3 == "You love your favorite sport, Arun!"
    assert "  " not in res3

    # 4. Raw template placeholder as value must trigger natural fallback
    res4 = safe_format_template("You love {outdoor_game}, {user_name}!", {"outdoor_game": "{outdoor_game}", "user_name": "Arun"})
    assert res4 == "You love your favorite sport, Arun!"


def test_outdoor_game_query_resolution():
    # 1. User has outdoor_game in memory
    facts_with_outdoor = {"outdoor_game": "Football"}
    ans1 = resolve_personal_memory_query("What is my outdoor game?", "Arun", facts_with_outdoor, [])
    assert ans1 is not None
    assert "Football" in ans1
    assert "  " not in ans1

    # 2. User has NO outdoor_game in memory (variable missing)
    # Must use natural fallback 'your favorite sport' or ask directly to confirm
    facts_empty = {}
    ans2 = resolve_personal_memory_query("What is my outdoor game?", "Arun", facts_empty, [])
    assert ans2 is not None
    assert "your favorite sport" in ans2.lower()
    assert "  " not in ans2
    assert "{outdoor_game}" not in ans2

    # 3. User has outdoor_game as empty string ""
    facts_blank = {"outdoor_game": "   "}
    ans3 = resolve_personal_memory_query("What outdoor game do I play?", "Arun", facts_blank, [])
    assert ans3 is not None
    assert "your favorite sport" in ans3.lower()
    assert "  " not in ans3

    # 4. User has general sport saved, but not specific outdoor game
    facts_sport = {"sport": "Cricket"}
    ans4 = resolve_personal_memory_query("Do you know my outdoor game?", "Arun", facts_sport, [])
    assert ans4 is not None
    assert "Cricket" in ans4
    assert "  " not in ans4


def test_general_query_variable_non_empty_verification():
    # Test age query with empty string
    ans_age = resolve_personal_memory_query("What is my age?", "Arun", {"age": ""}, [])
    assert ans_age == "I don't know your age yet, Arun. How old are you?"

    # Test games query with empty string
    ans_game = resolve_personal_memory_query("What games do I like to play?", "Arun", {"game": " "}, [])
    assert "favorite sport or game" in ans_game.lower()
    assert "  " not in ans_game

    # Test generic attribute query when missing
    ans_food = resolve_personal_memory_query("What is my favorite food?", "Arun", {}, [])
    assert ans_food is not None
    assert "favorite food" in ans_food
    assert "  " not in ans_food
