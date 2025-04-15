from typing import Callable, List

from typesafe_llm.parser.parser_base import IncrementalParsingState


def assert_partial(states):
    assert states, "States should not be empty"


def assert_strict_partial(states):
    assert states, "States should not be empty"
    assert all(not state.accept for state in states), "State should not accept"


def assert_weak_full(states):
    assert states, "States should not be empty"
    assert any(state.accept for state in states), "some State should accept"


def assert_reject(states):
    assert not states, "States should be empty"


def assert_strict_partial_or_reject(states):
    assert all(not state.accept for state in states), "State should not accept"


def assert_just_before_reject_generic(
    parse_program: Callable[[str], List[IncrementalParsingState]],
    incremental_parse: Callable[
        [List[IncrementalParsingState], str], List[IncrementalParsingState]
    ],
    program: str,
):
    program_before = program[:-1]
    states = parse_program(program_before)
    assert_partial(states)
    states = incremental_parse(states, program[-1])
    assert_reject(states)
