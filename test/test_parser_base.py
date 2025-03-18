from typesafe_llm.parser.parser_base import (
    TerminalParserState,
    incremental_parse,
    IncrementalParserState,
)
from test.utils import assert_weak_full, assert_reject, assert_strict_partial

initial_ws_terminal = TerminalParserState(target_value="hello world\t !")
initial_ws_terminal_state = IncrementalParserState([initial_ws_terminal], "")


def test_accept_initial_char_terminal():
    states = initial_ws_terminal.parse_char("h")
    assert_strict_partial(states)


def test_dont_accept_ws_initial_char_terminal():
    states = initial_ws_terminal.parse_char(" ")
    assert_reject(states)


def test_accept_ws_optional_one():
    states = incremental_parse(initial_ws_terminal_state, "hello ")
    assert_strict_partial(states)


def test_accept_ws_optional_many():
    states = incremental_parse(initial_ws_terminal_state, "hello \t\n w")
    assert_strict_partial(states)


def test_accept_ws_optional_none():
    states = incremental_parse(initial_ws_terminal_state, "hellow")
    assert_strict_partial(states)


def test_accept_ws_required_one():
    states = incremental_parse(initial_ws_terminal_state, "hello world !")
    assert_weak_full(states)


def test_accept_ws_required_many():
    states = incremental_parse(initial_ws_terminal_state, "hello world    \t!")
    assert_weak_full(states)


def test_accept_ws_required_none():
    states = incremental_parse(initial_ws_terminal_state, "hello world!")
    assert_reject(states)
