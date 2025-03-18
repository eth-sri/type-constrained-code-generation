from dataclasses import field, replace
from functools import partial
from typing import List, Type, Self


from typesafe_llm.parser.parser_base import (
    ConcatParserState,
    IncrementalParsingState,
    TerminalParserState,
    MAX_COMMENT_LENGTH,
    UnionParserState,
)
from typesafe_llm.parser.util import fnr_dataclass


@fnr_dataclass
class AnyStringParser(IncrementalParsingState):
    terminator_chars: List[str] = field(default_factory=lambda: [])
    accept: bool = False
    length: int = 0

    def parse_char(self, char: str) -> List[Self]:
        if self.accept:
            return []
        if char in self.terminator_chars:
            return [replace(self, accept=True)]
        if self.length >= MAX_COMMENT_LENGTH:
            return []
        return [replace(self, length=self.length + 1)]

    def num_active_states(self):
        return 1


@fnr_dataclass
class AnyStringParserNoSeq(IncrementalParsingState):
    terminator_seq: List[str] = field(default_factory=lambda: [])
    accepted: str = ""
    accept: bool = False

    @property
    def max_bs_len(self):
        return max(len(bs) for bs in self.terminator_seq)

    def parse_char(self, char: str) -> List[Self]:
        new_str = self.accepted + char
        if self.accept:
            return []
        if any(new_str[-len(bs) :] == bs for bs in self.terminator_seq):
            return [replace(self, accept=True)]
        return [replace(self, accepted=new_str[-self.max_bs_len :])]

    def num_active_states(self):
        return 1


@fnr_dataclass()
class CommentParserState(ConcatParserState):
    # TODO: comments are allowed basically anywhere where whitespace is allowed -> need to add it in all parsers?
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" //"),
            partial(AnyStringParser, terminator_chars=["\n"]),
        ),
        repr=False,
    )


@fnr_dataclass()
class MultilineCommentParserState(ConcatParserState):
    # TODO: comments are allowed basically anywhere where whitespace is allowed -> need to add it in all parsers?
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" /*"),
            partial(AnyStringParserNoSeq, terminator_seq=["*/"]),
        ),
        repr=False,
    )


@fnr_dataclass()
class EOLParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            partial(TerminalParserState, target_value=" ;"),
        ]
    )


@fnr_dataclass()
class BreakStmtParserState(TerminalParserState):
    target_value: str = " break ;"

    def parse_char(self, char: str) -> List[Self]:
        if not self.in_loop:
            return []
        return super().parse_char(char)


@fnr_dataclass()
class ContinueStmtParserState(TerminalParserState):
    target_value: str = " continue ;"

    def parse_char(self, char: str) -> List[Self]:
        if not self.in_loop:
            return []
        return super().parse_char(char)


EmptyStmtParserState = EOLParserState
