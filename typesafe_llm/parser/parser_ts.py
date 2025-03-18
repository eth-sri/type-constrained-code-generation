"""
A parser that only accepts programs in the typesafe part of TypeScript

Invariants:
- non-recoverable states do not exist (i.e. are removed from returned states)
- states are immutable hence reversable
- accept means that the state is done and control can flow back to the parent state (but does not have to)
"""

import functools
import types

import regex
from functools import partial

from termcolor import colored

from .parser_shared import (
    EOLParserState,
    CommentParserState,
    MultilineCommentParserState,
    BreakStmtParserState,
    ContinueStmtParserState,
    EmptyStmtParserState,
)
from .util import (
    fnr_dataclass,
    union_dict,
    sum_list,
    WHITESPACE,
    update_keys,
)
from .parser_base import (
    IncrementalParsingState,
    UnionParserState,
    PlusParserState,
    TerminalParserState,
    ConcatParserState,
    IdentifierParserState,
    DerivableTypeMixin,
    MAX_STRING_LITERAL_LENGTH,
    # MAX_NUMERICAL_LITERAL_LENGTH,
    MAX_COMMENT_LENGTH,
    MAX_STATEMENT_NUM,
    MAX_CONSECUTIVE_WHITESPACE,
    MAX_CONSECUTIVE_NEWLINE,
    WhitespaceParserState,
    IncrementalParserState,
    MAX_REGEX_LITERAL_LENGTH,
)
from .parser_ts_types import (
    CallSignatureParserState,
    TypeParserState,
    DefiningIdentifierParserState,
    call_signature_from_accepted_call_signature_parser,
    identifiers_from_accepted_call_signature_parser,
)
from .types_base import OperatorPrecedence, PType, AnyPType
from .types_ts import (
    NumberPType,
    StringPType,
    FunctionPType,
    VoidPType,
    BooleanPType,
    ArrayPType,
    reachable,
    any_reachable,
    MathPType,
    GenericPType,
    # NeverPType,
    ReduceFunctionPType,
    UnionPType,
    TuplePType,
    AbsTuplePType,
    NullPType,
    UndefinedPType,
    MAX_OPERATOR_PRECEDENCE,
    MIN_OPERATOR_PRECEDENCE,
    OPERATOR_ASSOCIATIVITY,
    OPERATOR_PRECEDENCES,
    extract_type_params,
    TypeParameterPType,
    AbsArrayPType,
    SetPType,
    MapPType,
    RegExpPType,
    IndexSignaturePType,
    LengthPType,
    BigIntPType,
    ObjectPType,
    AbsNumberPType,
    CryptoPType,
    BaseTsObject,
    JSONPType,
    merge_typs,
    CommandPType,
    AbsStringPType,
    FALSEY_TYPES,
    OverlapsWith,
)
from dataclasses import field, replace
from typing import List, Type, Set, Self, Tuple, Optional, Union, Dict


@fnr_dataclass()
class ExistingIdentifierParserState(IdentifierParserState, DerivableTypeMixin):
    force_mutable: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "whitelist",
            # if mutable is True, only allow mutable variables
            [
                v
                for v, (t, m) in self.identifiers.items()
                if (not self.force_mutable or m) and (self.typ is None or self.typ >= t)
            ],
        )

    @property
    def is_mutable(self):
        return self.identifiers.get(self.id_name, (None, False))[1]

    @property
    def described_type(self):
        return self.identifiers.get(self.id_name, (None,))[0]

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        typs = set()
        for v, (t, _) in self.identifiers.items():
            if v.startswith(self.id_name) and v in self.whitelist:
                typs.add(t)
        return any_reachable(
            typs,
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            self.max_steps,
        )


@fnr_dataclass()
class LiteralParserState(UnionParserState, DerivableTypeMixin):
    pass_type: bool = True

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            NumericalLiteralParserState,
            StringLiteralParserState,
            BooleanLiteralParserState,
            TemplateStringLiteralParserState,
            RegExpLiteralParserState,
            BigIntLiteralParserState,
            # IndexSignatureLiteralParserState,
            # LengthLiteralParserState,
        ],
        repr=False,
    )

    @property
    def described_type(self):
        return None

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return any(
            x(
                max_steps=self.max_steps,
            ).derivable(
                goal,
                min_operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path + [self.__class__],
            )
            for x in self.parse_classes
        )


@fnr_dataclass()
class StringLiteralParserState(IncrementalParsingState, DerivableTypeMixin):
    parsed: str = ""
    start: Optional[str] = None
    preceding_whitespace: int = 0
    preceding_newlines: int = 0

    @property
    def described_type(self):
        return StringPType()

    def parse_char(self, char: str) -> List[Self]:
        assert len(char) <= 1, "parse char accepts at most one character"
        if self.typ is not None and not self.typ >= StringPType():
            # must expect a string
            return []
        if self.start is None and char in WHITESPACE:
            # skip whitespace
            if self.preceding_whitespace + 1 > MAX_CONSECUTIVE_WHITESPACE or (
                self.preceding_newlines + 1 > MAX_CONSECUTIVE_NEWLINE and char == "\n"
            ):
                return []
            return [
                replace(
                    self,
                    preceding_whitespace=self.preceding_whitespace + 1,
                    preceding_newlines=self.preceding_newlines + int(char == "\n"),
                )
            ]
        if self.start is None:
            if char == '"':
                return [replace(self, start='"')]
            if char == "'":
                return [replace(self, start="'")]
            else:
                return []
        if self.accept or (
            char == "\n" and (not self.parsed or self.parsed[-1] != "\\")
        ):
            return []
        if char == self.start:
            return [replace(self, accept=True)]
        elif len(self.parsed) < MAX_STRING_LITERAL_LENGTH:
            return [replace(self, parsed=self.parsed + char, accept=False)]
        else:
            return []

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            StringPType(),
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def num_active_states(self):
        return 1


@fnr_dataclass()
class TemplateStringLiteralExpressionParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value="${"),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )


@fnr_dataclass()
class TemplateStringLiteralParserState(IncrementalParsingState, DerivableTypeMixin):
    parsed: str = ""
    start: Optional[str] = None
    active: List[IncrementalParsingState] = None
    accept: bool = False
    preceding_whitespace: int = 0
    preceding_newlines: int = 0

    @property
    def described_type(self):
        return StringPType()

    def parse_char(self, char: str) -> List[Self]:
        if not self.typ >= StringPType():
            # must expect a string
            return []
        if self.parsed == "" and char in WHITESPACE:
            # skip whitespace
            if self.preceding_whitespace + 1 > MAX_CONSECUTIVE_WHITESPACE or (
                self.preceding_newlines + 1 > MAX_CONSECUTIVE_NEWLINE and char == "\n"
            ):
                return []
            return [
                replace(
                    self,
                    preceding_whitespace=self.preceding_whitespace + 1,
                    preceding_newlines=self.preceding_newlines + int(char == "\n"),
                )
            ]
        if self.start is None:
            if char == "`":
                return [replace(self, start="`")]
            else:
                return []
        if self.active is not None:
            new_active = sum_list(s.parse_char(char) for s in self.active)
            ress = []
            if new_active:
                ress.append(replace(self, active=new_active))
            if any(s.accept for s in new_active):
                ress.append(replace(self, active=None))
            return ress
        if self.accept or (char == "\n" and self.parsed[-1] != "\\"):
            return []
        if char == self.start:
            return [replace(self, accept=True)]
        res = [replace(self, parsed=self.parsed + char, accept=False)]
        if char != "\\":
            res.append(
                replace(
                    self,
                    active=[
                        TemplateStringLiteralExpressionParserState(
                            identifiers=self.identifiers,
                            typ=AnyPType(),
                        )
                    ],
                )
            )
        return res

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            StringPType(),
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def num_active_states(self):
        return (
            sum(x.num_active_states() for x in self.active)
            if self.active is not None
            else 1
        )


@fnr_dataclass()
class IndexElementLiteralParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            LiteralParserState,  # numeric literal or string literal depending on type
            partial(TerminalParserState, target_value=" :"),
            ExpressionParserState,
        )
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        typ: IndexSignaturePType = self.typ
        if pos == 0:
            if isinstance(typ.key_type, NumberPType):
                return NumericalLiteralParserState(), self
            elif isinstance(typ.key_type, StringPType):
                return StringLiteralParserState(), self
            else:
                raise NotImplementedError(
                    "can not yet support anything but string/number key type"
                )
        elif pos == 2:
            return (
                ExpressionParserState(
                    typ=typ.value_type,
                    identifiers=self.identifiers,
                    max_steps=self.max_steps,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


EmptyIndexSignatureLiteralParserState = partial(TerminalParserState, target_value=" }")
IndexSignatureEndParserState = partial(
    UnionParserState,
    parse_classes=[
        partial(TerminalParserState, target_value=" }"),
        partial(TerminalParserState, target_value=" , }"),
    ],
)


@fnr_dataclass()
class SingleIndexSignatureLiteralParserState(ConcatParserState, DerivableTypeMixin):
    pass_type: bool = True
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            IndexElementLiteralParserState,
            IndexSignatureEndParserState,
        )
    )


@fnr_dataclass()
class MultiIndexSignatureLiteralParserState(ConcatParserState, DerivableTypeMixin):
    pass_type: bool = True
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            IndexElementLiteralParserState,
            partial(
                PlusParserState,
                parse_class=partial(
                    ConcatParserState,
                    pass_type=True,
                    parse_classes=[
                        partial(TerminalParserState, target_value=" ,"),
                        IndexElementLiteralParserState,
                    ],
                ),
            ),
            IndexSignatureEndParserState,
        )
    )


@fnr_dataclass()
class IndexSignatureLiteralParserState(ConcatParserState, DerivableTypeMixin):
    pass_type: bool = True
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" {"),
            partial(
                UnionParserState,
                pass_type=True,
                parse_classes=[
                    EmptyIndexSignatureLiteralParserState,
                    SingleIndexSignatureLiteralParserState,
                    MultiIndexSignatureLiteralParserState,
                ],
            ),
        )
    )

    @property
    def described_type(self):
        return self.typ

    def parse_char(self, char: str) -> List[Self]:
        if self.typ is None or not isinstance(self.typ, IndexSignaturePType):
            # must expect an index sig
            return []
        return super().parse_char(char)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            self.typ,
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass()
class LengthLiteralParserState(ConcatParserState, DerivableTypeMixin):
    pass_type = False
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" { length:"),
            partial(ExpressionParserState, typ=NumberPType()),
            partial(TerminalParserState, target_value=" }"),
        )
    )

    @property
    def described_type(self):
        return LengthPType()

    def parse_char(self, char: str) -> List[Self]:
        if self.typ is None or not isinstance(self.typ, LengthPType):
            # must expect an index sig
            return []
        return super().parse_char(char)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            LengthPType(),
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass()
class RegExpLiteralParserState(IncrementalParsingState, DerivableTypeMixin):
    parsed: str = ""
    start: Optional[str] = None
    preceding_whitespace: int = 0
    preceding_newlines: int = 0

    @property
    def described_type(self):
        return RegExpPType()

    def parse_char(self, char: str) -> List[Self]:
        assert len(char) <= 1, "parse char accepts at most one character"
        if self.parsed == "" and char in WHITESPACE:
            # skip whitespace
            if self.preceding_whitespace + 1 > MAX_CONSECUTIVE_WHITESPACE or (
                self.preceding_newlines + 1 > MAX_CONSECUTIVE_NEWLINE and char == "\n"
            ):
                return []
            return [
                replace(
                    self,
                    preceding_whitespace=self.preceding_whitespace + 1,
                    preceding_newlines=self.preceding_newlines + int(char == "\n"),
                )
            ]
        if not self.typ >= RegExpPType():
            # must expect a string
            return []
        if self.start is None:
            if char == "/":
                return [replace(self, start="/", parsed="/")]
            else:
                return []
        if (self.accept and (char not in "gimusy")) or (char == "\n"):
            return []
        if char == self.start and self.parsed == self.start:
            # empty regex is invalid (//)
            return []
        if char == self.start or (self.accept and char in "gimusy"):
            return [replace(self, accept=True)]
        if len(self.parsed) >= MAX_REGEX_LITERAL_LENGTH:
            return []
        return [replace(self, parsed=self.parsed + char, accept=False)]

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            StringPType(),
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def num_active_states(self):
        return 1


OCT_REGEX = r"0[oO]([1-7][0-7]*|0)"
HEX_REGEX = r"0[xX]([1-9a-fA-F][0-9a-fA-F]*|0)"
BINARY_REGEX = r"0[bB](1[01]*|0)"
DECIMAL_REGEX = r"([1-9]\d*|0)"
INTEGER_REGEX = f"({OCT_REGEX}|{HEX_REGEX}|{BINARY_REGEX}|{DECIMAL_REGEX})"
FLOATING_REGEX = r"(([1-9]\d*|0)(\.\d*)?|\.\d\d*)([eE][+-]?\d*)?"
NUMBER_REGEX = f"({INTEGER_REGEX}|{FLOATING_REGEX})"
NUMBER_REGEX_COMPILED = regex.compile(NUMBER_REGEX)
BIGINT_REGEX = f"{INTEGER_REGEX}n"
BIGINT_REGEX_COMPILED = regex.compile(BIGINT_REGEX)


@fnr_dataclass()
class NumericalLiteralParserState(IncrementalParsingState, DerivableTypeMixin):
    parsed: str = ""
    preceding_whitespace: int = 0
    preceding_newlines: int = 0

    @property
    def described_type(self):
        return NumberPType()

    def parse_char(self, char: str) -> List[Self]:
        assert len(char) <= 1, "parse char accepts at most one character"
        if self.parsed == "" and char in WHITESPACE:
            # skip whitespace
            if self.preceding_whitespace + 1 > MAX_CONSECUTIVE_WHITESPACE or (
                self.preceding_newlines + 1 > MAX_CONSECUTIVE_NEWLINE and char == "\n"
            ):
                return []
            return [
                replace(
                    self,
                    preceding_whitespace=self.preceding_whitespace + 1,
                    preceding_newlines=self.preceding_newlines + int(char == "\n"),
                )
            ]
        new_parsed = self.parsed + char
        if NUMBER_REGEX_COMPILED.fullmatch(new_parsed, partial=True):
            return [
                replace(
                    self,
                    parsed=new_parsed,
                    accept=NUMBER_REGEX_COMPILED.fullmatch(new_parsed, partial=False),
                )
            ]
        else:
            return []

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            NumberPType(),
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def num_active_states(self):
        return 1


@fnr_dataclass()
class BigIntLiteralParserState(IncrementalParsingState, DerivableTypeMixin):
    parsed: str = ""
    preceding_whitespace: int = 0
    preceding_newlines: int = 0

    @property
    def described_type(self):
        return BigIntPType()

    def parse_char(self, char: str) -> List[Self]:
        assert len(char) <= 1, "parse char accepts at most one character"
        if self.parsed == "" and char in WHITESPACE:
            # skip whitespace
            if self.preceding_whitespace + 1 > MAX_CONSECUTIVE_WHITESPACE or (
                self.preceding_newlines + 1 > MAX_CONSECUTIVE_NEWLINE and char == "\n"
            ):
                return []
            return [
                replace(
                    self,
                    preceding_whitespace=self.preceding_whitespace + 1,
                    preceding_newlines=self.preceding_newlines + int(char == "\n"),
                )
            ]
        new_parsed = self.parsed + char
        if BIGINT_REGEX_COMPILED.fullmatch(new_parsed, partial=True):
            return [
                replace(
                    self,
                    parsed=new_parsed,
                    accept=BIGINT_REGEX_COMPILED.fullmatch(new_parsed, partial=False),
                )
            ]
        else:
            return []

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            BigIntPType(),
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def num_active_states(self):
        return 1


@fnr_dataclass()
class BooleanLiteralParserState(IncrementalParsingState, DerivableTypeMixin):
    states: List[TerminalParserState] = field(
        default_factory=lambda: [
            TerminalParserState(target_value=" true"),
            TerminalParserState(target_value=" false"),
        ]
    )

    @property
    def described_type(self):
        return BooleanPType()

    def parse_char(self, char: str) -> List[Self]:
        new_states = sum_list(s.parse_char(char) for s in self.states)
        if not new_states:
            return []
        accept = any(s.accept for s in new_states)
        return [replace(self, accept=accept, states=new_states)]

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            BooleanPType(),
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def num_active_states(self):
        return sum(x.num_active_states() for x in self.states)


@fnr_dataclass()
class TypedDeclarationTargetParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" let\t"),
                    partial(TerminalParserState, target_value=" const\t"),
                ],
            ),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" :"),
            # TODO in theory we have to restrict here to all values that we _can_ create in the current state (think i.e. private class in Rust)
            TypeParserState,
        ]
    )

    @property
    def mutable(self):
        return self.parsed_states[0].target_value == " let\t"

    @property
    def declared_identifier(self):
        return self.parsed_states[1].id_name

    @property
    def described_type(self):
        if len(self.parsed_states) < 4:
            return None
        return self.parsed_states[3].described_type


@fnr_dataclass()
class ComputedMemberAssignmentTargetParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ExpressionParserState,
            partial(TerminalParserState, target_value=" ["),
            partial(ExpressionParserState, typ=NumberPType()),
            partial(TerminalParserState, target_value=" ]"),
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (17, "left")

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            return (
                ExpressionParserState(
                    typ=ArrayPType(
                        GenericPType()
                        if (self.typ is None or self.typ == AnyPType())
                        else self.typ
                    ),
                    identifiers=self.identifiers,
                    rhs_operator_precedence=self.operator_precedence,
                ),
                self,
            )

        return super().init_class_at_pos_hook(pos)

    @property
    def described_type(self):
        if (
            len(self.parsed_states) < 1
            or self.parsed_states[0].described_type.element_type == GenericPType()
        ):
            return None
        return self.parsed_states[0].described_type.element_type

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if len(self.parsed_states) < 1:
            active = self.active
            if active is None:
                active = self.init_class_at_pos_hook(0)[0]
            if self.__class__ not in recursing_path:
                return active.derivable(
                    ArrayPType(goal),
                    min_operator_precedence,
                    as_array,
                    as_nested_expression,
                    as_pattern,
                    [self.__class__] + recursing_path,
                )
            return False
        return self.parsed_states[0].derivable(
            goal,
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )


@fnr_dataclass()
class TupleComputedMemberAssignmentTargetParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ExpressionParserState,
            partial(TerminalParserState, target_value=" ["),
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" 0"),
                    partial(TerminalParserState, target_value=" 1"),
                    partial(TerminalParserState, target_value=" 2"),
                ],
            ),
            partial(TerminalParserState, target_value=" ]"),
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (17, "left")

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            return (
                ExpressionParserState(
                    typ=AbsTuplePType(types=[]),
                    identifiers=self.identifiers,
                    rhs_operator_precedence=self.operator_precedence,
                ),
                self,
            )

        if pos == 2:
            tuple_size = len(self.parsed_states[0].described_type.types)
            parse_classes = [
                partial(TerminalParserState, target_value=f" {i}")
                for i in range(tuple_size)
            ]
            return UnionParserState(parse_classes=parse_classes), self

        return super().init_class_at_pos_hook(pos)

    @property
    def described_type(self):
        if len(self.parsed_states) == 0:
            if self.active is None:
                return AnyPType()
            else:
                return self.active.typ
        elif len(self.parsed_states) == 1:
            lhs_types = self.parsed_states[0].described_type.types
            return merge_typs(*lhs_types)
        elif len(self.parsed_states) == 2:
            lhs_types = self.parsed_states[0].described_type.types
            if self.active is None:
                return merge_typs(*lhs_types)
            else:
                index = int(self.parsed_states[2].target_value.strip())
                return lhs_types[index]
        elif len(self.parsed_states) >= 3:
            lhs_types = self.parsed_states[0].described_type.types
            index = int(self.parsed_states[2].target_value.strip())
            return lhs_types[index]

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
        )


@fnr_dataclass()
class MemberAssignmentTargetParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ExpressionParserState,
            partial(TerminalParserState, target_value=" ."),
            IdentifierParserState,
        ),
        repr=False,
    )
    attributes: Dict[str, Tuple[PType, bool]] = None
    operator_precedence: OperatorPrecedence = (17, "left")

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            # TODO properly determine all types with mutable attributes of given type
            if isinstance(self.typ, AnyPType):
                target_type = AnyPType()
            elif isinstance(self.typ, NumberPType):
                # the only mutable number attribute is array.length
                target_type = ArrayPType(GenericPType())
            else:
                raise NotImplementedError(
                    f"no mutable attributes for type {self.typ} known"
                )
            return (
                ExpressionParserState(
                    typ=target_type,
                    identifiers=self.identifiers,
                    rhs_operator_precedence=self.operator_precedence,
                ),
                self,
            )
        if pos == 2:
            attributes = self.parsed_states[0].described_type.attributes
            return (
                ExistingIdentifierParserState(
                    identifiers={
                        k: v for k, v in attributes.items() if self.typ >= v[0]
                    },
                    force_mutable=True,
                ),
                replace(self, attributes=attributes),
            )

        return super().init_class_at_pos_hook(pos)

    @property
    def described_type(self):
        if self.attributes is None or len(self.parsed_states) < 3:
            return None
        return self.attributes[self.parsed_states[2].id_name][0]

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if self.active is None or len(self.parsed_states) == 0:
            if self.__class__ not in recursing_path:
                return self.init_class_at_pos_hook(0)[0].derivable(
                    goal,
                    min_operator_precedence,
                    as_array,
                    as_nested_expression,
                    as_pattern,
                    [self.__class__] + recursing_path,
                )
            return False
        if len(self.parsed_states) == 1:
            return self.parsed_states[0].derivable(
                goal,
                min_operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
        if len(self.parsed_states) == 2:
            return self.active.derivable(
                goal,
                min_operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
        return self.parsed_states[2].derivable(
            goal,
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )


@fnr_dataclass()
class TypedAssignmentTargetParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            partial(ExistingIdentifierParserState, force_mutable=True),
            ComputedMemberAssignmentTargetParserState,
            TupleComputedMemberAssignmentTargetParserState,
            MemberAssignmentTargetParserState,
        ]
    )
    pass_type: bool = True

    def init_classes(self):
        return [
            x(
                identifiers=self.identifiers,
                typ=self.typ,
                **(
                    {}
                    if not isinstance(x(), ComputedMemberAssignmentTargetParserState)
                    and not isinstance(x(), MemberAssignmentTargetParserState)
                    else {"rhs_operator_precedence": self.rhs_operator_precedence}
                ),
            )
            for x in self.parse_classes
        ]


@fnr_dataclass()
class ExistingArrayIdentifierParserState(ExistingIdentifierParserState):
    mutable: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "whitelist",
            # if mutable is True, only allow mutable variables
            [
                v
                for v, (t, m) in self.identifiers.items()
                if (not self.mutable or m) and (isinstance(t, ArrayPType))
            ],
        )


# Anything that can be assigned to and is an array
ArrayAssignmentTargetParserState = partial(
    ExistingArrayIdentifierParserState, mutable=True, operator_precedence=2
)


@fnr_dataclass()
class ReturnVoidParserState(TerminalParserState):
    target_value: str = " return ;"

    def __post_init__(self):
        object.__setattr__(self, "returned_in_branches", True)
        super().__post_init__()

    def parse_char(self, char: str) -> List[Self]:
        if self.return_type is not None and self.return_type >= VoidPType():
            return super().parse_char(char)
        return []


@fnr_dataclass()
class ReturnValueParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" return"),
            # split so that we can disallow newline in this part and at the beginning of the expression parser
            partial(TerminalParserState, target_value="\t"),
            ExpressionParserState,
            EOLParserState,
        ),
        repr=False,
    )
    expression_start: bool = False

    def __post_init__(self):
        object.__setattr__(self, "returned_in_branches", True)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 2:
            return (
                ExpressionParserState(
                    typ=self.return_type,
                    identifiers=self.identifiers,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        # disallow return when there is no type
        if self.return_type is None:
            return []
        # check if we can return the type TODO is this redundant?
        if len(self.parsed_states) == 2 and self.active is not None:
            active = self.active
        elif len(self.parsed_states) > 2:
            active = self.parsed_states[2]
        else:
            active = self.init_class_at_pos_hook(2)[0]
        if not active.derivable(self.return_type):
            return []
        return super().parse_char(char)


@fnr_dataclass()
class ReturnStatementParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ReturnVoidParserState,
            ReturnValueParserState,
        ),
        repr=False,
    )


@fnr_dataclass()
class ExistingConstructorIdentifierParserState(ExistingIdentifierParserState):
    mutable: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "whitelist",
            # if mutable is True, only allow mutable variables
            [
                v
                for v, (t, m) in self.identifiers.items()
                if (not self.mutable or m)
                and (isinstance(t, FunctionPType) and t.is_constructor)
            ],
        )


@fnr_dataclass
class ExistingConstructorCallParserState(IncrementalParsingState, DerivableTypeMixin):
    """
    parses "x()" expressions
    """

    accept: bool = False
    active: Optional[ExistingConstructorIdentifierParserState] = None

    @property
    def described_type(self):
        if self.active is None:
            return None
        if not isinstance(self.active.described_type, FunctionPType):
            return None
        return self.active.described_type.return_type

    def parse_char(self, char: str) -> List[Self]:
        if self.active is None:
            active = ExistingConstructorIdentifierParserState(
                identifiers=self.identifiers,
                mutable=False,
            )
        else:
            active = self.active
        new_states = active.parse_char(char)
        update_states = [replace(self, active=s) for s in new_states if not s.accept]
        forward_states = [
            FunctionCallParserState(lhs=s, identifiers=self.identifiers)
            for s in new_states
            if s.accept
            and not (
                isinstance(s, ExistingConstructorIdentifierParserState)
                and s.id_name == "Map"
            )
        ] + [
            replace(
                self,
                active=FunctionParameterizationParserState(
                    lhs=s, identifiers=self.identifiers
                ),
            )
            for s in new_states
            if s.accept and isinstance(s, ExistingConstructorIdentifierParserState)
        ]
        return update_states + forward_states

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        active = self.active
        if active is None:
            active = ExistingConstructorIdentifierParserState(
                identifiers=self.identifiers,
                mutable=False,
            )
        return active.derivable(
            goal,
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )


@fnr_dataclass
class NewParserState(ConcatParserState, DerivableTypeMixin):
    """
    Parses "new ...()" expressions
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" new"),
            ExistingConstructorCallParserState,
        ),
        repr=False,
    )

    @property
    def described_type(self):
        active = self.active
        if len(self.parsed_states) == 2:
            active = self.parsed_states[1]
        elif active is None or len(self.parsed_states) == 0:
            active = self.init_class_at_pos_hook(1)[0]
        return active.described_type

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        active = self.active
        if len(self.parsed_states) == 2:
            active = self.parsed_states[1]
        elif active is None or len(self.parsed_states) == 0:
            active = self.init_class_at_pos_hook(1)[0]
        return active.derivable(
            goal,
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )


@fnr_dataclass()
class StmtParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            ReturnStatementParserState,
            ContinueStmtParserState,
            BreakStmtParserState,
            TypedDeclarationParserState,
            UntypedDeclarationParserState,
            ComputedMemberGenericAssignmentParserState,
            TypedDeclarationAssignmentParserState,
            ExpressionStatementParserState,
            EmptyStmtParserState,
            CommentParserState,
            MultilineCommentParserState,
            StmtsBlockParserState,
            ITEParserState,
            IfParserState,
            ForLoopParserState,
            ForOfExistingIdentifierParserState,
            ForOfDeclarationParserState,
            ForOfTupleDeclarationParserState,
            WhileLoopParserState,
            DoWhileLoopParserState,
            FunctionDeclarationParserState,
            TryCatchParserState,
            ThrowErrorParser,
        ],
        repr=False,
    )


@fnr_dataclass()
class StmtsParserState(PlusParserState):
    parse_class: Type[IncrementalParsingState] = StmtParserState

    def parse_char(self, char: str) -> List[Self]:
        if len(self.accepted) >= MAX_STATEMENT_NUM and self.active is None:
            # force a return stmt if we still have to return, otherwise just reject creating a new parser
            if self.has_to_return:
                return replace(
                    self,
                    active=ReturnValueParserState(
                        identifiers=self.identifiers,
                        return_type=self.return_type,
                        returned_in_branches=self.returned_in_branches,
                        in_loop=self.in_loop,
                        **({"typ": self.typ} if self.pass_type else {}),
                        max_steps=self.max_steps,
                    ),
                ).parse_char(char)
            else:
                return []
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.active is None and r.accepted[-1].accept:
                r = replace(
                    r,
                    # add previously defined identifiers to new state
                    identifiers=r.identifiers | r.accepted[-1].identifiers,
                    # only accept if last statement is a return statement when return type is not None
                    accept=(
                        (not self.has_to_return)
                        or (r.return_type is None)
                        or (r.accepted[-1].returned_in_branches)
                    ),
                    # If the last statement returned in all branches, we do not need to return anymore
                    has_to_return=(
                        not r.accepted[-1].returned_in_branches and self.has_to_return
                    ),
                    returned_in_branches=(
                        self.returned_in_branches or r.accepted[-1].returned_in_branches
                    ),
                )
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class ProgramAndEmptyStrParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            StmtsParserState,
            WhitespaceParserState,
        ),
        repr=False,
    )


@fnr_dataclass()
class ProgramParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            ProgramAndEmptyStrParserState,
            StmtsParserState,
        ],
        repr=False,
    )


ARITHMETIC_OPS = {"+", "-", "*", "**", "/", "%", "<<", ">>", ">>>", "&", "|", "^"}
ARITHMETIC_OP_PRECEDENCE = {
    "**": (13, "right"),
    "+": (11, "left"),
    "-": (11, "left"),
    "*": (12, "left"),
    "/": (12, "left"),
    "%": (12, "left"),
    "<<": (10, "left"),
    ">>": (10, "left"),
    ">>>": (10, "left"),
    "&": (7, "left"),
    "|": (5, "left"),
    "^": (6, "left"),
}
BOOL_OPS = {"<", "==", "===", "<=", ">", ">=", "!=", "!=="}
BOOL_OPS_PRECEDENCE = {
    "<": (9, "left"),
    "==": (8, "left"),
    "===": (8, "left"),
    "<=": (9, "left"),
    ">": (9, "left"),
    ">=": (9, "left"),
    "!=": (8, "left"),
    "!==": (8, "left"),
}
LOGIC_OPS = {"&&", "||"}
LOGIC_OPS_PRECEDENCE = {
    "&&": (4, "left"),
    "||": (3, "left"),
    "??": (3, "left"),
}
UNOPS = {"!", "~", "-", "+"}
UNOPS_PRECEDENCE = {
    "!": (14, "right"),
    "~": (14, "right"),
    "-": (14, "right"),
    "+": (14, "right"),
}


@fnr_dataclass()
class ExpressionParserState(IncrementalParsingState, DerivableTypeMixin):
    """
    starting with i.e. an existing identifier, many types can be reached using arbitrarily repeated attribute accesses or function calls
    think i.e. of Number(12).toString().propertyIsEnumerable() (number -> string -> boolean)
    """

    pass_type: bool = True

    parse_classes: List[
        Union[Type[IncrementalParsingState], Type[DerivableTypeMixin]]
    ] = field(
        default_factory=lambda: [
            ExistingIdentifierParserState,
            LiteralParserState,
            IndexSignatureLiteralParserState,
            LengthLiteralParserState,
            GroupedExpressionParserState,
            ArrayExpressionParserState,
            TupleExpressionParserState,
            UnopExpressionParserState,
            AssigningPrefixParserState,
            NewParserState,
            RequireCryptoParserState,
            UntypedLambdaExprParserState,
            UntypedLambdaFunParserState,
            UntypedAnonymousFunParserState,
            TypedLambdaExprParserState,
            TypedLambdaFunParserState,
        ],
        repr=False,
    )
    extend_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            # p.x
            MemberAccessStateParser,
            # p[]
            ComputedMemberAccessParserState,
            TupleComputedMemberAccessParserState,
            # p()
            FunctionCallParserState,
            # +
            *ArithmeticParserStates,
            # < and ==
            *BoolOpParserStates,
            # &&
            AndLogicOpParserState,
            OrLogicOpParserState,
            # ?
            TernaryOpParserState,
            # FunctionParameterizationParserState,
            OptionalChainingStateParser,
            # NullishCoalescingLogicOpParserState,
            AssigningPostfixParserState,
            AssignmentParserState,
            AsExpressionParserState,
        ],
        repr=False,
    )
    active: List[DerivableTypeMixin] = None
    operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE
    rhs_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE
    def_id_name: str = None

    def parse_char(self, char: str) -> List[Self]:
        orig_active_classes = self.active if self.active is not None else None
        if orig_active_classes is None:
            orig_active_classes = []
            for x in self.parse_classes:
                if x.operator_precedence < self.rhs_operator_precedence:
                    continue
                orig_active_classes.append(
                    x(
                        identifiers=self.identifiers,
                        max_steps=self.max_steps,
                        **(
                            {"typ": self.typ}
                            if x
                            in (
                                UntypedLambdaFunParserState,
                                UntypedLambdaExprParserState,
                                UntypedAnonymousFunParserState,
                                IndexSignatureLiteralParserState,
                                LengthLiteralParserState,
                            )
                            else {"def_id_name": self.def_id_name}
                            if x
                            in (
                                TypedLambdaExprParserState,
                                TypedLambdaFunParserState,
                            )
                            else {}
                        ),
                    )
                )
                if (
                    x == ArrayExpressionParserState and isinstance(self.typ, ArrayPType)
                ):  # or (x == TupleExpressionParserState and isinstance(self.typ, TuplePType)):
                    orig_active_classes.append(
                        x(
                            identifiers=self.identifiers,
                            typ=self.typ,
                        )
                    )
        active_classes = orig_active_classes.copy()
        new_active_classes = []
        # all of these are the same:
        # - check if the previous state accepted
        # - if so, check if any extension (functioncall, arith, etc) can continue this
        min_precedence = max(self.operator_precedence, self.rhs_operator_precedence)
        for a in orig_active_classes:
            # reject completions that have too many steps
            if not a.accept or a.max_steps <= 0:
                continue
            # transition to arithmetic parsing
            # check
            # a) is the current state a number
            # b) can arithmetic operations result in the desired type
            active_classes += [
                ext(lhs=a, identifiers=self.identifiers, max_steps=a.max_steps - 1)
                for ext in self.extend_classes
                # only allow if the lhs has a higher precedence than the operator
                if (
                    ext.operator_precedence < a.operator_precedence
                    or (
                        ext.operator_precedence[0] == a.operator_precedence[0]
                        and ext.operator_precedence[1] == "left"
                    )
                )
                and (
                    ext.operator_precedence > min_precedence
                    or (
                        ext.operator_precedence[0] == min_precedence[0]
                        and ext.operator_precedence[1] == "right"
                    )
                )
            ]
        for ac in active_classes:
            new_states = ac.parse_char(char)
            for ns in new_states:
                reachable = ns.derivable(
                    self.typ,
                    min_operator_precedence=min_precedence,
                )
                if reachable:
                    if (
                        ns.accept
                        and (
                            isinstance(ns, NonEmptyTupleExpressionParserState)
                            or isinstance(ns, EmptyTupleExpressionParserState)
                        )
                        and (
                            self.typ is None
                            or isinstance(self.typ, AnyPType)
                            or isinstance(self.typ, TypeParameterPType)
                        )
                    ):
                        new_active_classes.append(
                            AssignmentParserState(lhs=ns, identifiers=self.identifiers)
                        )
                    else:
                        new_active_classes.append(ns)

        final_states, final_active_classes = [], []
        for s in new_active_classes:
            s_described_type = s.described_type
            if s.accept and (
                self.typ >= s_described_type
                or extract_type_params(self.typ, s_described_type)[0]
            ):
                final_s = replace(
                    self,
                    active=[s],
                    accept=True,
                )
                final_states.append(final_s)
            else:
                final_active_classes.append(s)

        if final_active_classes:
            final_states.append(
                replace(
                    self,
                    active=final_active_classes,
                    accept=False,
                )
            )

        return final_states

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        active_classes = self.active
        if active_classes is None:
            active_classes = [
                x(
                    identifiers=self.identifiers,
                    max_steps=self.max_steps,
                    **(
                        {}
                        if not any(
                            isinstance(x(), T)
                            for T in (
                                UntypedLambdaFunParserState,
                                UntypedLambdaExprParserState,
                                UntypedAnonymousFunParserState,
                                IndexSignatureLiteralParserState,
                                LengthLiteralParserState,
                            )
                        )
                        else {"typ": self.typ}
                    ),
                )
                for x in self.parse_classes
            ]
            recursing_path = recursing_path + [self.__class__]
        return any(
            x.derivable(
                goal,
                min_operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
            for x in active_classes
        )

    @property
    def described_type(self):
        active_classes = self.active
        if active_classes is None:
            return None
        accepting_states_described_types = {
            x.described_type for x in active_classes if x.accept
        }
        if len(accepting_states_described_types) == 1:
            return next(iter(accepting_states_described_types))
        return None

    def num_active_states(self):
        return (
            sum(x.num_active_states() for x in self.active)
            if self.active is not None
            else 1
        )


@fnr_dataclass()
class AsExpressionParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value="\v\bas\t"),
            TypeParserState,
        ),
        repr=False,
    )
    lhs: DerivableTypeMixin = None
    operator_precedence: OperatorPrecedence = (15, "left")

    @property
    def described_type(self):
        # TODO check this
        if len(self.parsed_states) < 2:
            return self.lhs.described_type
        else:
            return self.parsed_states[1].described_type

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        res = reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
        )
        return res


@fnr_dataclass()
class ExpressionStatementParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ExpressionParserState,
            EOLParserState,
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            # we use this to extract potential array type updates from the expression parser
            parser, self2 = super().init_class_at_pos_hook(pos)
            updates = extract_array_type_update_from_stmt(
                self.parsed_states[0], self2.identifiers
            )
            return (
                parser,
                replace(self2, identifiers=union_dict(self2.identifiers, updates)),
            )
        return super().init_class_at_pos_hook(pos)


ASSIGNMENT_OPS = [
    # Which type is allowed to be lhs of assignment for list of ops
    (AnyPType(), ["=", "||=", "&&=", "??="]),
    (
        NumberPType(),
        ["+=", "-=", "*=", "**=", "/=", "%=", "<<=", ">>=", ">>>=", "&=", "|=", "^="],
    ),
    (StringPType(), ["+="]),
]


@fnr_dataclass
class AssignmentParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ="),
            ExpressionParserState,
        ),  # replaced in post_init
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "left")
    lhs: DerivableTypeMixin = None

    @property
    def described_type(self):
        return self.lhs.described_type

    def parse_char(self, char: str) -> List[Self]:
        # check that lhs is actually a valid assignment target
        if self.lhs is None:
            return []
        if not any(x[0] >= self.lhs.described_type for x in ASSIGNMENT_OPS):
            return []
        if not is_assignable(self.lhs):
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        lhs_described_typ = self.lhs.described_type
        if pos == 0 and not isinstance(self.lhs, TypedDeclarationAssignmentParserState):
            # We only allow updates if the variable was not declared in the same statement
            return (
                UnionParserState(
                    parse_classes=[
                        partial(
                            TerminalParserState,
                            target_value=f" {op}",
                        )
                        for typ, ops in ASSIGNMENT_OPS
                        for op in ops
                        if typ >= lhs_described_typ
                    ],
                ),
                self,
            )
        if pos == 1:
            return (
                ExpressionParserState(
                    typ=lhs_described_typ,
                    identifiers=self.identifiers,
                    operator_precedence=self.operator_precedence,
                    max_steps=self.max_steps,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass()
class TypedDeclarationParserState(ConcatParserState):
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            TypedDeclarationTargetParserState,
            EOLParserState,
        ),
        repr=False,
    )

    def parse_char(self, char: str) -> List[Self]:
        ress = super().parse_char(char)
        fixed_ress = []
        for res in ress:
            if res.accept:
                first_state: TypedDeclarationTargetParserState = res.parsed_states[0]
                res = replace(
                    res,
                    accept=True,
                    identifiers=union_dict(
                        res.identifiers,
                        {
                            first_state.declared_identifier: (
                                first_state.described_type,
                                first_state.mutable,
                            )
                        },
                    ),
                )
            fixed_ress.append(res)
        return fixed_ress


@fnr_dataclass()
class TypedDeclarationAssignmentParserState(ConcatParserState):
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            TypedDeclarationTargetParserState,
            partial(TerminalParserState, target_value=" ="),
            ExpressionParserState,
            EOLParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "right")

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 2:
            lhs_described_typ = self.parsed_states[0].described_type
            return (
                ExpressionParserState(
                    typ=lhs_described_typ,
                    identifiers=self.identifiers,
                    operator_precedence=self.operator_precedence,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        ress = super().parse_char(char)
        fixed_ress = []
        for res in ress:
            if res.accept:
                first_state: TypedDeclarationTargetParserState = res.parsed_states[0]
                res = replace(
                    res,
                    accept=True,
                    identifiers=union_dict(
                        res.identifiers,
                        {
                            first_state.declared_identifier: (
                                first_state.described_type,
                                first_state.mutable,
                            )
                        },
                    ),
                )
            fixed_ress.append(res)
        return fixed_ress


@fnr_dataclass()
class UntypedDeclarationParserState(ConcatParserState):
    parse_classes: List[IncrementalParsingState] = field(
        default_factory=lambda: (
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" let\t"),
                    partial(TerminalParserState, target_value=" const\t"),
                ],
            ),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" ="),
            ExpressionParserState,
            EOLParserState,
        ),
        repr=False,
    )

    def parse_char(self, char: str) -> List[Self]:
        ress = super().parse_char(char)
        fixed_ress = []
        for res in ress:
            if res.accept:
                modifier: TerminalParserState = res.parsed_states[0]
                res = replace(
                    res,
                    accept=True,
                    identifiers=union_dict(
                        res.identifiers,
                        {
                            res.parsed_states[1].id_name: (
                                res.parsed_states[3].described_type,
                                modifier.target_value == " let\t",
                            )
                        },
                    ),
                )
            fixed_ress.append(res)
        return fixed_ress

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 3:
            def_id_name = self.parsed_states[1].id_name
            return (
                ExpressionParserState(
                    def_id_name=def_id_name,
                    identifiers=self.identifiers,
                    typ=self.typ,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class ITEParserState(ConcatParserState):
    """
    ITE statement
    Parsers:
    0: if(
    1: expression
    2: ){
    3: stmts
    4: }else{
    5: stmts
    6: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" if ("),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
            StmtParserState,
            # NOTE: can have dangling else issue! convention: else attaches to closest if
            # i.e. if previous state is ITE or If with unbraced group, then we can not parse else
            partial(TerminalParserState, target_value=" else"),
            StmtParserState,  # Note: these include If and ITE
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            s = super().init_class_at_pos_hook(pos)[0]
            # remove expressions that start with brackets and allow invalid parameter names
            s.parse_classes.remove(UntypedLambdaExprParserState)
            s.parse_classes.remove(UntypedLambdaFunParserState)
            s.parse_classes.remove(UntypedAnonymousFunParserState)
            s.parse_classes.remove(TypedLambdaExprParserState)
            s.parse_classes.remove(TypedLambdaFunParserState)
            return s, self

        if pos == 3:
            updates_to_identifiers_if, updates_to_identifiers_else = (
                extract_type_cast_from_expression(
                    self.parsed_states[1].active[0], self.identifiers
                )
            )
            return (
                StmtParserState(
                    identifiers=union_dict(self.identifiers, updates_to_identifiers_if),
                    return_type=self.return_type,
                    returned_in_branches=self.returned_in_branches,
                    in_loop=self.in_loop,
                    **({"typ": self.typ} if self.pass_type else {}),
                ),
                self,
            )
        if pos == 4:
            updates_to_identifiers_if, updates_to_identifiers_else = (
                extract_type_cast_from_expression(
                    self.parsed_states[1].active[0], self.identifiers
                )
            )
            # propagate type re-assignments up, except for identifiers that where updated due to the expression
            new_ids = self.parsed_states[3].identifiers
            updated_due_to_expr = {
                *updates_to_identifiers_if.keys(),
                *updates_to_identifiers_else.keys(),
            }
            new_ids = {k: v for k, v in new_ids.items() if k not in updated_due_to_expr}
            new_parser, new_self = super().init_class_at_pos_hook(pos)
            return (
                new_parser,
                replace(self, identifiers=update_keys(new_self.identifiers, new_ids)),
            )
        if pos == 5:
            updates_to_identifiers_if, updates_to_identifiers_else = (
                extract_type_cast_from_expression(
                    self.parsed_states[1].active[0], self.identifiers
                )
            )
            return (
                StmtParserState(
                    identifiers=union_dict(
                        self.identifiers, updates_to_identifiers_else
                    ),
                    return_type=self.return_type,
                    returned_in_branches=self.returned_in_branches,
                    in_loop=self.in_loop,
                    **({"typ": self.typ} if self.pass_type else {}),
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                updates_to_identifiers_if, updates_to_identifiers_else = (
                    extract_type_cast_from_expression(
                        self.parsed_states[1].active[0], self.identifiers
                    )
                )
                # propagate type re-assignments up, except for identifiers that where updated due to the expression
                new_ids = self.parsed_states[3].identifiers
                updated_due_to_expr = {
                    *updates_to_identifiers_if.keys(),
                    *updates_to_identifiers_else.keys(),
                }
                new_ids = {
                    k: v for k, v in new_ids.items() if k not in updated_due_to_expr
                }
                returned_in_branches = (
                    r.parsed_states[3].returned_in_branches
                    and r.parsed_states[5].returned_in_branches
                )
                r = replace(
                    r,
                    returned_in_branches=returned_in_branches,
                    identifiers=update_keys(r.identifiers, new_ids),
                )
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class StmtsBlockParserState(ConcatParserState):
    """
    A group of statements, seperated by brackets {}
    Parsers:
    0: {
    1: stmts
    2: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 2:
            # propagate type re-assignments up
            new_ids = self.parsed_states[1].identifiers
            new_parser, new_self = super().init_class_at_pos_hook(pos)
            return (
                new_parser,
                replace(self, identifiers=update_keys(new_self.identifiers, new_ids)),
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                returned_in_branches = r.parsed_states[1].returned_in_branches
                r = replace(r, returned_in_branches=returned_in_branches)
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class IfParserState(ConcatParserState):
    """
    If statement
    NOTE: this introduces ambiguity with the ITE parser state,
    but it should not be too common to cause issues
    Parsers:
    0: if(
    1: expression
    2: ){
    3: stmts
    4: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" if ("),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
            StmtParserState,
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            s = super().init_class_at_pos_hook(pos)[0]
            # remove expressions that start with brackets and allow invalid parameter names
            s.parse_classes.remove(UntypedLambdaExprParserState)
            s.parse_classes.remove(UntypedLambdaFunParserState)
            s.parse_classes.remove(UntypedAnonymousFunParserState)
            s.parse_classes.remove(TypedLambdaExprParserState)
            s.parse_classes.remove(TypedLambdaFunParserState)
            return s, self
        if pos == 3:
            updates_to_identifiers_if, updates_to_identifiers_else = (
                extract_type_cast_from_expression(
                    self.parsed_states[1].active[0], self.identifiers
                )
            )
            return (
                StmtParserState(
                    identifiers=union_dict(self.identifiers, updates_to_identifiers_if),
                    return_type=self.return_type,
                    returned_in_branches=self.returned_in_branches,
                    in_loop=self.in_loop,
                    **({"typ": self.typ} if self.pass_type else {}),
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                # get id updates from the stmt and propagate it
                new_ids = r.parsed_states[3].identifiers
                # We can not be sure whether the if statement will be executed
                # so we have to assume that the return statement is not executed
                returned_in_branches = False
                r = replace(
                    r,
                    returned_in_branches=returned_in_branches,
                    identifiers=update_keys(r.identifiers, new_ids),
                )
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class FixedCallSignatureParserState(ConcatParserState):
    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.parsed_states:
                last_parsed_state = r.parsed_states[-1]
                if (
                    isinstance(last_parsed_state, ConcatParserState)
                    and last_parsed_state.parsed_states
                    and isinstance(
                        last_parsed_state.parsed_states[0],
                        DefiningIdentifierParserState,
                    )
                ):
                    last_parsed_state = last_parsed_state.parsed_states[0]
                # update identifiers just after a defining identifier was parsed
                if (
                    isinstance(last_parsed_state, DefiningIdentifierParserState)
                    and r.active is None
                ):
                    r = replace(
                        r,
                        identifiers=union_dict(
                            self.identifiers,
                            {
                                last_parsed_state.id_name: (
                                    last_parsed_state.typ,
                                    False,
                                )
                            },
                        ),
                    )
            fixed_res.append(r)
        return fixed_res

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        """Assumes that all parsers are either terminal " ," or DefiningIdentifierParserState with a type"""
        if len(self.parsed_states) > 0:
            parser = super().init_class_at_pos_hook(pos)[0]
            if isinstance(parser, TerminalParserState) and (
                parser.target_value in (" ,", " )")
            ):
                last_parsed_state = self.parsed_states[-1]
                if (
                    isinstance(last_parsed_state, ConcatParserState)
                    and last_parsed_state.parsed_states
                    and isinstance(
                        last_parsed_state.parsed_states[0], TerminalParserState
                    )
                    and last_parsed_state.parsed_states[0].target_value == " ["
                ):
                    # TODO here we first handle the special case of 2-element tuples, need to generalize
                    first_arg = last_parsed_state.parsed_states[1]
                    second_arg = last_parsed_state.parsed_states[3]
                    return (
                        parser,
                        replace(
                            self,
                            identifiers=union_dict(
                                self.identifiers,
                                {
                                    first_arg.id_name: (
                                        first_arg.typ,
                                        False,
                                    ),
                                    second_arg.id_name: (
                                        second_arg.typ,
                                        False,
                                    ),
                                },
                            ),
                        ),
                    )
                else:
                    if (
                        isinstance(last_parsed_state, ConcatParserState)
                        and last_parsed_state.parsed_states
                        and isinstance(
                            last_parsed_state.parsed_states[0],
                            DefiningIdentifierParserState,
                        )
                    ):
                        last_parsed_state = last_parsed_state.parsed_states[0]
                    return (
                        parser,
                        replace(
                            self,
                            identifiers=union_dict(
                                self.identifiers,
                                {
                                    last_parsed_state.id_name: (
                                        last_parsed_state.typ,
                                        False,
                                    )
                                },
                            ),
                        ),
                    )
            return (
                parser,
                self,
            )
        return super().init_class_at_pos_hook(pos)


def parser_for_fixed_fun_typ(fun_typ: FunctionPType, allow_no_brackets=True):
    """
    Generates a parser for the argument list of the function type
    accepts:
    (typ1, typ2, ...)
    (typ1) if only one type
    typ1 if only one type and allow_no_brackets
    """
    fixed_call_parsers = []
    for i in range(fun_typ.optional_args + 1):
        custom_parsers = [
            partial(TerminalParserState, target_value=" ("),
        ]
        truncated_sig = fun_typ.call_signature[: -i if i > 0 else None]
        for j, arg_typ in enumerate(truncated_sig):
            parse_classes = [
                partial(
                    DefiningIdentifierParserState,
                    typ=arg_typ,
                ),
                # also allow specifying the argument type (which is represented by the str of the type)
                partial(
                    ConcatParserState,
                    parse_classes=[
                        partial(
                            DefiningIdentifierParserState,
                            typ=arg_typ,
                        ),
                        partial(
                            TerminalParserState,
                            target_value=f" : {arg_typ}",
                        ),
                    ],
                ),
            ]
            if j == 0 and isinstance(fun_typ, ReduceFunctionPType):
                parse_classes.pop(0)
            if isinstance(arg_typ, TuplePType):
                # TODO here we first handle the special case of 2-element tuples, need to generalize
                assert len(arg_typ.types) == 2
                parse_classes.append(
                    partial(
                        ConcatParserState,
                        parse_classes=[
                            partial(TerminalParserState, target_value=" ["),
                            partial(
                                DefiningIdentifierParserState, typ=arg_typ.types[0]
                            ),
                            partial(TerminalParserState, target_value=" ,"),
                            partial(
                                DefiningIdentifierParserState, typ=arg_typ.types[1]
                            ),
                            partial(TerminalParserState, target_value=" ]"),
                        ],
                    )
                )
            custom_parsers.extend(
                [
                    partial(
                        UnionParserState,
                        parse_classes=parse_classes,
                    ),
                    partial(TerminalParserState, target_value=" ,"),
                ]
            )
        if custom_parsers:
            custom_parsers.pop()
        fixed_call_parsers.append(
            partial(
                FixedCallSignatureParserState,
                parse_classes=[
                    *custom_parsers,
                    partial(TerminalParserState, target_value=" )"),
                ],
            ),
        )
        if len(truncated_sig) == 1 and allow_no_brackets:
            fixed_call_parsers.append(
                partial(
                    FixedCallSignatureParserState,
                    parse_classes=[
                        partial(
                            DefiningIdentifierParserState,
                            typ=truncated_sig[0],
                        ),
                    ],
                )
            )

    if len(fixed_call_parsers) == 1:
        return fixed_call_parsers[0]
    return partial(
        UnionParserState,
        parse_classes=fixed_call_parsers,
        pass_type=False,
    )


@fnr_dataclass()
class UntypedLambdaExprParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            CallSignatureParserState,
            partial(TerminalParserState, target_value=" =>"),
            ExpressionParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "right")
    typ: FunctionPType = None

    @property
    def described_type(self):
        if len(self.parsed_states) < 3:
            return self.typ
        return FunctionPType(
            self.typ.call_signature,
            self.parsed_states[2].described_type,
            self.typ.optional_args,
        )

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if len(self.parsed_states) < 2 and as_pattern:
            return self.typ == as_pattern[0][0]
        elif len(self.parsed_states) < 3 and as_pattern and self.active is not None:
            # bascially if the return type of this lambda will instantiate the pattern
            # TODO need to assert?
            return self.active.derivable(
                goal,
                MIN_OPERATOR_PRECEDENCE,
                as_array,
                as_nested_expression,
                [
                    (
                        # TODO
                        TypeParameterPType(list(as_pattern[0][0].type_params())[0]),
                        as_pattern[0][0],
                        min_operator_precedence,
                    )
                ]
                + as_pattern,
                recursing_path,
            )
        return extract_type_params(goal, self.typ)[0]

    def parse_char(self, char: str) -> List[Self]:
        """
        Ensure that an expected type is given and that that type is a function type
        """
        if self.typ is not None and isinstance(self.typ, UnionPType):
            sum = []
            for typ in self.typ.types:
                sum.extend(replace(self, typ=typ).parse_char(char))
            return sum
        if self.typ is None or not isinstance(self.typ, FunctionPType):
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        own_typ: FunctionPType = self.typ
        if pos == 0:
            # Construct custom call signature parser based on type
            call_parser = parser_for_fixed_fun_typ(own_typ)

            return (
                call_parser(
                    identifiers=self.identifiers,
                ),
                self,
            )
        elif pos == 2:
            return (
                ExpressionParserState(
                    typ=own_typ.return_type,
                    identifiers=union_dict(
                        self.identifiers,
                        self.parsed_states[0].identifiers,
                    ),
                    max_steps=self.max_steps,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class UntypedLambdaFunParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            CallSignatureParserState,
            partial(TerminalParserState, target_value=" => {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "right")

    @property
    def described_type(self):
        if len(self.parsed_states) < 4:
            return self.typ
        return FunctionPType(
            call_signature=self.typ.call_signature,
            return_type=extract_return_type_stmts(self.parsed_states[2]) or VoidPType(),
        )

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if len(self.parsed_states) < 2 and as_pattern:
            return self.typ == as_pattern[0][0]
        elif len(self.parsed_states) < 3 and as_pattern and self.active is not None:
            active_return = extract_active_return(self.active)
            # bascially if the return type of this lambda will instantiate the pattern
            # -> need to find current return stmt and run derivable from there
            # if no return stmt yet, use AnyPType (#TODO union with already placed return)
            if active_return is None or (
                isinstance(active_return, ReturnValueParserState)
                and (
                    active_return.active is None
                    or len(active_return.parsed_states) < 2
                    or 3 >= len(active_return.parsed_states)
                )
            ):
                start_t = AnyPType()
            elif isinstance(active_return, ReturnVoidParserState):
                start_t = VoidPType()
            else:
                start_t = None
            if start_t is not None:
                return reachable(
                    start_t,
                    goal,
                    MIN_OPERATOR_PRECEDENCE,
                    MAX_OPERATOR_PRECEDENCE,
                    as_array,
                    as_nested_expression,
                    [
                        (
                            # TODO
                            TypeParameterPType(list(as_pattern[0][0].type_params())[0]),
                            as_pattern[0][0],
                            min_operator_precedence,
                        )
                    ]
                    + as_pattern,
                    max_steps=self.max_steps,
                )
            return active_return.active.derivable(
                goal,
                MIN_OPERATOR_PRECEDENCE,
                as_array,
                as_nested_expression,
                [
                    (
                        # TODO
                        TypeParameterPType(list(as_pattern[0][0].type_params())[0]),
                        as_pattern[0][0],
                        min_operator_precedence,
                    )
                ]
                + as_pattern,
                recursing_path,
                max_steps=self.max_steps,
            )
        return extract_type_params(goal, self.typ)[0]

    def parse_char(self, char: str) -> List[Self]:
        """
        Ensure that an expected type is given and that that type is a function type
        """
        if self.typ is not None and isinstance(self.typ, UnionPType):
            sum = []
            for typ in self.typ.types:
                sum.extend(replace(self, typ=typ).parse_char(char))
            return sum
        if self.typ is None or not isinstance(self.typ, FunctionPType):
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        own_typ: FunctionPType = self.typ
        if pos == 0:
            # Construct custom call signature parser based on type
            call_parser = parser_for_fixed_fun_typ(own_typ)

            return (
                call_parser(
                    identifiers=self.identifiers,
                ),
                self,
            )
        elif pos == 2:
            return (
                StmtsParserState(
                    identifiers=union_dict(
                        self.identifiers,
                        self.parsed_states[0].identifiers,
                    ),
                    return_type=own_typ.return_type,
                    has_to_return=not isinstance(own_typ.return_type, VoidPType)
                    and not isinstance(own_typ.return_type, BaseTsObject),
                    in_loop=False,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class UntypedAnonymousFunParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" function"),
            CallSignatureParserState,
            partial(TerminalParserState, target_value=" {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "right")

    @property
    def described_type(self):
        if len(self.parsed_states) < 5:
            return self.typ
        return FunctionPType(
            call_signature=self.typ.call_signature,
            return_type=extract_return_type_stmts(self.parsed_states[3]) or VoidPType(),
        )

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return extract_type_params(goal, self.typ)[0]

    def parse_char(self, char: str) -> List[Self]:
        """
        Ensure that an expected type is given and that that type is a function type
        """
        if self.typ is not None and isinstance(self.typ, UnionPType):
            sum = []
            for typ in self.typ.types:
                sum.extend(replace(self, typ=typ).parse_char(char))
            return sum
        if self.typ is None or not isinstance(self.typ, FunctionPType):
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        own_typ: FunctionPType = self.typ
        if pos == 1:
            # Construct custom call signature parser based on type
            # for anonymous functions we need to force brackets
            call_parser = parser_for_fixed_fun_typ(own_typ, False)

            return (
                call_parser(
                    identifiers=self.identifiers,
                ),
                self,
            )
        elif pos == 3:
            return (
                StmtsParserState(
                    identifiers=union_dict(
                        self.identifiers,
                        self.parsed_states[1].identifiers,
                    ),
                    return_type=own_typ.return_type,
                    has_to_return=not isinstance(own_typ.return_type, VoidPType),
                    in_loop=False,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class TypedLambdaExprParserState(ConcatParserState, DerivableTypeMixin):
    """
    parses a lambda expression with complete type signature in the header
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ("),
            CallSignatureParserState,
            partial(TerminalParserState, target_value=" :"),
            TypeParserState,
            partial(TerminalParserState, target_value=" =>"),
            ExpressionParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "right")
    typ: FunctionPType = None
    def_id_name: str = None

    @property
    def described_type(self):
        if len(self.parsed_states) < 4:
            return self.typ
        return FunctionPType(
            call_signature_from_accepted_call_signature_parser(self.parsed_states[1]),
            self.parsed_states[3].described_type,
        )

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if not isinstance(goal, AnyPType):
            return False
        return True

    def parse_char(self, char: str) -> List[Self]:
        """
        Ensure that there is no expected type, we wouldn't know how to search for a match
        """
        if self.typ is not None and not isinstance(self.typ, AnyPType):
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 5:
            own_typ: FunctionPType = self.described_type
            return (
                ExpressionParserState(
                    typ=own_typ.return_type,
                    identifiers=union_dict(
                        self.identifiers,
                        {self.def_id_name: (self.described_type, False)}
                        if self.def_id_name
                        else {},
                        identifiers_from_accepted_call_signature_parser(
                            self.parsed_states[1]
                        ),
                    ),
                    operator_precedence=self.operator_precedence,
                    max_steps=self.max_steps,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class TypedLambdaFunParserState(ConcatParserState, DerivableTypeMixin):
    """
    parses a lambda expression with complete type signature in the header
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ("),
            CallSignatureParserState,
            partial(TerminalParserState, target_value=" :"),
            TypeParserState,
            partial(TerminalParserState, target_value=" => {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "right")
    typ: FunctionPType = None
    def_id_name: str = None

    @property
    def described_type(self):
        if len(self.parsed_states) < 4:
            return self.typ
        return FunctionPType(
            call_signature_from_accepted_call_signature_parser(self.parsed_states[1]),
            self.parsed_states[3].described_type,
        )

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if not isinstance(goal, AnyPType):
            return False
        return True

    def parse_char(self, char: str) -> List[Self]:
        """
        Ensure that there is no expected type, we wouldn't know how to search for a match
        """
        if self.typ is not None and not isinstance(self.typ, AnyPType):
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 5:
            own_typ: FunctionPType = self.described_type
            return (
                StmtsParserState(
                    identifiers=union_dict(
                        self.identifiers,
                        {self.def_id_name: (self.described_type, False)}
                        if self.def_id_name
                        else {},
                        identifiers_from_accepted_call_signature_parser(
                            self.parsed_states[1]
                        ),
                    ),
                    return_type=own_typ.return_type,
                    has_to_return=not isinstance(own_typ.return_type, VoidPType),
                    in_loop=False,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class FunctionDeclarationParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" function\t"),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" ("),
            CallSignatureParserState,
            partial(TerminalParserState, target_value=" :"),
            TypeParserState,
            partial(TerminalParserState, target_value=" {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos > 1:
            fun_name = self.parsed_states[1].id_name
            if pos == 3:
                # extract function name to prevent name conflicts with params
                return (
                    CallSignatureParserState(
                        identifiers=union_dict(
                            self.identifiers, {fun_name: (AnyPType(), False)}
                        ),
                        return_type=self.return_type,
                    ),
                    self,
                )
            elif pos >= 7:
                fun_type = FunctionPType(
                    call_signature=call_signature_from_accepted_call_signature_parser(
                        self.parsed_states[3]
                    ),
                    return_type=self.parsed_states[5].described_type,
                )
                if pos == 7:
                    # extract function type and argument type to enforce in stmts
                    # Note functions are not mutable
                    # Note parameters are not mutable
                    return (
                        StmtsParserState(
                            identifiers=union_dict(
                                self.identifiers,
                                {fun_name: (fun_type, False)},
                                identifiers_from_accepted_call_signature_parser(
                                    self.parsed_states[3]
                                ),
                            ),
                            return_type=fun_type.return_type,
                            has_to_return=not isinstance(
                                fun_type.return_type, VoidPType
                            ),
                            in_loop=False,
                        ),
                        self,
                    )
                elif pos == 8:
                    # extract function type to pass on to parent
                    # Note functions are not mutable
                    return (
                        self.parse_classes[8](),
                        replace(
                            self,
                            identifiers=union_dict(
                                self.identifiers, {fun_name: (fun_type, False)}
                            ),
                        ),
                    )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class AnyStringParser(IncrementalParsingState):
    terminator_chars: List[str] = field(default_factory=lambda: [])
    accept: bool = False
    length: int = 0

    def parse_char(self, char: str) -> List[Self]:
        if self.accept or self.length >= MAX_COMMENT_LENGTH:
            return []
        if char in self.terminator_chars:
            return [replace(self, accept=True)]
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


@fnr_dataclass
class FunctionCallParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ("),
            None,
        ),
        repr=False,
    )
    # the function being called
    lhs: DerivableTypeMixin = None
    fn_typ: FunctionPType = None
    operator_precedence: OperatorPrecedence = (17, "left")

    @property
    def described_type(self):
        if self.fn_typ is None:
            return None
        return self.fn_typ.return_type

    def parse_char(self, char: str) -> List[Self]:
        if self.fn_typ is None:
            # check whether fn is actually a function
            fn_typ = self.lhs.described_type
            if fn_typ is None:
                return []
            if isinstance(fn_typ, UnionPType) and all(
                isinstance(x, FunctionPType) for x in fn_typ.types
            ):
                return sum(
                    (replace(self, fn_typ=t).parse_char(char) for t in fn_typ.types), []
                )
            if not isinstance(fn_typ, FunctionPType):
                return []
            return replace(self, fn_typ=fn_typ).parse_char(char)
        states = super().parse_char(char)
        if (
            len(self.parsed_states) >= 1
            and self.active is not None
            and self.fn_typ.type_params()
            and not isinstance(
                self.active, TerminalParserState
            )  # edge case where we dont have params
        ):
            # try to match already parsed params with the type param
            # update the own fn typ accordingly
            active: ConcatParserState = self.active
            accepted_expression_parser = (
                active.parsed_states[-1] if active.parsed_states else None
            )
            nth_param = (
                len(
                    [
                        ps
                        for ps in active.parsed_states
                        if isinstance(ps, ExpressionParserState)
                    ]
                )
                - 1
            )
            if accepted_expression_parser is not None and isinstance(
                accepted_expression_parser, ExpressionParserState
            ):
                typ = accepted_expression_parser.described_type
                matches, extracted_params = extract_type_params(
                    self.fn_typ.call_signature[nth_param], typ
                )
                if not matches:
                    return []
                if extracted_params:
                    updated_fun_typ = self.fn_typ.instantiate_type_params(
                        extracted_params
                    )
                    updated_fun_classes = list(active.parse_classes).copy()
                    for i, at in enumerate(
                        updated_fun_typ.call_signature[
                            nth_param + 1 : len(updated_fun_classes) // 2
                        ],
                        start=nth_param + 1,
                    ):
                        updated_param_parser = partial(ExpressionParserState, typ=at)
                        updated_fun_classes[2 * i] = updated_param_parser
                    updated_active = replace(active, parse_classes=updated_fun_classes)
                    updated_fun_parser = replace(
                        self, fn_typ=updated_fun_typ, active=updated_active
                    )
                    states.extend(updated_fun_parser.parse_char(char))
        # we can not accept calls where the type params are not instantiated
        return [
            s
            for s in states
            # remove states where the function type is invalid entirely
            if not s.accept or (s.fn_typ is not None and not s.fn_typ.type_params())
        ]

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        fn_typ = self.fn_typ
        if fn_typ is None:
            fn_typ = self.lhs.described_type
        if fn_typ is None or not isinstance(fn_typ, FunctionPType):
            # in this case the "function" is not actually a function
            return False
        # if we are currently parsing call params and have type params
        # run reachable from the call param
        # type parameter in currently parsing call param is relevant
        if (
            len(self.parsed_states) == 1
            and self.active is not None
            and self.fn_typ.type_params()
            and not isinstance(
                self.active, TerminalParserState
            )  # edge case where we don't have params
        ):
            # try to match already parsed params with the type param
            # update the own fn typ accordingly
            active: ConcatParserState = self.active
            active_expression_parser = active.active
            nth_param = len(
                [
                    ps
                    for ps in active.parsed_states
                    if isinstance(ps, ExpressionParserState)
                ]
            )
            if (
                active_expression_parser is not None
                and isinstance(active_expression_parser, ExpressionParserState)
                and fn_typ.call_signature[nth_param].type_params()
            ):
                return active_expression_parser.derivable(
                    goal,
                    MIN_OPERATOR_PRECEDENCE,
                    as_array,
                    as_nested_expression,
                    [
                        (
                            fn_typ.call_signature[nth_param],
                            fn_typ.return_type,
                            min_operator_precedence,
                        )
                    ]
                    + as_pattern,
                    recursing_path,
                )
        return reachable(
            fn_typ.return_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            # initialize a number of parsers for the different number of optional arguments (in each enforced to a specific number)
            argument_types = self.fn_typ.call_signature
            all_pcs = []
            for i in range(self.fn_typ.optional_args + 1):
                pcs = []
                for at in argument_types[: -i if i != 0 else None]:
                    expr = make_expr_class([], typ=at)
                    pcs.extend([partial(TerminalParserState, target_value=" ,"), expr])
                if not pcs:
                    right_paren = " )"
                    if (
                        isinstance(self.lhs, MemberAccessStateParser)
                        and self.lhs.parsed_states[1].id_name in ("shift", "pop")
                        and isinstance(self.lhs.lhs.described_type, ArrayPType)
                    ):
                        right_paren = " )!"
                    pc = partial(TerminalParserState, target_value=right_paren)
                else:
                    pcs.pop(0)
                    # if we allow inf_args, the last parser could also accept spread
                    if i == 0 and self.fn_typ.inf_args:
                        pcs[-1] = partial(
                            UnionParserState,
                            parse_classes=[
                                SpreadOpParserState,
                                ExpressionParserState,
                            ],
                            typ=argument_types[-1],
                            pass_type=True,
                        )
                    right_paren = " )"
                    if (
                        isinstance(self.lhs, ExistingIdentifierParserState)
                        and self.lhs.id_name == "String"
                    ):
                        right_paren = " ).toString()"
                    if (
                        isinstance(self.lhs, MemberAccessStateParser)
                        and self.lhs.parsed_states[1].id_name in ("find", "findLast")
                        and isinstance(self.lhs.lhs.described_type, ArrayPType)
                    ):
                        right_paren = " )!"
                    if (
                        isinstance(self.lhs, MemberAccessStateParser)
                        and self.lhs.parsed_states[1].id_name == "get"
                        and isinstance(self.lhs.lhs.described_type, MapPType)
                    ):
                        right_paren = " )!"
                    if (
                        isinstance(self.lhs, MemberAccessStateParser)
                        and self.lhs.parsed_states[1].id_name == "match"
                        and isinstance(self.lhs.lhs.described_type, StringPType)
                    ):
                        right_paren = " )!"
                    pc = partial(
                        ConcatParserState,
                        parse_classes=pcs
                        + [partial(TerminalParserState, target_value=right_paren)],
                    )
                all_pcs.append(pc)
                if i == 0 and self.fn_typ.inf_args:
                    # Further append a plus parser to allow infinitely many more args
                    pcs = pcs.copy()
                    pcs.append(
                        partial(
                            PlusParserState,
                            parse_class=partial(
                                ConcatParserState,
                                parse_classes=[
                                    partial(TerminalParserState, target_value=" ,"),
                                    partial(
                                        UnionParserState,
                                        parse_classes=[
                                            SpreadOpParserState,
                                            ExpressionParserState,
                                        ],
                                        typ=argument_types[-1],
                                        pass_type=True,
                                    ),
                                ],
                            ),
                        ),
                    )
                    pc = partial(
                        ConcatParserState,
                        parse_classes=pcs
                        + [partial(TerminalParserState, target_value=" )")],
                    )
                    all_pcs.append(pc)

            return (
                UnionParserState(
                    parse_classes=all_pcs,
                    identifiers=self.identifiers,
                    pass_type=False,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class FunctionParameterizationParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" <"),
            None,
        ),
        repr=False,
    )
    # the function to parameterize
    lhs: DerivableTypeMixin = None
    operator_precedence: OperatorPrecedence = (17, "left")

    @property
    def fn_typ(self):
        typ = self.lhs.described_type
        if typ is None:
            return None
        if isinstance(typ, UnionPType):
            for t in typ.types:
                if isinstance(t, FunctionPType) and t.type_params():
                    typ = t
        if not isinstance(typ, FunctionPType) or not typ.type_params():
            return None
        return typ

    @property
    def described_type(self):
        if self.fn_typ is None:
            return None
        if len(self.parsed_states) == 1 and self.active or len(self.parsed_states) == 2:
            # incrementally insert type parameters in fn_typ
            active: ConcatParserState = self.active or self.parsed_states[-1]
            type_params = sorted(self.fn_typ.type_params())
            update_dicts = {}
            for param, parsed_state in zip(
                type_params,
                (
                    s
                    for s in active.parsed_states
                    if not isinstance(s, TerminalParserState)
                ),
            ):
                update_dicts[param] = parsed_state.described_type
            return self.fn_typ.instantiate_type_params(update_dicts)
        return self.fn_typ

    def parse_char(self, char: str) -> List[Self]:
        if self.fn_typ is None:
            return []
        return super().parse_char(char)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        fn_typ = self.described_type
        if fn_typ is None:
            return False
        return reachable(
            fn_typ,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            # initialize a parser for the type arguments
            # TODO we assume here that type parameters are named alphabetically, but this can be easily enforces through renaming
            argument_types = sorted(self.fn_typ.type_params())
            pcs = []
            for _ in argument_types:
                pcs.extend(
                    [
                        partial(TerminalParserState, target_value=" ,"),
                        partial(TypeParserState),
                    ]
                )
            pcs.pop(0)
            pc = ConcatParserState(
                parse_classes=pcs + [partial(TerminalParserState, target_value=" >")]
            )
            return pc, self
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class ComputedMemberAccessParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ["),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" ]"),
        ),
        repr=False,
    )
    # the array in which we access
    lhs: DerivableTypeMixin = None
    operator_precedence: int = (17, "left")

    @property
    def element_type(self):
        # check whether element is actually an array
        if self.lhs is None:
            return None
        if isinstance(self.lhs.described_type, StringPType):
            return StringPType()
        if isinstance(self.lhs.described_type, IndexSignaturePType):
            return self.lhs.described_type.value_type
        if (
            not isinstance(self.lhs.described_type, ArrayPType)
            or self.lhs.described_type.element_type == GenericPType()
        ):
            return None
        return self.lhs.described_type.element_type

    @property
    def described_type(self):
        return self.element_type

    def parse_char(self, char: str) -> List[Self]:
        if self.element_type is None:
            return []
        return super().parse_char(char)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        element_typ = self.element_type
        if element_typ is None:
            return False
        return reachable(
            element_typ,
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            return (
                ExpressionParserState(
                    typ=(
                        NumberPType()
                        if not isinstance(self.lhs.described_type, IndexSignaturePType)
                        else self.lhs.described_type.key_type
                    ),
                    identifiers=self.identifiers,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class ComputedMemberGenericAssignmentParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ExistingIdentifierParserState,
            partial(TerminalParserState, target_value=" ["),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" ] ="),
            ExpressionParserState,
            EOLParserState,
        ),
        repr=False,
    )
    operator_precedence: int = (17, "left")

    def parse_char(self, char: str) -> List[Self]:
        if len(self.parsed_states) == 1 and self.active is None:
            arr_typ = self.parsed_states[0].described_type
            if not (
                isinstance(arr_typ, ArrayPType)
                and isinstance(arr_typ.element_type, GenericPType)
            ):
                return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 2:
            return (
                ExpressionParserState(
                    typ=(NumberPType()),
                    identifiers=self.identifiers,
                ),
                self,
            )

        if pos == 5:
            parser, self2 = super().init_class_at_pos_hook(pos)
            updated_name = self.parsed_states[0].id_name
            updated_el_type = self.parsed_states[4].described_type
            self2 = replace(
                self2,
                identifiers=union_dict(
                    self2.identifiers,
                    {
                        updated_name: (
                            ArrayPType(updated_el_type),
                            self2.identifiers[updated_name][0],
                        )
                    },
                ),
            )
            return parser, self2
        return super().init_class_at_pos_hook(pos)


ListEndParserState = partial(
    UnionParserState,
    parse_classes=[
        partial(TerminalParserState, target_value=" ]"),
        partial(TerminalParserState, target_value=" , ]"),
    ],
)


@fnr_dataclass
class TupleComputedMemberAccessParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ["),
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" 0"),
                    partial(TerminalParserState, target_value=" 1"),
                    partial(TerminalParserState, target_value=" 2"),
                ],
            ),
            ListEndParserState,
        ),
        repr=False,
    )
    # the array in which we access
    lhs: DerivableTypeMixin = None
    operator_precedence: int = (17, "left")

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            tuple_size = len(self.lhs.described_type.types)
            parse_classes = [
                partial(TerminalParserState, target_value=f" {i}")
                for i in range(tuple_size)
            ]
            return UnionParserState(parse_classes=parse_classes), self
        return super().init_class_at_pos_hook(pos)

    @property
    def described_type(self):
        lhs_types = self.lhs.described_type.types
        if len(self.parsed_states) == 0:
            return merge_typs(*lhs_types)
        elif len(self.parsed_states) == 1:
            if self.active is None:
                return merge_typs(*lhs_types)
            else:
                index = int(self.active.target_value.strip())
                return lhs_types[index]
        else:
            index = int(self.parsed_states[1].target_value.strip())
            return lhs_types[index]

    def parse_char(self, char: str) -> List[Self]:
        # can not access if not a tuple or tuple empty
        if (
            not isinstance(self.lhs.described_type, TuplePType)
            or not self.lhs.described_type.types
        ):
            return []
        return super().parse_char(char)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            MAX_OPERATOR_PRECEDENCE,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass
class TernaryOpParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ?"),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" :"),
            ExpressionParserState,
        ),
        repr=False,
    )
    # the array in which we access
    lhs: DerivableTypeMixin = None
    operator_precedence: OperatorPrecedence = (2, "right")

    def parse_char(self, char: str) -> List[Self]:
        # disallow parsing a "." directly after the "?" (would be optional chaining)
        if not any(
            isinstance(self.lhs.described_type, S)
            for S in (UnionPType, NumberPType, StringPType, BooleanPType, ArrayPType)
        ):
            return []
        if len(self.parsed_states) == 1 and self.active is None and char == ".":
            return []
        return super().parse_char(char)

    @property
    def described_type(self):
        if len(self.parsed_states) < 2:
            return None
        l_described_type = self.parsed_states[1].described_type
        if len(self.parsed_states) < 4:
            return l_described_type
        r_described_type = self.parsed_states[3].described_type
        return merge_typs(l_described_type, r_described_type)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        own_described_type = AnyPType()
        if len(self.parsed_states) == 1:
            active = self.active
            if active is None:
                active = self.init_class_at_pos_hook(1)[0]
            return active.derivable(
                goal,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
        # own type should be at most as strong as x in y?x:z (final result will be union)
        if self.described_type is not None:
            own_described_type = self.described_type
        return reachable(
            own_described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            return (
                ExpressionParserState(
                    typ=self.typ,
                    identifiers=self.identifiers,
                    max_steps=self.max_steps,
                ),
                self,
            )
        if pos == 3:
            return (
                ExpressionParserState(
                    # we don't enforce that the two return types match here
                    typ=self.typ,
                    identifiers=self.identifiers,
                    max_steps=self.max_steps,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class MemberAccessStateParser(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ."),
            IdentifierParserState,
        ),
        repr=False,
    )
    # the object into which we access
    lhs: DerivableTypeMixin = None
    attributes: Dict[str, Tuple[PType, bool]] = None
    operator_precedence: OperatorPrecedence = (17, "left")
    force_mutable: bool = False

    def parse_char(self, char: str) -> List[Self]:
        if (
            len(self.parsed_states) == 0
            and not self.active
            and isinstance(self.lhs, NumericalLiteralParserState)
        ):
            # edge case: it is not allowed to write 0.toString()
            return []
        return super().parse_char(char)

    @property
    def is_mutable(self):
        """Helper function to determine whether the accessed attribute is mutable"""
        if self.attributes is None:
            return False
        return self.attributes[self.parsed_states[1].id_name][1]

    @property
    def described_type(self):
        if self.attributes is None or len(self.parsed_states) < 2:
            return None
        # TODO narrow down to actual type
        return self.attributes[self.parsed_states[1].id_name][0]

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if len(self.parsed_states) == 0 or (
            len(self.parsed_states) == 1 and self.active is None
        ):
            attributes = self.lhs.described_type.attributes
            return any_reachable(
                set(x for x, _ in attributes.values()),
                goal,
                min_operator_precedence,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                max_steps=self.max_steps,
            )
        if len(self.parsed_states) == 1:
            return self.active.derivable(
                goal,
                min_operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
        return self.parsed_states[1].derivable(
            goal,
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            attributes = self.lhs.described_type.attributes
            return (
                ExistingIdentifierParserState(
                    identifiers={k: v for k, v in attributes.items()},
                    force_mutable=self.force_mutable,
                    max_steps=self.max_steps,
                ),
                replace(self, attributes=attributes),
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class OptionalChainingStateParser(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ?."),
            IdentifierParserState,
        ),
        repr=False,
    )
    # the object into which we access
    lhs: DerivableTypeMixin = None
    attributes: Dict[str, Tuple[PType, bool]] = None
    operator_precedence: OperatorPrecedence = (17, "left")
    force_mutable: bool = False

    def parse_char(self, char: str) -> List[Self]:
        if not isinstance(self.lhs.described_type, UnionPType) or not any(
            t in FALSEY_TYPES for t in self.lhs.described_type.types
        ):
            # forbid optional chaining on something that is not a potentially falsey types
            return []
        return super().parse_char(char)

    @property
    def is_mutable(self):
        """Helper function to determine whether the accessed attribute is mutable"""
        return False

    @property
    def described_type(self):
        if self.attributes is None or len(self.parsed_states) < 2:
            return None
        typ = self.parsed_states[1].described_type
        return merge_typs(typ, UndefinedPType())

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if self.active is None or len(self.parsed_states) == 0:
            attributes = self.lhs.described_type.attributes
            lhs_typ = self.lhs.described_type
            # if lhs contains undefined as union
            attribute_typs = set(x for x, _ in attributes.values())
            if isinstance(lhs_typ, UnionPType):
                actual_lhs_typ = merge_typs(*lhs_typ.types.difference(FALSEY_TYPES))
                attribute_typs = set(x for x, _ in actual_lhs_typ.attributes.values())
                # actually anything reachable from here | undefined is what can be reached
            return any_reachable(
                attribute_typs,
                goal,
                min_operator_precedence,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                [
                    (
                        TypeParameterPType("T"),
                        UnionPType([TypeParameterPType("T"), UndefinedPType()]),
                        min_operator_precedence,
                    )
                ]
                + as_pattern,
                max_steps=self.max_steps,
            )
        if len(self.parsed_states) == 1:
            # wrapping this into union undefined
            return self.active.derivable(
                goal,
                (16, "left"),
                as_array,
                as_nested_expression,
                [
                    (
                        TypeParameterPType("T"),
                        UnionPType([TypeParameterPType("T"), UndefinedPType()]),
                        min_operator_precedence,
                    )
                ]
                + as_pattern,
                recursing_path,
            )
        return self.parsed_states[1].derivable(
            goal,
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            attributes = self.lhs.described_type.attributes
            lhs_typ = self.lhs.described_type
            actual_lhs_typ = merge_typs(*lhs_typ.types.difference(FALSEY_TYPES))
            attributes = actual_lhs_typ.attributes

            return (
                ExpressionParserState(
                    active=[
                        ExistingIdentifierParserState(
                            identifiers={k: v for k, v in attributes.items()},
                            force_mutable=self.force_mutable,
                        ),
                    ],
                    rhs_operator_precedence=(16, "left"),
                    max_steps=self.max_steps,
                ),
                replace(self, attributes=attributes),
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class SpreadOpParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ..."),
            ExpressionParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (2, "right")
    pass_type: bool = True

    @property
    def described_type(self):
        if len(self.parsed_states) < 2:
            return None
        else:
            return self.parsed_states[1].described_type.element_type

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            return (
                UnionParserState(
                    parse_classes=[
                        partial(
                            ExpressionParserState,
                            typ=ArrayPType(
                                GenericPType()
                                if (self.typ is None or self.typ == AnyPType())
                                else self.typ
                            ),
                            max_steps=self.max_steps,
                        ),
                        partial(
                            ExpressionParserState,
                            typ=SetPType(
                                GenericPType()
                                if (self.typ is None or self.typ == AnyPType())
                                else self.typ
                            ),
                            max_steps=self.max_steps,
                        ),
                    ],
                    identifiers=self.identifiers,
                    pass_type=False,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if len(self.parsed_states) == 0:
            s = self.init_class_at_pos_hook(1)[0]
        elif len(self.parsed_states) == 1:
            s = self.active
            if s is None:
                s = self.init_class_at_pos_hook(1)[0]
        else:
            s = self.parsed_states[1]
        return s.derivable(
            merge_typs(ArrayPType(goal), SetPType(goal))
            if not isinstance(goal, AnyPType)
            else merge_typs(
                ArrayPType(TypeParameterPType("X")), SetPType(TypeParameterPType("X"))
            ),
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )


# ALLOWED_ARITH_OPS[T1][T2] = {(op1, T3), ...} means that sig of op1 is (T1, T2) -> T3
ALLOWED_ARITH_OPS = {
    NumberPType(): {
        NumberPType(): {(op, NumberPType()) for op in ARITHMETIC_OPS},
        StringPType(): {("+", StringPType())},
    },
    BigIntPType(): {
        BigIntPType(): {(o, BigIntPType()) for o in ARITHMETIC_OPS - {">>>"}}
    },
    StringPType(): {AnyPType(): {("+", StringPType())}},
    AnyPType(): {StringPType(): {("+", StringPType())}},
}


def extract_array_type_update_from_stmt(parser: IncrementalParsingState, identifiers):
    # check if we call array.push
    update_dict = {}
    if isinstance(parser, ExpressionParserState) and isinstance(
        parser.active[0], FunctionCallParserState
    ):
        call: FunctionCallParserState = parser.active[0]
        called_fun = None
        if isinstance(call.lhs, MemberAccessStateParser) and isinstance(
            call.lhs.parsed_states[1], ExistingIdentifierParserState
        ):
            called_fun = call.lhs.parsed_states[1].id_name
        updated_array_name = None
        updated_array_el_type = None
        if called_fun == "push":
            updated_array = call.lhs.lhs
            updated_array_type = updated_array.described_type
            if (
                isinstance(updated_array, ExistingIdentifierParserState)
                and isinstance(updated_array_type, ArrayPType)
                and isinstance(updated_array_type.element_type, GenericPType)
            ):
                updated_array_name = updated_array.id_name
                updated_array_el_type = (
                    call.parsed_states[1].parsed_states[0].described_type
                )
        if updated_array_name is not None and updated_array_el_type is not None:
            # update the type of this identifier
            update_dict[updated_array_name] = (
                ArrayPType(updated_array_el_type),
                identifiers[updated_array_name][1],
            )
    return update_dict


def make_falsey(identifier, identifiers, only_these=tuple(FALSEY_TYPES)):
    identifier_type, identifier_mutable = identifiers[identifier]
    if isinstance(identifier_type, UnionPType) and any(
        ft in identifier_type.types for ft in only_these
    ):
        return {
            identifier: (
                # merge_typs(*identifier_type.types.intersection(only_these)),
                # dont actually cast to the falsey type, being that type is quite uninteresting
                merge_typs(*identifier_type.types),
                identifier_mutable,
            )
        }, {
            identifier: (
                merge_typs(*identifier_type.types.difference(only_these)),
                identifier_mutable,
            )
        }
    return {}, {}


def make_non_falsey(identifier, identifiers, only_these=tuple(FALSEY_TYPES)):
    ifb, elseb = make_falsey(identifier, identifiers)
    return elseb, ifb


def extract_type_cast_from_expression(parser: DerivableTypeMixin, identifiers: dict):
    """
    For an expression, extract the resulting type case in the (if-branch, else-branch)
    """
    if isinstance(parser, FixedOpParserState) and parser.op in ("==", "==="):
        # check if either one side is ExistingIdentifier and the other is undefined
        # one side is lhs and the other is parser.parsed_states[1]
        checked_type_rhs = parser.parsed_states[1].described_type
        checked_type_lhs = parser.lhs.described_type
        if (
            isinstance(parser.lhs, ExistingIdentifierParserState)
            and checked_type_rhs in FALSEY_TYPES
        ):
            identifier = parser.lhs.id_name
            return make_falsey(identifier, identifiers, (checked_type_rhs,))
        elif (
            isinstance(parser.parsed_states[1], ExistingIdentifierParserState)
            and checked_type_lhs in FALSEY_TYPES
        ):
            identifier = parser.parsed_states[1].id_name
            return make_falsey(identifier, identifiers, (checked_type_lhs,))
    elif isinstance(parser, FixedOpParserState) and parser.op in ("!=", "!=="):
        # check if either one side is ExistingIdentifier and the other is undefined
        # one side is lhs and the other is parser.parsed_states[1]
        checked_type_rhs = parser.parsed_states[1].described_type
        checked_type_lhs = parser.lhs.described_type
        if (
            isinstance(parser.lhs, ExistingIdentifierParserState)
            and checked_type_rhs in FALSEY_TYPES
        ):
            identifier = parser.lhs.id_name
            return make_non_falsey(identifier, identifiers, (checked_type_rhs,))
        elif (
            isinstance(parser.parsed_states[1], ExistingIdentifierParserState)
            and checked_type_lhs in FALSEY_TYPES
        ):
            identifier = parser.parsed_states[1].id_name
            return make_non_falsey(identifier, identifiers, (checked_type_lhs,))
    elif isinstance(parser, ExistingIdentifierParserState):
        identifier = parser.id_name
        return make_non_falsey(identifier, identifiers)
    elif (
        isinstance(parser, UnopExpressionParserState)
        and parser.parsed_states[0].target_value == " !"
    ):
        if isinstance(parser.parsed_states[1], ExpressionParserState) and isinstance(
            parser.parsed_states[1].active[0], ExistingIdentifierParserState
        ):
            identifier = parser.parsed_states[1].active[0].id_name
            return make_falsey(identifier, identifiers)
    # TODO recurse
    return {}, {}


@fnr_dataclass
class LogicOpParserState(ConcatParserState, DerivableTypeMixin):
    lhs: DerivableTypeMixin = None
    # override this field in subclass
    logic_op: str = None
    operator_precedence: OperatorPrecedence = None
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TerminalParserState,  # replaced in post_init
            ExpressionParserState,
        ),
        repr=False,
    )

    def parse_char(self, char: str) -> List[Self]:
        # forbid using this for non-falsy values/change inferred type
        if not any(
            isinstance(self.lhs.described_type, S)
            for S in (
                BooleanPType,
                NumberPType,
                StringPType,
                UnionPType,
                ArrayPType,
            )
        ):
            # if the lhs has a type that is never false, we disallow logic operators
            return []
        return super().parse_char(char)

    @property
    def described_type(self):
        l_described_type = self.lhs.described_type
        # For or/?? only preserve truthy types
        if self.logic_op in ("||", "??"):
            if isinstance(l_described_type, UnionPType):
                new_types = set(l_described_type.types)
                for typ in FALSEY_TYPES:
                    if typ in new_types:
                        new_types.remove(typ)
                l_described_type = merge_typs(*new_types)
        # For && only preserve falsy types
        if self.logic_op in ("&&",):
            if isinstance(l_described_type, UnionPType):
                new_types = set()
                for typ in FALSEY_TYPES:
                    new_types.add(typ)
                l_described_type = merge_typs(*new_types)
        if len(self.parsed_states) < 2:
            return None
        r_described_type = self.parsed_states[1].described_type
        return merge_typs(l_described_type, r_described_type)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        own_described_type = AnyPType()
        if self.described_type is not None:
            own_described_type = self.described_type
        return reachable(
            own_described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            return (TerminalParserState(target_value=f" {self.logic_op}"), self)
        if pos == 1:
            update_to_identifiers_if, update_to_identifiers_else = (
                extract_type_cast_from_expression(self.lhs, self.identifiers)
            )
            parsed_op = self.logic_op
            return (
                make_expr_class(
                    excluded=[TypedLambdaExprParserState, TypedLambdaFunParserState],
                    # we don't enforce that the two return types match here
                    typ=self.typ,
                    identifiers=union_dict(
                        self.identifiers,
                        (
                            update_to_identifiers_if
                            if parsed_op == "&&"
                            else update_to_identifiers_else
                        ),
                    ),
                    operator_precedence=self.operator_precedence,
                    max_steps=self.max_steps,
                )(),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class AndLogicOpParserState(LogicOpParserState):
    logic_op: str = "&&"
    operator_precedence: OperatorPrecedence = LOGIC_OPS_PRECEDENCE["&&"]


@fnr_dataclass
class OrLogicOpParserState(LogicOpParserState):
    logic_op: str = "||"
    operator_precedence: OperatorPrecedence = LOGIC_OPS_PRECEDENCE["||"]


@fnr_dataclass
class NullishCoalescingLogicOpParserState(LogicOpParserState):
    logic_op: str = "??"
    operator_precedence: OperatorPrecedence = LOGIC_OPS_PRECEDENCE["??"]


@fnr_dataclass
class FixedOpParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TerminalParserState,  # replaced in post_init
            ExpressionParserState,
        ),
        repr=False,
    )
    lhs: DerivableTypeMixin = None
    op: str = None
    described_type: PType = None
    rhs_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        if any(
            self.op == s
            and len(self.parsed_states) == 1
            and self.active is None
            and char == s
            for s in ("+", "-", "/")
        ):
            # disallow two immediately following "/" -> would be parsed as comment, same for "+" and "-" --> would be decrement/increment
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            return (
                TerminalParserState(target_value=f" {self.op}"),
                self,
            )
        if pos == 1:
            excluded = []
            if self.op in ("==", "==="):
                if isinstance(self.lhs.described_type, ArrayPType):
                    excluded = [ArrayExpressionParserState]
                elif isinstance(self.lhs.described_type, TuplePType):
                    excluded = [TupleExpressionParserState]
            return (
                make_expr_class(
                    excluded,
                    typ=self.rhs_type,
                    identifiers=self.identifiers,
                    operator_precedence=(
                        OPERATOR_PRECEDENCES[self.op],
                        OPERATOR_ASSOCIATIVITY[self.op],
                    ),
                    max_steps=self.max_steps,
                )(),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass
class ArithmeticParserState(IncrementalParsingState, DerivableTypeMixin):
    lhs: DerivableTypeMixin = None
    # Override this in subclass
    op: str = field(init=False)
    operator_precedence: OperatorPrecedence = field(init=False)

    def parse_char(self, char: str) -> List[Self]:
        if self.lhs is None:
            return []
        # TODO fix this mess, quite inefficient
        next_states = [
            FixedOpParserState(
                lhs=self.lhs,
                identifiers=self.identifiers,
                op=op,
                operator_precedence=ARITHMETIC_OP_PRECEDENCE[op],
                described_type=res_typ,
                rhs_type=rhs_typ,
                max_steps=self.max_steps,
            )
            for lhs_typ, op_map in ALLOWED_ARITH_OPS.items()
            if lhs_typ >= self.lhs.described_type
            for rhs_typ, ops in op_map.items()
            for (op, res_typ) in ops
            if op == self.op
        ]
        return sum([ns.parse_char(char) for ns in next_states], [])

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        for lhs_typ, op_map in ALLOWED_ARITH_OPS.items():
            if lhs_typ >= self.lhs.described_type:
                for rhs_typ, ops in op_map.items():
                    for op, res_typ in ops:
                        if op == self.op and reachable(
                            res_typ,
                            goal,
                            min_operator_precedence,
                            (
                                ARITHMETIC_OP_PRECEDENCE[op],
                                OPERATOR_ASSOCIATIVITY[op],
                            ),
                            as_array,
                            as_nested_expression,
                            as_pattern,
                            max_steps=self.max_steps,
                        ):
                            return True
        return False

    def num_active_states(self):
        return 1


def op_parser_class_populator(op, prec):
    def populate(obj):
        obj["op"] = op
        obj["operator_precedence"] = prec

    return populate


ArithmeticParserStates = [
    fnr_dataclass(
        types.new_class(
            f"{op}ArithmeticParserState",
            (ArithmeticParserState,),
            {},
            op_parser_class_populator(op, prec),
        )
    )
    for op, prec in ARITHMETIC_OP_PRECEDENCE.items()
]


@fnr_dataclass
class BoolOpParserState(IncrementalParsingState, DerivableTypeMixin):
    lhs: DerivableTypeMixin = None
    # override this in subclass
    op: str = field(init=False)
    operator_precedence: OperatorPrecedence = field(init=False)

    def parse_char(self, char: str) -> List[Self]:
        if self.lhs is None:
            return []
        lhs_typ = self.lhs.described_type
        next_states = []
        op = self.op
        if (isinstance(lhs_typ, NumberPType) or isinstance(lhs_typ, BigIntPType)) and (
            "<" in op or ">" in op
        ):
            rhs_typ = UnionPType([NumberPType(), BigIntPType()])
        elif "<" in op or ">" in op:
            # exception: undefined/null never work here
            if lhs_typ in FALSEY_TYPES or (
                isinstance(lhs_typ, UnionPType)
                and set(FALSEY_TYPES).intersection(lhs_typ.types)
            ):
                return []
            # the rhs type has to match, because TS can detect if there is no overlap between the types and will reject it
            # TODO does setting equality ensure here that any allowed overlap is fine?
            rhs_typ = lhs_typ
        else:
            # in theory the rhs type has to have some overlap, because TS can detect if there is no overlap between the types and will reject it
            # using this type imlpements a reachability analysis for "some type that has overlap with X"
            rhs_typ = OverlapsWith(merge_typs(lhs_typ, NullPType(), UndefinedPType()))
            # rhs_typ = UnionPType([lhs_typ, NullPType(), UndefinedPType()])
        next_states.append(
            FixedOpParserState(
                lhs=self.lhs,
                identifiers=self.identifiers,
                op=op,
                operator_precedence=BOOL_OPS_PRECEDENCE[op],
                described_type=BooleanPType(),
                rhs_type=rhs_typ,
                max_steps=self.max_steps,
            )
        )
        return sum([ns.parse_char(char) for ns in next_states], [])

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            BooleanPType(),
            goal,
            min_operator_precedence,
            BOOL_OPS_PRECEDENCE[self.op],
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )

    def num_active_states(self):
        return 1


BoolOpParserStates = [
    fnr_dataclass(
        types.new_class(
            f"{op}BoolOpParserState",
            (BoolOpParserState,),
            {},
            op_parser_class_populator(op, prec),
        )
    )
    for op, prec in BOOL_OPS_PRECEDENCE.items()
]

# assumption is that signature is "for elem in list: first in tuple match/ge -> return typ is second in tuple"
ALLOWED_UNOPS = {
    " +": [(BigIntPType(), None), (AnyPType(), NumberPType())],
    " -": [(NumberPType(), NumberPType()), (BigIntPType(), BigIntPType())],
    " ~": [(NumberPType(), NumberPType()), (BigIntPType(), BigIntPType())],
    " !": [(AnyPType(), BooleanPType())],
    " typeof": [(AnyPType(), StringPType())],
}


@functools.lru_cache()
def allowed_operand_types_of_unop(operator: str):
    return {arg for arg, ret in ALLOWED_UNOPS[operator] if ret is not None}


@functools.lru_cache()
def return_typ_of_unop(operator: str, operand_typ: PType):
    """
    Return None if not a valid operation
    """
    for optyp, rettyp in ALLOWED_UNOPS[operator]:
        if optyp >= operand_typ:
            return rettyp
    return None


@functools.lru_cache()
def return_typs_of_unop(operator: str):
    return {rettyp for _, rettyp in ALLOWED_UNOPS[operator] if rettyp is not None}


@fnr_dataclass
class UnopExpressionParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TerminalParserState,  # replaced in post_init
            ExpressionParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (14, "right")

    @property
    def described_type(self):
        if len(self.parsed_states) < 2:
            return None
        return return_typ_of_unop(
            self.parsed_states[0].target_value, self.parsed_states[1].typ
        )

    def parse_char(self, char: str) -> List[Self]:
        if len(self.parsed_states) == 1 and self.active is None:
            # intercept just before we parse the next character after the operator
            parsed_op: TerminalParserState = self.parsed_states[0]
            if parsed_op.target_value == " +" and char == "+":
                return []
            elif parsed_op.target_value == " -" and char == "-":
                return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            allowed_ops = {
                op
                for op, op_typ_list in ALLOWED_UNOPS.items()
                for _, rettyp in op_typ_list
                if rettyp is not None and (self.typ is None or self.typ >= rettyp)
            }
            return (
                UnionParserState(
                    parse_classes=[
                        partial(TerminalParserState, target_value=op)
                        for op in allowed_ops
                    ],
                    identifiers=self.identifiers,
                ),
                self,
            )
        if pos == 1:
            parsed_op: TerminalParserState = self.parsed_states[0]
            return (
                UnionParserState(
                    parse_classes=[
                        partial(
                            ExpressionParserState,
                            typ=t,
                            operator_precedence=self.operator_precedence,
                            max_steps=self.max_steps,
                        )
                        for t in allowed_operand_types_of_unop(parsed_op.target_value)
                    ],
                    identifiers=self.identifiers,
                    pass_type=False,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        # TODO need to adapt to described type of current operand
        if len(self.parsed_states) >= 1:
            if not self.active and len(self.parsed_states) < 2:
                possible_rettyps = return_typs_of_unop(
                    self.parsed_states[0].target_value
                )
            else:
                active = self.active
                if active is None:
                    active = self.parsed_states[1]
                possible_rettyps = {
                    return_typ_of_unop(self.parsed_states[0].target_value, active.typ)
                }
            return any_reachable(
                possible_rettyps,
                goal,
                min_operator_precedence,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                max_steps=self.max_steps,
            )
        active = self.active
        if active is None:
            active = self.init_class_at_pos_hook(len(self.parsed_states))[0]
        if isinstance(active, TerminalParserState):
            return any_reachable(
                return_typs_of_unop(active.target_value),
                goal,
                min_operator_precedence,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                max_steps=self.max_steps,
            )
        # else we know that it is a union type of terminal states
        return any(
            any_reachable(
                return_typs_of_unop(act.target_value),
                goal,
                min_operator_precedence,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                max_steps=self.max_steps,
            )
            for act in active.init_classes()
        )


ALLOWED_PREFIX_OPS = {
    " ++": {NumberPType()},
    " --": {NumberPType()},
}


@fnr_dataclass
class AssigningPrefixParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TerminalParserState,  # replaced in post_init
            TypedAssignmentTargetParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (14, "right")

    @property
    def described_type(self):
        if len(self.parsed_states) < 2:
            return None
        return self.parsed_states[1].described_type

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            return (
                UnionParserState(
                    parse_classes=[
                        partial(TerminalParserState, target_value=op)
                        for op, typs in ALLOWED_PREFIX_OPS.items()
                        if self.typ is None or any(self.typ >= t for t in typs)
                    ],
                ),
                self,
            )
        if pos == 1:
            # since all ops operate on numbers we don't need to distinguish here
            return (
                TypedAssignmentTargetParserState(
                    typ=NumberPType(),
                    identifiers=self.identifiers,
                    max_steps=self.max_steps,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if self.__class__ in recursing_path:
            # if we already tried this, no need to try again
            return False
        # TODO part of the assumption that signature is (T, T) -> T
        if len(self.parsed_states) == 2:
            return self.parsed_states[1].derivable(
                goal,
                min_operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
        active = self.active
        if active is None:
            active = self.init_class_at_pos_hook(len(self.parsed_states))[0]
            recursing_path = recursing_path + [self.__class__]
        if len(self.parsed_states) == 1:
            return active.derivable(
                goal,
                min_operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
        if isinstance(active, TerminalParserState):
            return any_reachable(
                ALLOWED_PREFIX_OPS[active.target_value],
                goal,
                min_operator_precedence,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                max_steps=self.max_steps,
            )
        # else we know that it is a union type of terminal states
        return any(
            any_reachable(
                ALLOWED_PREFIX_OPS[act.target_value],
                goal,
                min_operator_precedence,
                self.operator_precedence,
                as_array,
                as_nested_expression,
                as_pattern,
                max_steps=self.max_steps,
            )
            for act in active.init_classes()
        )


ALLOWED_POSTFIX_OPS = {
    NumberPType(): {" ++", " --"},
}


@fnr_dataclass
class AssigningPostfixParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (TerminalParserState,),  # replaced in post_init
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (15, "left")
    lhs: DerivableTypeMixin = None

    @property
    def described_type(self):
        return NumberPType()

    def parse_char(self, char: str) -> List[Self]:
        # check that lhs is actually a valid assignment target
        if self.lhs is None:
            return []
        if self.lhs.described_type not in ALLOWED_POSTFIX_OPS:
            return []
        if not is_assignable(self.lhs):
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 0:
            return (
                UnionParserState(
                    parse_classes=[
                        partial(TerminalParserState, target_value=op)
                        for op in ALLOWED_POSTFIX_OPS.get(self.described_type, [])
                    ],
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass
class GroupedExpressionParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ("),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (18, "right")

    def parse_char(self, char: str) -> List[Self]:
        if self.max_steps <= 0:
            return []
        return super().parse_char(char)

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            return (
                ExpressionParserState(
                    identifiers=self.identifiers,
                    typ=self.typ,
                    max_steps=self.max_steps - 1,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    @property
    def described_type(self):
        if len(self.parsed_states) < 3:
            return None
        return self.parsed_states[1].described_type

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        if self.__class__ in recursing_path:
            # if we already tried this, no need to try again
            return False
        # TODO all types are derivable until the expression starts
        if len(self.parsed_states) <= 1:
            active = self.active
            if active is None or len(self.parsed_states) == 0:
                active = self.init_class_at_pos_hook(1)[0]
                # we only need to avoid recursion when initializing the step
                recursing_path = recursing_path + [self.__class__]
                # grouping resets the min precedence
            return active.derivable(
                goal,
                MIN_OPERATOR_PRECEDENCE,
                as_array,
                [min_operator_precedence] + as_nested_expression,
                as_pattern,
                recursing_path,
            )
        # outside of the group the normal rules apply
        return self.parsed_states[1].derivable(
            goal,
            min_operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            recursing_path,
        )


@fnr_dataclass
class ArrayMoreExpressionsParserState(PlusParserState):
    parse_class: Type[IncrementalParsingState] = field(
        default_factory=lambda: partial(
            ConcatParserState,
            parse_classes=(
                partial(TerminalParserState, target_value=" ,"),
                partial(
                    UnionParserState,
                    parse_classes=[ExpressionParserState, SpreadOpParserState],
                    pass_type=True,
                ),
            ),
            pass_type=True,
        )
    )

    def get_expr_types(self):
        types = []
        for s in self.accepted:
            types.append(s.parsed_states[1].described_type)
        return types


@fnr_dataclass
class ArrayMoreExpressionsEndParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ArrayMoreExpressionsParserState,
            ListEndParserState,
        ),
        repr=False,
    )
    pass_type: bool = True

    def get_expr_types(self):
        if len(self.parsed_states) >= 1:
            s = self.parsed_states[0]
        else:
            if self.active is not None:
                return []
            else:
                s = self.active

        return s.get_expr_types()


@fnr_dataclass
class ArrayEndOrMoreExpressionsParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            ArrayMoreExpressionsEndParserState,
            ListEndParserState,
        ),
        repr=False,
    )
    pass_type: bool = True


@fnr_dataclass
class EmptyArrayExpressionParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (partial(TerminalParserState, target_value=" [ ]"),),
        repr=False,
    )

    @property
    def described_type(self):
        if self.typ is not None and isinstance(self.typ, ArrayPType):
            return self.typ
        # return ArrayPType(NeverPType())
        return ArrayPType(GenericPType())

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        # otherwise we can reach anything reached by a generic array
        return reachable(
            ArrayPType(GenericPType()),
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass
class NonEmptyArrayExpressionParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ["),
            partial(
                UnionParserState,
                parse_classes=[ExpressionParserState, SpreadOpParserState],
                pass_type=True,
            ),
            ArrayEndOrMoreExpressionsParserState,
        ),
        repr=False,
    )

    def parse_char(self, char: str) -> List[Self]:
        res = []
        if self.active is None:
            if (
                len(self.parsed_states) in (1, 2)
                and not self.pass_type
                and isinstance(self.typ, ArrayPType)
            ):
                obj = replace(self, pass_type=True, typ=self.typ.element_type)
                res += super(NonEmptyArrayExpressionParserState, obj).parse_char(char)
        res += super().parse_char(char)
        return res

    @property
    def described_type(self):
        types = []

        if len(self.parsed_states) <= 1:
            return None
        else:
            s = self.parsed_states[1]
            types.append(s.described_type)

            if len(self.parsed_states) == 3:
                s = self.parsed_states[2]
            else:
                s = self.active
            if isinstance(s, ArrayMoreExpressionsEndParserState):
                types += s.get_expr_types()

        if len(types) == 0:
            if self.typ is not None:
                return self.typ
            else:
                # TODO check if this is correct
                return ArrayPType(GenericPType())
        elif len(types) >= 1:
            types = list(set(types))
            if len(types) == 1:
                return ArrayPType(types[0])
            # TODO handle subtyping
            else:
                return ArrayPType(UnionPType(types))

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        # TODO avoid recursion if we are already deeper nested than goal to array
        goal_nesting_depth = goal.nesting_depth
        if len(self.parsed_states) <= 1:
            if (
                goal_nesting_depth[0] < len(as_array)
                and self.__class__ in recursing_path
            ):
                # this is unsound if we can not guarantee that we were called with a level of recursion less
                return False
            active = self.active
            if len(self.parsed_states) == 0 or active is None:
                active = self.init_class_at_pos_hook(1)[0]
                # we only need to avoid recursion when initializing the step
                recursing_path = recursing_path + [self.__class__]
            # first determine all types reachable from the expression
            # this resets the min precedence
            return active.derivable(
                goal,
                MIN_OPERATOR_PRECEDENCE,
                [min_operator_precedence] + as_array,
                as_nested_expression,
                as_pattern,
                recursing_path,
            )
        else:
            return reachable(
                self.described_type,
                goal,
                min_operator_precedence,
                MAX_OPERATOR_PRECEDENCE,
                as_array,
                as_nested_expression,
                as_pattern,
                max_steps=self.max_steps,
            )


@fnr_dataclass
class ArrayExpressionParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            EmptyArrayExpressionParserState,
            NonEmptyArrayExpressionParserState,
        ),
        repr=False,
    )
    pass_type: bool = True


@fnr_dataclass
class EmptyTupleExpressionParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (partial(TerminalParserState, target_value=" [ ]"),),
        repr=False,
    )

    @property
    def described_type(self):
        return TuplePType(types=[])

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            TuplePType(types=[]),
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


@fnr_dataclass
class TupleMoreExpressionsParserState(PlusParserState):
    parse_class: Type[IncrementalParsingState] = field(
        default_factory=lambda: partial(
            ConcatParserState,
            parse_classes=(
                partial(TerminalParserState, target_value=" ,"),
                ExpressionParserState,
            ),
            pass_type=True,
        )
    )
    pass_type: bool = True

    def get_types(self):
        types = []
        for s in self.accepted:
            if len(s.parsed_states) < 2:
                types.append(self.typ)
            else:
                types.append(s.parsed_states[1].described_type)
        if self.active is not None:
            types.append(self.active.typ)
        return types


@fnr_dataclass
class TupleMoreExpressionsEndParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TupleMoreExpressionsParserState,
            ListEndParserState,
        ),
        repr=False,
    )
    pass_type: bool = True

    def get_types(self):
        if len(self.parsed_states) == 0:
            if self.active is None:
                return []
            else:
                return self.active.get_types()
        elif len(self.parsed_states) >= 1:
            return self.parsed_states[0].get_types()


@fnr_dataclass
class TupleEndOrMoreExpressionsParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TupleMoreExpressionsEndParserState,
            ListEndParserState,
        ),
        repr=False,
    )
    pass_type: bool = True


@fnr_dataclass
class NonEmptyTupleExpressionParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ["),
            ExpressionParserState,
            TupleEndOrMoreExpressionsParserState,
        ),
        repr=False,
    )

    @property
    def described_type(self):
        if len(self.parsed_states) == 0:
            return AbsTuplePType(types=[])
        elif len(self.parsed_states) == 1:
            if self.active is None:
                return AbsTuplePType(types=[])
            else:
                return AbsTuplePType(types=[self.active.typ])
        elif len(self.parsed_states) == 2:
            first_type = self.parsed_states[1].described_type
            if self.active is None:
                return AbsTuplePType(types=[first_type])
            else:
                if isinstance(self.active, TerminalParserState):
                    later_types = []
                else:
                    later_types = self.active.get_types()
                return AbsTuplePType(types=[first_type] + later_types)
        else:
            first_type = self.parsed_states[1].described_type
            if isinstance(self.parsed_states[2], TerminalParserState):
                later_types = []
            else:
                later_types = self.parsed_states[2].get_types()
            return TuplePType(types=[first_type] + later_types)

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        res = reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )
        return res


@fnr_dataclass
class TupleExpressionParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            EmptyTupleExpressionParserState,
            NonEmptyTupleExpressionParserState,
        ),
        repr=False,
    )
    pass_type: bool = True


def extract_expressions_from_tuple_parser(
    parser: NonEmptyTupleExpressionParserState,
) -> List[IncrementalParsingState]:
    """
    Returns all the expression parsers inside a parsed tuple expression
    """
    first_expr = parser.parsed_states[1].active
    all_expr = list(first_expr)
    further_exprs_end = parser.parsed_states[2]
    if isinstance(further_exprs_end, TupleMoreExpressionsEndParserState):
        further_exprs: TupleMoreExpressionsParserState = (
            further_exprs_end.parsed_states[0]
        )
        for concat_parser in further_exprs.accepted:
            further_expr = concat_parser.parsed_states[1].active
            all_expr.extend(further_expr)
    return all_expr


def is_assignable(state: IncrementalParsingState):
    return (
        isinstance(state, ComputedMemberAccessParserState)
        or (isinstance(state, MemberAccessStateParser) and state.is_mutable)
        or (isinstance(state, ExistingIdentifierParserState) and state.is_mutable)
        or isinstance(state, TupleComputedMemberAccessParserState)
        or (
            isinstance(state, NonEmptyTupleExpressionParserState)
            and all(
                is_assignable(x) for x in extract_expressions_from_tuple_parser(state)
            )
        )
    )


@fnr_dataclass()
class ForLoopParserState(ConcatParserState):
    """
    For statement
    Parsers:
    0: for(
    1: stmt
    2: expression
    3: ;
    3: expression
    2: ){
    3: stmts
    4: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" for ("),
            StmtParserState,
            ExpressionParserState,
            partial(TerminalParserState, target_value=" ;"),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
            StmtParserState,
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos >= 2:
            new_identifiers = union_dict(
                self.identifiers,
                self.parsed_states[1].identifiers,
            )
            if pos == 2 or pos == 4:
                return (
                    ExpressionParserState(
                        # typ=BooleanPType(), TODO
                        identifiers=new_identifiers,
                    ),
                    self,
                )
            if pos == 6:
                return (
                    StmtParserState(
                        identifiers=new_identifiers,
                        return_type=self.return_type,
                        has_to_return=False,
                        in_loop=True,
                    ),
                    self,
                )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                # get id updates from the stmt and propagate it
                new_ids = r.parsed_states[-1].identifiers
                # We can not be sure whether the if statement will be executed
                # so we have to assume that the return statement is not executed
                returned_in_branches = False
                r = replace(
                    r,
                    returned_in_branches=returned_in_branches,
                    identifiers=update_keys(r.identifiers, new_ids),
                )
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class ForOfExistingIdentifierParserState(ConcatParserState):
    """
    Only for existing identifiers
    For statement
    Parsers:
    0: for(
    2: ref identifier
    3: of
    4: iterable expression
    5: ){
    6: stmts
    7: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" for ("),
            TypedAssignmentTargetParserState,
            partial(TerminalParserState, target_value=" of"),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
            StmtParserState,
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 3:
            variable_type = self.parsed_states[1].described_type
            return (
                ExpressionParserState(
                    typ=UnionPType(types=[ArrayPType(variable_type), StringPType()]),
                    identifiers=self.identifiers,
                ),
                self,
            )
        if pos == 5:
            new_identifiers = union_dict(
                self.identifiers,
                self.parsed_states[1].identifiers,
            )
            return (
                StmtParserState(
                    identifiers=new_identifiers,
                    return_type=self.return_type,
                    has_to_return=False,
                    in_loop=True,
                ),
                self,
            )
        if pos == 6:
            # propagate type re-assignments up
            new_ids = self.parsed_states[5].identifiers
            new_parser, new_self = super().init_class_at_pos_hook(pos)
            return (
                new_parser,
                replace(self, identifiers=update_keys(new_self.identifiers, new_ids)),
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                # it can happen that the loop is never executed, so
                # we can never be sure that this returns
                returned_in_branches = False
                r = replace(r, returned_in_branches=returned_in_branches)
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class ForOfDeclarationParserState(ConcatParserState):
    """
    Note only untyped declarations are allowed
    Typescript does not allow typed declarations in for of loops
    For statement
    Parsers:
    0: for(
    1: const/let
    2: identifier
    3: of
    4: iterable expression
    5: ){
    6: stmts
    7: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" for ("),
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" const\t"),
                    partial(TerminalParserState, target_value=" let\t"),
                ],
            ),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" of"),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
            StmtParserState,
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 4:
            return (
                UnionParserState(
                    parse_classes=[
                        partial(ExpressionParserState, typ=ArrayPType(GenericPType())),
                        partial(ExpressionParserState, typ=StringPType()),
                    ],
                    identifiers=self.identifiers,
                    pass_type=False,
                ),
                self,
            )
        if pos == 6:
            expr_type = self.parsed_states[4].described_type
            if isinstance(expr_type, ArrayPType) and not isinstance(
                expr_type.element_type, GenericPType
            ):
                element_type = expr_type.element_type
            elif isinstance(expr_type, StringPType):
                element_type = StringPType()
            else:
                raise ValueError("Invalid type for for of loop")
            mutable = self.parsed_states[1].target_value == " let\t"
            new_identifiers = union_dict(
                self.identifiers,
                {self.parsed_states[2].id_name: (element_type, mutable)},
            )
            return (
                StmtParserState(
                    identifiers=new_identifiers,
                    return_type=self.return_type,
                    has_to_return=False,
                    in_loop=True,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                defined_name = self.parsed_states[2].id_name
                # propagate type re-assignments up, but skip the locally defined name
                new_ids = r.parsed_states[6].identifiers
                updated = {
                    id: val
                    for id, val in update_keys(r.identifiers, new_ids).items()
                    if id != defined_name
                }
                # it can happen that the loop is never executed, so
                # we can never be sure that this returns
                returned_in_branches = False
                r = replace(
                    r, returned_in_branches=returned_in_branches, identifiers=updated
                )
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class ForOfTupleDeclarationParserState(ConcatParserState):
    """
    Note only untyped declarations are allowed
    Typescript does not allow typed declarations in for of loops
    For statement
    Parsers:
    0: for(
    1: const/let
    2: identifier
    3: of
    4: iterable expression
    5: ){
    6: stmts
    7: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" for ("),
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" const\t"),
                    partial(TerminalParserState, target_value=" let\t"),
                ],
            ),
            partial(
                ConcatParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" ["),
                    DefiningIdentifierParserState,
                    partial(TerminalParserState, target_value=" ,"),
                    DefiningIdentifierParserState,
                    partial(TerminalParserState, target_value=" ]"),
                ],
            ),
            partial(TerminalParserState, target_value=" of"),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
            StmtParserState,
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 4:
            return (
                ExpressionParserState(
                    # typ=ArrayPType(TuplePType([GenericPType(), GenericPType()])),
                    # TODO make this more precise
                    typ=ArrayPType(GenericPType()),
                    identifiers=self.identifiers,
                    pass_type=False,
                ),
                self,
            )
        if pos == 6:
            expr_type = self.parsed_states[4].described_type
            if (
                isinstance(expr_type, ArrayPType)
                and isinstance(expr_type.element_type, TuplePType)
                and len(expr_type.element_type.types) == 2
            ):
                element_type0 = expr_type.element_type.types[0]
                element_type1 = expr_type.element_type.types[1]
            elif isinstance(expr_type, ArrayPType) and isinstance(
                expr_type.element_type, ArrayPType
            ):
                element_type0 = expr_type.element_type.element_type
                element_type1 = expr_type.element_type.element_type
            else:
                raise ValueError("Invalid type for for of loop")
            mutable = self.parsed_states[1].target_value == " let\t"
            new_identifiers = union_dict(
                self.identifiers,
                {
                    self.parsed_states[2].parsed_states[1].id_name: (
                        element_type0,
                        mutable,
                    )
                },
                {
                    self.parsed_states[2].parsed_states[3].id_name: (
                        element_type1,
                        mutable,
                    )
                },
            )
            return (
                StmtParserState(
                    identifiers=new_identifiers,
                    return_type=self.return_type,
                    has_to_return=False,
                    in_loop=True,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                defined_names = (
                    self.parsed_states[2].parsed_states[1].id_name,
                    self.parsed_states[2].parsed_states[3].id_name,
                )
                # propagate type re-assignments up, but skip the locally defined name
                new_ids = r.parsed_states[6].identifiers
                updated = {
                    id: val
                    for id, val in update_keys(r.identifiers, new_ids).items()
                    if id not in defined_names
                }
                # it can happen that the loop is never executed, so
                # we can never be sure that this returns
                returned_in_branches = False
                r = replace(
                    r, returned_in_branches=returned_in_branches, identifiers=updated
                )
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class WhileLoopParserState(ConcatParserState):
    """
    For statement
    Parsers:
    0: while(
    1: expression
    2: ){
    3: stmts
    4: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" while ("),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" ) {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 3:
            return (
                StmtsParserState(
                    identifiers=self.identifiers,
                    return_type=self.return_type,
                    has_to_return=False,
                    in_loop=True,
                ),
                self,
            )
        if pos == 4:
            # propagate type re-assignments up
            new_ids = self.parsed_states[3].identifiers
            new_parser, new_self = super().init_class_at_pos_hook(pos)
            return (
                new_parser,
                replace(self, identifiers=update_keys(new_self.identifiers, new_ids)),
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                # it can happen that the loop is never executed, so
                # we can never be sure that this returns
                returned_in_branches = False
                r = replace(r, returned_in_branches=returned_in_branches)
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class DoWhileLoopParserState(ConcatParserState):
    """
    For statement
    Parsers:
    0: do {
    1: stmts
    2: } while (
    3: expression
    4: )
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" do {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" } while ("),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            return (
                StmtsParserState(
                    identifiers=self.identifiers,
                    return_type=self.return_type,
                    has_to_return=False,
                    in_loop=True,
                ),
                self,
            )
        if pos == 2:
            # propagate type re-assignments up
            new_ids = self.parsed_states[1].identifiers
            new_parser, new_self = super().init_class_at_pos_hook(pos)
            return (
                new_parser,
                replace(self, identifiers=update_keys(new_self.identifiers, new_ids)),
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                # it can happen that the loop is never executed, so
                # we can never be sure that this returns
                returned_in_branches = False
                r = replace(r, returned_in_branches=returned_in_branches)
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class TryCatchParserState(ConcatParserState):
    """
    try/catch statement
    Parsers:
    0: try {
    1: stmts
    2: } catch (
    3: error_name
    4: ) {
    5: stmts
    6: }
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" try {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" } catch ("),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" ) {"),
            StmtsParserState,
            partial(TerminalParserState, target_value=" }"),
        ),
        repr=False,
    )

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 5:
            error_name_parser: DefiningIdentifierParserState = self.parsed_states[3]
            return (
                StmtsParserState(
                    identifiers=union_dict(
                        self.identifiers,
                        {error_name_parser.id_name: (AnyPType(), False)},
                    ),
                    return_type=self.return_type,
                    has_to_return=False,
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for r in res:
            if r.accept:
                # it can happen that the loop is never executed, so
                # we can never be sure that this returns
                returned_in_branches = (
                    self.parsed_states[1].returned_in_branches and self.parsed_states[5]
                )
                r = replace(r, returned_in_branches=returned_in_branches)
            fixed_res.append(r)
        return fixed_res


@fnr_dataclass()
class ThrowErrorParser(ConcatParserState):
    """
    try/catch statement
    Parsers:
    0: throw new Error(
    1: string
    2: );
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" throw new Error ("),
            ExpressionParserState,
            partial(TerminalParserState, target_value=" )"),
            EOLParserState,
        ),
        repr=False,
    )

    def __post_init__(self):
        object.__setattr__(self, "returned_in_branches", True)
        # super().__post_init__()

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            return (
                ExpressionParserState(
                    identifiers=self.identifiers,
                    typ=StringPType(),
                ),
                self,
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass
class RequireCryptoParserState(ConcatParserState, DerivableTypeMixin):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" require ( 'crypto' )"),
                    partial(TerminalParserState, target_value=' require ( "crypto" )'),
                ],
            ),
        ),
        repr=False,
    )

    @property
    def described_type(self):
        return CryptoPType()

    def derivable(
        self,
        goal: PType,
        min_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE,
        as_array: List[OperatorPrecedence] = [],
        as_nested_expression: List[OperatorPrecedence] = [],
        as_pattern: List[Tuple[PType, PType, OperatorPrecedence]] = [],
        recursing_path: List[Type[IncrementalParsingState]] = [],
    ) -> Union[bool, Set[PType]]:
        return reachable(
            self.described_type,
            goal,
            min_operator_precedence,
            self.operator_precedence,
            as_array,
            as_nested_expression,
            as_pattern,
            max_steps=self.max_steps,
        )


NESTED_STMTS = (
    ForLoopParserState,
    ForOfExistingIdentifierParserState,
    ForOfDeclarationParserState,
    ForOfTupleDeclarationParserState,
    WhileLoopParserState,
    DoWhileLoopParserState,
    IfParserState,
    ITEParserState,
    StmtsBlockParserState,
)
NESTED_STMTS_POSS = tuple(
    tuple(
        i
        for i, x in enumerate(parser_class().parse_classes)
        if isinstance(x(), StmtsParserState) or isinstance(x(), StmtParserState)
    )
    for parser_class in NESTED_STMTS
)


def extract_active_return(parser: IncrementalParsingState):
    """
    Derives the currently active return stmt so that we can call "derivable" on it -> constrain it
    """
    if isinstance(parser, StmtsParserState):
        active = parser.active
        if active is None:
            return None
        return extract_active_return(active)
    if isinstance(parser, ReturnValueParserState):
        return parser
    if isinstance(parser, ReturnVoidParserState):
        return parser
    for parser_class in NESTED_STMTS:
        if isinstance(parser, parser_class):
            active = parser.active
            if active is None:
                return None
            return extract_active_return(active)
    return None


def extract_return_type_stmts(parser: IncrementalParsingState):
    return_typ = _extract_return_type_stmts(parser) or VoidPType()
    if not parser.returned_in_branches and parser.accept:
        # if the stmts parser did not return in every path, there is a path that implicitly returns void
        return_typ = merge_typs(VoidPType(), return_typ)
    return return_typ


def _extract_return_type_stmts(parser: IncrementalParsingState):
    """
    Derives the described return type(s) of a (partial) statement.
    """
    if isinstance(parser, StmtsParserState):
        return_typs = tuple(
            filter(None, (_extract_return_type_stmts(x) for x in parser.accepted))
        )
        return merge_typs(*return_typs)
    if isinstance(parser, ReturnValueParserState) and parser.accept:
        return parser.parsed_states[2].described_type
    if isinstance(parser, ReturnVoidParserState) and parser.accept:
        return VoidPType()
    for parser_class, stmt_poss in zip(NESTED_STMTS, NESTED_STMTS_POSS):
        if isinstance(parser, parser_class):
            return_typs = tuple(
                filter(
                    None,
                    (
                        _extract_return_type_stmts(parser.parsed_states[stmt_pos])
                        for stmt_pos in stmt_poss
                        if stmt_pos < len(parser.parsed_states)
                    ),
                )
            )
            return merge_typs(*(return_typs))
    return None


GLOBAL_OBJECTS = {
    "Array": (AbsArrayPType(), False),
    "BigInt": (
        FunctionPType(
            [UnionPType([StringPType(), NumberPType(), BooleanPType()])], BigIntPType()
        ),
        False,
    ),
    "Boolean": (FunctionPType([AnyPType()], BooleanPType()), False),
    "decodeURI": (FunctionPType([StringPType()], StringPType()), False),
    "decodeURIComponent": (FunctionPType([StringPType()], StringPType()), False),
    "encodeURI": (FunctionPType([StringPType()], StringPType()), False),
    "encodeURIComponent": (FunctionPType([StringPType()], StringPType()), False),
    "eval": (FunctionPType([StringPType()], BaseTsObject()), False),
    "Infinity": (NumberPType(), False),
    "isFinite": (FunctionPType([NumberPType()], BooleanPType()), False),
    "isNaN": (FunctionPType([NumberPType()], BooleanPType()), False),
    "JSON": (JSONPType(), False),
    "NaN": (NumberPType(), False),
    "Number": (AbsNumberPType(), False),
    "Map": (
        FunctionPType(
            [
                ArrayPType(
                    TuplePType([TypeParameterPType("S"), TypeParameterPType("T")])
                )
            ],
            MapPType(TypeParameterPType("S"), TypeParameterPType("T")),
            is_constructor=True,
            force_new=True,
            optional_args=1,
        ),
        False,
    ),
    "Math": (MathPType(), False),
    "Object": (ObjectPType(), False),
    "parseFloat": (FunctionPType([StringPType()], NumberPType()), False),
    "parseInt": (
        FunctionPType([StringPType(), NumberPType()], NumberPType(), 1),
        False,
    ),
    "RegExp": (
        FunctionPType(
            [StringPType(), StringPType()],
            RegExpPType(),
            1,
            is_constructor=True,
            force_new=False,
        ),
        False,
    ),
    "Set": (
        FunctionPType(
            [ArrayPType(TypeParameterPType("T"))],
            SetPType(TypeParameterPType("T")),
            is_constructor=True,
            force_new=True,
            optional_args=1,
        ),
        False,
    ),
    # TODO model static properties
    "String": (AbsStringPType(), False),
    # "Command": (FunctionPType([], CommandPType(), is_constructor=True), False),
    "undefined": (UndefinedPType(), False),
    "null": (NullPType(), False),
}

INITIAL_STATE = IncrementalParserState(
    (ProgramParserState(identifiers=GLOBAL_OBJECTS),)
)

INITIAL_STATE_STUDY = IncrementalParserState(
    (
        ProgramParserState(
            identifiers={
                "Command": (
                    FunctionPType([], CommandPType(), is_constructor=True),
                    False,
                ),
                **GLOBAL_OBJECTS,
            }
        ),
    )
)


def custom_end_initial_state(end: str, custom_identifiers: dict):
    initial_state_class = ProgramParserState
    if end is None:
        return IncrementalParserState(
            (
                initial_state_class(
                    identifiers=union_dict(GLOBAL_OBJECTS, custom_identifiers)
                ),
            )
        )
    return IncrementalParserState(
        (
            ConcatParserState(
                parse_classes=(
                    initial_state_class,
                    partial(TerminalParserState, target_value=end),
                ),
                identifiers=union_dict(GLOBAL_OBJECTS, custom_identifiers),
            ),
        )
    )


def make_expr_class(excluded, **kwargs):
    parse_classes = ExpressionParserState().parse_classes
    for e in excluded:
        parse_classes.remove(e)
    return partial(ExpressionParserState, parse_classes=parse_classes, **kwargs)


def parse_ts_program(chars, print_failure_point=False):
    return incremental_ts_parse(
        state=INITIAL_STATE,
        chars=chars,
        print_failure_point=print_failure_point,
        # states=INITIAL_STATE_STUDY, chars=chars, print_failure_point=print_failure_point
    )


def incremental_ts_parse(
    state: IncrementalParserState,
    chars: str,
    print_failure_point=False,
) -> IncrementalParserState:
    """
    Parse a string incrementally
    This method performs automatic semicolon insertion as specified in https://262.ecma-international.org/7.0/index.html#sec-rules-of-automatic-semicolon-insertion
    :param chars:
    :return: True if parsed and state updated, False if not parsed
    """
    updated_states = (state.active_states, state.parsed_code)
    for i, char in enumerate(chars):
        if char in ("\r", "\b"):
            return IncrementalParserState([], updated_states[1])
        # ASI rule 3
        if char == "\n":
            previous_token = updated_states[1].strip().split(" ")[-1]
            if previous_token in ("break", "continue", "return", "throw"):
                updated_states = (
                    sum_list([state.parse_char(";") for state in updated_states[0]]),
                    updated_states[1] + ";",
                )

        # perform actual transition
        previous_states = updated_states
        updated_states = (
            sum_list([state.parse_char(char) for state in updated_states[0]]),
            updated_states[1] + char,
        )
        # ASI rule 1
        if updated_states[0] == []:
            currently_parsed = previous_states[1]
            preceding_whitespace = currently_parsed[len(currently_parsed.rstrip()) :]
            previous_token = (
                currently_parsed.strip()[-1] if currently_parsed.strip() else None
            )
            if "\n" in preceding_whitespace or "}" == previous_token:
                updated_states = (
                    sum_list([state.parse_char(";") for state in previous_states[0]]),
                    previous_states[1] + ";",
                )
                updated_states = (
                    sum_list([state.parse_char(char) for state in updated_states[0]]),
                    updated_states[1] + char,
                )
        if updated_states[0] == []:
            if print_failure_point:
                print(chars[:i], end="!X!")
                print(colored(chars[i:], "red"))
            break
    return IncrementalParserState(*updated_states)
