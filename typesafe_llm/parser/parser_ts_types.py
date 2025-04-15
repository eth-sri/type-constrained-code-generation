import frozenlist

from .types_base import OperatorPrecedence
from .util import fnr_dataclass
from .parser_base import (
    IncrementalParsingState,
    UnionParserState,
    PlusParserState,
    TerminalParserState,
    ConcatParserState,
    GenericDefiningIdentifierParserState,
    IdentifierParserState,
    AnyPType,
)
from dataclasses import field, replace
from typing import List, Type, Self, Tuple, Union, Set
from functools import partial
from .types_ts import (
    PType,
    FunctionPType,
    NumberPType,
    StringPType,
    BooleanPType,
    VoidPType,
    ArrayPType,
    TuplePType,
    NullPType,
    UndefinedPType,
    MIN_OPERATOR_PRECEDENCE,
    RegExpPType,
    TypeParameterPType,
    MapPType,
    SetPType,
    IndexSignaturePType,
    BigIntPType,
    BaseTsObject,
    merge_typs,
)


@fnr_dataclass()
class AbstractTypeParserState(IncrementalParsingState):
    """
    Base class for type parsers
    """

    @property
    def described_type(self):
        raise NotImplementedError()


@fnr_dataclass()
class TypeParserState(AbstractTypeParserState):
    """
    starting with i.e. an existing identifier, we can reach an arbitrarily deeply nested array by appending [] to the identifier
    """

    pass_type: bool = True
    force_array: bool = False

    parse_classes: List[
        Union[Type[IncrementalParsingState], Type[AbstractTypeParserState]]
    ] = field(
        default_factory=lambda: [
            FunctionTypeParserState,
            PrimitiveTypeParserState,
            GroupedTypeParserState,
            TupleTypeParserState,
            SetTypeParserState,
            MapTypeParserState,
            IndexSignatureTypeParserState,
        ],
        repr=False,
    )
    extend_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            ArrayTypeParserState,
            UnionTypeParserState,
        ],
        repr=False,
    )
    active: List[IncrementalParsingState] = None
    operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE
    rhs_operator_precedence: OperatorPrecedence = MIN_OPERATOR_PRECEDENCE
    described_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        active_classes = self.active.copy() if self.active is not None else None
        if active_classes is None:
            active_classes = [
                x(
                    identifiers=self.identifiers,
                )
                for x in self.parse_classes
            ]
        new_active_classes = []
        # all of these are the same:
        # - check if can accept the char at all
        # - check if reachable types allow reaching desired type
        for a in active_classes:
            if not a.accept:
                continue
            # transition to arithmetic parsing
            # check
            # a) is the current state a number
            # b) can arithmetic operations result in the desired type
            for ext in self.extend_classes:
                if ext.operator_precedence < a.operator_precedence or (
                    ext.operator_precedence == a.operator_precedence
                    and ext.operator_precedence[1] == "left"
                ):
                    active_classes += [ext(lhs=a)]
        for ac in active_classes:
            new_states = ac.parse_char(char)
            new_active_classes.extend(new_states)

        final_states, final_active_classes = [], []
        for s in new_active_classes:
            if s.accept:
                final_s = replace(
                    self, active=[s], accept=True, described_type=s.described_type
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
                    described_type=None,
                )
            )

        return final_states

    def num_active_states(self):
        return sum([x.num_active_states() for x in self.active]) if self.active else 1


PRIMITIVE_TYPE_MAP = {
    "any": BaseTsObject(),
    "number": NumberPType(),
    "string": StringPType(),
    "boolean": BooleanPType(),
    "void": VoidPType(),
    "null": NullPType(),
    "undefined": UndefinedPType(),
    "RegExp": RegExpPType(),
    "bigint": BigIntPType(),
    # Only enable in custom settings
    # "Command": CommandPType(),
}
primitive_types = frozenlist.FrozenList(PRIMITIVE_TYPE_MAP.keys())
primitive_types.freeze()


@fnr_dataclass()
class PrimitiveTypeParserState(IdentifierParserState, AbstractTypeParserState):
    whitelist: List[str] = primitive_types
    described_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(s, described_type=PRIMITIVE_TYPE_MAP[s.id_name])
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class GroupedTypeParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ("),
            TypeParserState,
            partial(TerminalParserState, target_value=" )"),
        ),
        repr=False,
    )
    id_name: str = None
    described_type: PType = None
    operator_precedence: OperatorPrecedence = (18, "left")

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    described_type=s.parsed_states[1].described_type,
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class DefiningIdentifierParserState(GenericDefiningIdentifierParserState):
    forbidden_ids: Set[str] = frozenset(
        {"if", "function", "switch", "while", "var", "const", "let", "else"}
    )


@fnr_dataclass()
class CallParameterParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ,"),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" :"),
            TypeParserState,
        ),
        repr=False,
    )
    id_name: str = None
    described_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    id_name=s.parsed_states[1].id_name,
                    described_type=s.parsed_states[3].described_type,
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class InitialCallParameterParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" :"),
            TypeParserState,
        )
    )
    id_name: str = None
    described_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    id_name=s.parsed_states[0].id_name,
                    described_type=s.parsed_states[2].described_type,
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class CallParameterListParserState(PlusParserState):
    parse_class: Type[IncrementalParsingState] = CallParameterParserState


@fnr_dataclass()
class NonEmptyCallSignatureParserState(ConcatParserState):
    """
    Parses a call signature of the form "<varname>:<type>(,<varname>:<type>)+)"
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            InitialCallParameterParserState,
            CallParameterListParserState,
            partial(TerminalParserState, target_value=" )"),
        ),
        repr=False,
    )
    accepted: List[Self] = ()

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    accepted=(self.parsed_states[0],) + self.parsed_states[1].accepted,
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class EmptyCallSignatureParserState(TerminalParserState):
    """
    Parses a call signature of the form ")"
    """

    target_value: str = " )"
    accepted: List[Self] = ()


@fnr_dataclass()
class SingleCallSignatureParserState(ConcatParserState):
    """
    Parses a call signature of the form "<varname>:<type>)"
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            InitialCallParameterParserState,
            partial(TerminalParserState, target_value=" )"),
        ]
    )
    accepted: List[Self] = ()

    def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
        if pos == 1:
            return super().init_class_at_pos_hook(pos)[0], replace(
                self, accepted=[self.parsed_states[0]]
            )
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class CallSignatureParserState(UnionParserState):
    """
    Parses a call signature of the form "(<varname>:<type>(,<varname>:<type>)*)?)"
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: [
            EmptyCallSignatureParserState,
            NonEmptyCallSignatureParserState,
            SingleCallSignatureParserState,
        ]
    )


def call_signature_from_accepted_call_signature_parser(
    parser: CallSignatureParserState,
):
    return [x.described_type for x in parser.accepted]


def identifiers_from_accepted_call_signature_parser(parser: CallSignatureParserState):
    return {x.id_name: (x.described_type, True) for x in parser.accepted}


@fnr_dataclass()
class FunctionTypeParserState(ConcatParserState, AbstractTypeParserState):
    """
    Function type
    Parsers:
    0: (
    1: CallSignatureParserState
    2: ) =>
    3: PrimitiveTypeParserState
    """

    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ("),
            CallSignatureParserState,
            partial(TerminalParserState, target_value=" =>"),
            TypeParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (1, "left")
    described_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    described_type=FunctionPType(
                        call_signature=[
                            x.described_type for x in s.parsed_states[1].accepted
                        ],
                        return_type=s.parsed_states[3].described_type,
                    ),
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class ArrayTypeParserState(TerminalParserState, AbstractTypeParserState):
    """
    Array type
    Parsers:
    0: [ ]
    """

    target_value: str = " [ ]"
    lhs: IncrementalParsingState = None
    operator_precedence: OperatorPrecedence = (7, "left")

    @property
    def element_type(self) -> AbstractTypeParserState:
        return self.lhs.described_type

    @property
    def described_type(self):
        return ArrayPType(element_type=self.element_type)

    def parse_char(self, char: str) -> List[Self]:
        # can not bind to a weaker operator
        if self.lhs.operator_precedence < self.operator_precedence:
            return []
        res = super().parse_char(char)
        return res


@fnr_dataclass()
class UnionTypeParserState(ConcatParserState, AbstractTypeParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" |"),
            TypeParserState,
        ),
        repr=False,
    )
    lhs: IncrementalParsingState = None
    operator_precedence: OperatorPrecedence = (5, "left")

    @property
    def described_type(self):
        types = [self.lhs.described_type, AnyPType()]
        if len(self.parsed_states) == 1 and self.active or len(self.parsed_states) == 2:
            active = self.active or self.parsed_states[-1]
            types[1] = active.described_type
        return merge_typs(*types)


@fnr_dataclass()
class TupleTypeParserState(ConcatParserState, AbstractTypeParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ["),
            TupleSignatureParserState,
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (16, "left")
    described_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    described_type=TuplePType(
                        types=[x.described_type for x in s.parsed_states[1].accepted]
                    ),
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class TupleSignatureParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            EmptyTupleTypeParserState,
            SingleTupleTypeParserState,
            MultiTupleTypeParserState,
        ),
    )


@fnr_dataclass()
class EmptyTupleTypeParserState(TerminalParserState):
    target_value: str = " ]"
    accepted: List[Self] = ()


@fnr_dataclass()
class SingleTupleTypeParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TypeParserState,
            partial(TerminalParserState, target_value=" ]"),
        ),
    )
    accepted: List[Self] = ()

    # def init_class_at_pos_hook(self, pos: int) -> Tuple[IncrementalParsingState, Self]:
    #     if pos == 1:
    #         return super().init_class_at_pos_hook(pos)[0], replace(
    #             self, accepted=[self.parsed_states[0]]
    #         )
    #     return super().init_class_at_pos_hook(pos)

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    accepted=[
                        self.parsed_states[0],
                    ],
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class MultiTupleTypeParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            TypeParserState,
            TupleParameterListParserState,
            partial(TerminalParserState, target_value=" ]"),
        ),
        repr=False,
    )
    accepted: List[Self] = ()

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    accepted=(self.parsed_states[0],) + self.parsed_states[1].accepted,
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class TupleParameterParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" ,"),
            TypeParserState,
        ),
        repr=False,
    )
    described_type: PType = None

    def parse_char(self, char: str) -> List[Self]:
        res = super().parse_char(char)
        fixed_res = []
        for s in res:
            if s.accept:
                s = replace(
                    s,
                    described_type=s.parsed_states[1].described_type,
                )
            fixed_res.append(s)
        return fixed_res


@fnr_dataclass()
class TupleParameterListParserState(PlusParserState):
    parse_class: Type[IncrementalParsingState] = TupleParameterParserState


@fnr_dataclass()
class MapTypeParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" Map<"),
            TypeParserState,
            partial(TerminalParserState, target_value=" ,"),
            TypeParserState,
            partial(TerminalParserState, target_value=">"),
        ),
        repr=False,
    )
    id_name: str = None
    operator_precedence: OperatorPrecedence = (18, "left")

    @property
    def described_type(self):
        types = [TypeParameterPType("S"), TypeParameterPType("T")]
        if len(self.parsed_states) == 1 and self.active or len(self.parsed_states) >= 2:
            active = (
                self.active if len(self.parsed_states) == 1 else self.parsed_states[1]
            )
            types[0] = active.described_type
        if len(self.parsed_states) == 3 and self.active or len(self.parsed_states) >= 4:
            active = (
                self.active if len(self.parsed_states) == 3 else self.parsed_states[3]
            )
            types[1] = active.described_type
        return MapPType(*types)


@fnr_dataclass()
class SetTypeParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" Set<"),
            TypeParserState,
            partial(TerminalParserState, target_value=">"),
        ),
        repr=False,
    )
    id_name: str = None
    operator_precedence: OperatorPrecedence = (18, "left")

    @property
    def described_type(self):
        types = [TypeParameterPType("S")]
        if len(self.parsed_states) == 1 and self.active or len(self.parsed_states) >= 2:
            active = (
                self.active if len(self.parsed_states) == 1 else self.parsed_states[1]
            )
            types[0] = active.described_type
        return SetPType(*types)


@fnr_dataclass()
class NumberOrStringPrimitiveTypeParserState(PrimitiveTypeParserState):
    whitelist: List[str] = (
        "number",
        "string",
    )


@fnr_dataclass()
class IndexSignatureKeyTypeParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            # number, string, number | string
            NumberOrStringPrimitiveTypeParserState,
        )
    )


@fnr_dataclass()
class IndexSignatureTypeParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" { ["),
            IdentifierParserState,
            partial(TerminalParserState, target_value=" :"),
            # TODO restrict to only string, number, type param or union thereof
            IndexSignatureKeyTypeParserState,
            partial(TerminalParserState, target_value=" ] :"),
            TypeParserState,
            partial(
                UnionParserState,
                parse_classes=[
                    partial(TerminalParserState, target_value=" ; }"),
                    partial(TerminalParserState, target_value=" }"),
                ],
            ),
        ),
        repr=False,
    )
    operator_precedence: OperatorPrecedence = (18, "left")

    @property
    def described_type(self):
        types = [TypeParameterPType("S"), TypeParameterPType("T")]
        if len(self.parsed_states) == 3 and self.active or len(self.parsed_states) >= 4:
            active = (
                self.active if len(self.parsed_states) == 3 else self.parsed_states[3]
            )
            types[0] = active.described_type
        if len(self.parsed_states) == 5 and self.active or len(self.parsed_states) >= 6:
            active = (
                self.active if len(self.parsed_states) == 5 else self.parsed_states[5]
            )
            types[1] = active.described_type
        return IndexSignaturePType(*types)
