import re
from dataclasses import field
from functools import partial
from typing import Dict, Type, List, Tuple, Self

from typesafe_llm.parser.parser_base import (
    ConcatParserState,
    IncrementalParsingState,
    TerminalParserState,
    incremental_parse,
    IncrementalParserState,
    UnionParserState,
)
from typesafe_llm.parser.parser_go import (
    DefiningIdentifierParserState,
    CallSignatureParserState,
    call_signature_from_accepted_call_signature_parser,
)
from typesafe_llm.parser.parser_go_types import (
    TypeParserState,
)
from typesafe_llm.parser.types_base import PType, AnyPType
from typesafe_llm.parser.types_go import (
    FunctionPType,
    VoidPType,
)
from typesafe_llm.parser.util import fnr_dataclass, union_dict

_defs = """
 func AppendBool(dst []byte, b bool) []byte
func AppendFloat(dst []byte, f float64, fmt byte, prec, bitSize int) []byte
func AppendInt(dst []byte, i int64, base int) []byte
func AppendQuote(dst []byte, s string) []byte
func AppendQuoteRune(dst []byte, r rune) []byte
func AppendQuoteRuneToASCII(dst []byte, r rune) []byte
func AppendQuoteRuneToGraphic(dst []byte, r rune) []byte
func AppendQuoteToASCII(dst []byte, s string) []byte
func AppendQuoteToGraphic(dst []byte, s string) []byte
func AppendUint(dst []byte, i uint64, base int) []byte
func Atoi(s string) (int, error)
func CanBackquote(s string) bool
func FormatBool(b bool) string
func FormatComplex(c complex128, fmt byte, prec, bitSize int) string
func FormatFloat(f float64, fmt byte, prec, bitSize int) string
func FormatInt(i int64, base int) string
func FormatUint(i uint64, base int) string
func IsGraphic(r rune) bool
func IsPrint(r rune) bool
func Itoa(i int) string
func ParseBool(str string) (bool, error)
func ParseComplex(s string, bitSize int) (complex128, error)
func ParseFloat(s string, bitSize int) (float64, error)
func ParseInt(s string, base int, bitSize int) (i int64, err error)
func ParseUint(s string, base int, bitSize int) (uint64, error)
func Quote(s string) string
func QuoteRune(r rune) string
func QuoteRuneToASCII(r rune) string
func QuoteRuneToGraphic(r rune) string
func QuoteToASCII(s string) string
func QuoteToGraphic(s string) string
func QuotedPrefix(s string) (string, error)
func Unquote(s string) (string, error)
func UnquoteChar(s string, quote byte) (value rune, multibyte bool, tail string, err error)
"""


@fnr_dataclass()
class FunctionDeclarationParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" func\t"),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" ("),
            CallSignatureParserState,
            TypeParserState,
        ),
        repr=False,
    )

    @property
    def described_type(self) -> FunctionPType:
        return FunctionPType(
            call_signature=call_signature_from_accepted_call_signature_parser(
                self.parsed_states[3]
            ),
            return_type=self.parsed_states[4].described_type,
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
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class VoidFunctionDeclarationParserState(ConcatParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            partial(TerminalParserState, target_value=" func\t"),
            DefiningIdentifierParserState,
            partial(TerminalParserState, target_value=" ("),
            CallSignatureParserState,
        ),
        repr=False,
    )

    @property
    def described_type(self) -> FunctionPType:
        return FunctionPType(
            call_signature=call_signature_from_accepted_call_signature_parser(
                self.parsed_states[3]
            ),
            return_type=VoidPType(),
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
        return super().init_class_at_pos_hook(pos)


@fnr_dataclass()
class SomeFunctionDeclarationParserState(UnionParserState):
    parse_classes: List[Type[IncrementalParsingState]] = field(
        default_factory=lambda: (
            FunctionDeclarationParserState,
            VoidFunctionDeclarationParserState,
        ),
        repr=False,
    )

    @property
    def described_type(self) -> FunctionPType:
        return self.parsed_states[0].described_type


def extract_defs(defs: str) -> Dict[str, PType]:
    res = {}
    func_pattern = re.compile(r"^\s*func\s+(\w+)\s*\(")
    # TODO this is still missing tuple return types

    for line in defs.splitlines():
        func_match = func_pattern.findall(line)

        if not func_match:
            continue
        func_name = func_match[0]
        parsed_fun = incremental_parse(
            IncrementalParserState.from_state(SomeFunctionDeclarationParserState()),
            line,
            print_failure_point=True,
        )
        res[func_name] = None
        if not parsed_fun.accept():
            continue
        for state in parsed_fun.active_states:
            if not state.accept:
                continue
            res[func_name] = state.described_type
            break

    return res


res = extract_defs(_defs)
formatted_res = {f'"{k}"': repr(v) for k, v in res.items()}
for k, v in formatted_res.items():
    print(k, ":", v, ",")
