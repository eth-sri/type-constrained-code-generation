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
)
from typesafe_llm.parser.util import fnr_dataclass, union_dict

_defs = """
func Clone(s string) string
func Compare(a, b string) int
func Contains(s, substr string) bool
func ContainsAny(s, chars string) bool
func ContainsFunc(s string, f func(rune) bool) bool
func ContainsRune(s string, r rune) bool
func Count(s, substr string) int
func Cut(s, sep string) (before, after string, found bool)
func CutPrefix(s, prefix string) (after string, found bool)
func CutSuffix(s, suffix string) (before string, found bool)
func EqualFold(s, t string) bool
func Fields(s string) []string
func FieldsFunc(s string, f func(rune) bool) []string
func HasPrefix(s, prefix string) bool
func HasSuffix(s, suffix string) bool
func Index(s, substr string) int
func IndexAny(s, chars string) int
func IndexByte(s string, c byte) int
func IndexFunc(s string, f func(rune) bool) int
func IndexRune(s string, r rune) int
func Join(elems []string, sep string) string
func LastIndex(s, substr string) int
func LastIndexAny(s, chars string) int
func LastIndexByte(s string, c byte) int
func LastIndexFunc(s string, f func(rune) bool) int
func Map(mapping func(rune) rune, s string) string
func Repeat(s string, count int) string
func Replace(s, old, new string, n int) string
func ReplaceAll(s, old, new string) string
func Split(s, sep string) []string
func SplitAfter(s, sep string) []string
func SplitAfterN(s, sep string, n int) []string
func SplitN(s, sep string, n int) []string
func Title(s string) stringdeprecated
func ToLower(s string) string
func ToLowerSpecial(c unicode.SpecialCase, s string) string
func ToTitle(s string) string
func ToTitleSpecial(c unicode.SpecialCase, s string) string
func ToUpper(s string) string
func ToUpperSpecial(c unicode.SpecialCase, s string) string
func ToValidUTF8(s, replacement string) string
func Trim(s, cutset string) string
func TrimFunc(s string, f func(rune) bool) string
func TrimLeft(s, cutset string) string
func TrimLeftFunc(s string, f func(rune) bool) string
func TrimPrefix(s, prefix string) string
func TrimRight(s, cutset string) string
func TrimRightFunc(s string, f func(rune) bool) string
func TrimSpace(s string) string
func TrimSuffix(s, suffix string) string
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
            IncrementalParserState.from_state(FunctionDeclarationParserState()),
            line,
            print_failure_point=True,
        )
        if not parsed_fun.active_states:
            res[func_name] = None
        else:
            res[func_name] = parsed_fun.active_states[0].described_type

    return res


res = extract_defs(_defs)
formatted_res = {f'"{k}"': repr(v) for k, v in res.items()}
for k, v in formatted_res.items():
    print(k, ":", v, ",")
