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
 func Append(b []byte, a ...any) []byte
func Appendf(b []byte, format string, a ...any) []byte
func Appendln(b []byte, a ...any) []byte
func Errorf(format string, a ...any) error
func FormatString(state State, verb rune) string
func Fprint(w io.Writer, a ...any) (n int, err error)
func Fprintf(w io.Writer, format string, a ...any) (n int, err error)
func Fprintln(w io.Writer, a ...any) (n int, err error)
func Fscan(r io.Reader, a ...any) (n int, err error)
func Fscanf(r io.Reader, format string, a ...any) (n int, err error)
func Fscanln(r io.Reader, a ...any) (n int, err error)
func Print(a ...any) (n int, err error)
func Printf(format string, a ...any) (n int, err error)
func Println(a ...any) (n int, err error)
func Scan(a ...any) (n int, err error)
func Scanf(format string, a ...any) (n int, err error)
func Scanln(a ...any) (n int, err error)
func Sprint(a ...any) string
func Sprintf(format string, a ...any) string
func Sprintln(a ...any) string
func Sscan(str string, a ...any) (n int, err error)
func Sscanf(str string, format string, a ...any) (n int, err error)
func Sscanln(str string, a ...any) (n int, err error)
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
