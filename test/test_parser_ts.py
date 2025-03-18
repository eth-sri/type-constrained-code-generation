import unittest
from functools import partial

from transformers import AutoTokenizer

from typesafe_llm.parser.parser_ts import parse_ts_program
from typesafe_llm.parser.types_ts import (
    reachable,
    FunctionPType,
    TypeParameterPType,
    ArrayPType,
    NumberPType,
    AbsTuplePType,
    TuplePType,
    StringPType,
    AnyPType,
)
from typesafe_llm.parser.parser_base import (
    incremental_parse,
    IncrementalParserState,
    TerminalParserState,
)
from test.utils import (
    assert_partial,
    assert_reject,
    assert_strict_partial,
    assert_weak_full,
    assert_just_before_reject_generic,
)

assert_just_before_reject = partial(
    assert_just_before_reject_generic, parse_ts_program, incremental_parse
)

TOKEN_REMAP = {
    # llama special tokens
    "Ġ": " ",
    "Ċ": "\n",
    "ĉ": "\t",
    # gemma special token
    "▁": " ",
}

MODEL_VOCAB_OFFSET = {
    "google/gemma-2b-it": -1,
}


def remap(v: str):
    res = v
    for old, new in TOKEN_REMAP.items():
        res = res.replace(old, new)
    return res


starcoder_tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
starcoder_vocab = {remap(x) for x in starcoder_tokenizer.get_vocab()}


def test_ite():
    states = parse_ts_program("let a:number = 2;if(a){a;}else{a;}")
    assert_weak_full(states)


def test_if():
    states = parse_ts_program("let a:number = 2;if(a){a;}")
    assert_weak_full(states)


def test_assignment_undefined():
    assert_just_before_reject("let a:number = b")


def test_assignment_numerical():
    states = parse_ts_program("let a:number = 2;a;")
    assert_weak_full(states)


def test_numericals():
    states = parse_ts_program("let a:number = 2.e100;")
    assert_weak_full(states)
    states = parse_ts_program("let a:number = 2e100;")
    assert_weak_full(states)
    states = parse_ts_program("let a:number = .2e-100;")
    assert_weak_full(states)
    states = parse_ts_program("let a:number = 0x2;")
    assert_weak_full(states)
    states = parse_ts_program("let a:number = 0b1;")
    assert_weak_full(states)
    states = parse_ts_program("let a:number = 0o1;")
    assert_weak_full(states)
    states = parse_ts_program("let a:number = 0b3;")
    assert_reject(states)
    states = parse_ts_program("let a:number = 0o9;")
    assert_reject(states)
    states = parse_ts_program("let a:number = 0xZ;")
    assert_reject(states)
    states = parse_ts_program("let a:number = 0..;")
    assert_reject(states)


def test_assignment_string():
    states = parse_ts_program('let a:string = "hello";a;')
    assert_weak_full(states)


def test_assignment_type_messup():
    assert_just_before_reject('let a:number = "2";')


def test_assignment_type_messup_2():
    assert_just_before_reject("let a:number = true;")


def test_assignment_type_ok():
    assert_just_before_reject("let a:string = 2.o")


@unittest.skip("does not work due to attributes")
def test_assignment_type_messup_var_recoverable():
    assert_just_before_reject('let bef:string = "";let b:number = 2;let a:number = be')


def test_assignment_type_messup_var_recoverable_2():
    assert_just_before_reject(
        'let bef:string = "";let befi:number = 2;let a:number = bef;'
    )


@unittest.skip("does not work due to attributes")
def test_assignment_type_messup_var_unrecoverable():
    assert_just_before_reject('let b:string = "2";let a:number = b')


def test_assignment_type_messup_var_unrecoverable_2():
    assert_just_before_reject('let bef:string = "";let a:number = ba')


def test_assignment():
    states = parse_ts_program(
        "let a:number = 2;let b:number = 2;if(a){let c:number = b;}else{b;}"
    )
    assert_weak_full(states)


def test_illegal_redeclaration():
    assert_just_before_reject("let a:number = 2;let b:number = 2;if(a){let a ")


def test_recoverable_program():
    states = parse_ts_program("let a:number = 2;if(a){a;}else{")
    assert_strict_partial(states)


def test_ungrammatical_program2():
    states = parse_ts_program("let a:number = 2;if(a){a;else{a;}")
    assert_reject(states)
    assert_just_before_reject("let a:number = 2;if(a){a;x")


def test_no_expr_return_outside_fun():
    states = parse_ts_program("let a:number = 2;return a;")
    assert_reject(states)
    assert_just_before_reject("let a:number = 2;ret")


def test_no_void_return_outside_fun():
    states = parse_ts_program("let a:number = 2;return;")
    assert_reject(states)
    assert_just_before_reject("let a:number = 2;ret")


def test_fun_def_wrong():
    states = parse_ts_program("functiona(b:number):void{return;}")
    assert_reject(states)
    assert_just_before_reject("functiona")


def test_fun_def():
    states = parse_ts_program("function a(b:number):void{return;}")
    assert_weak_full(states)


def test_fun_def_three_params():
    states = parse_ts_program("function a(b:number,c:number,d:number):void{return;}")
    assert_weak_full(states)


def test_fun_def_no_params():
    states = parse_ts_program("function a():void{return;}")
    assert_weak_full(states)


def test_fun_var_name_conflict():
    states = parse_ts_program("let a:number = 2;function a(")
    assert_reject(states)


def test_fun_var_name_no_conflict():
    states = parse_ts_program("let a:number = 2;function b(")
    assert_strict_partial(states)


def test_fun_params_conflict():
    states = parse_ts_program("function a(a:")
    assert_reject(states)


def test_fun_use_params():
    states = parse_ts_program("function a(b:number):number{return b;}")
    assert_weak_full(states)


def test_fun_call():
    states = parse_ts_program("function a(b:number):number{return b;}a(2);")
    assert_weak_full(states)


def test_fun_call_empty():
    states = parse_ts_program("function a():number{return 1;}a();")
    assert_weak_full(states)


@unittest.skip("does not work due to attributes")
def test_fun_call_invalid_param():
    states = parse_ts_program('function a(b:number):number{return b;}a("')
    assert_reject(states)


def test_fun_call_invalid_param_2():
    assert_just_before_reject(
        'let x:string = "2"; function a(b:number):number{return b;}a(x)'
    )


def test_fun_no_return():
    states = parse_ts_program("function a():number{2;}")
    assert_reject(states)


def test_fun_return_branches():
    states = parse_ts_program(
        "function a():number{if(2){return 2;}else{return 3;}return 1;}"
    )
    assert_weak_full(states)


def test_fun_some_return_branches():
    states = parse_ts_program("function a():number{if(2){return 2;}else{3;}return 1;}")
    assert_weak_full(states)


def test_fun_some_return_branches_if():
    states = parse_ts_program("function a():number{if(2){return 2;}return 1;}")
    assert_weak_full(states)


def test_fun_missing_return_branches_if():
    states = parse_ts_program("function a():number{if(2){return 2;}}")
    assert_reject(states)


def test_fun_fewer_return_branches():
    states = parse_ts_program("function a():number{if(2){2;}else{3;}return 1;}")
    assert_weak_full(states)


def test_fun_no_return_void():
    states = parse_ts_program("function a():void{2;}")
    assert_weak_full(states)


def test_fun_return_void():
    states = parse_ts_program("function a():void{return;}")
    assert_weak_full(states)


@unittest.skip(
    "does not work due to attributes - TODO would actually not work if literals were parsed as non-attributable"
)
def test_fun_return_wrong_void():
    assert_just_before_reject("function a():void{return 2")


def test_fun_return_wrong_void_2():
    assert_just_before_reject('function a():void{return "2";')


def test_fun_fewer_return_branches_void():
    states = parse_ts_program("function a():void{if(2){2;}else{return;}return;}")
    assert_weak_full(states)


def test_fun_inherit_return_void():
    assert_just_before_reject("function a():void{if(2){2;}else{return 2;")


def test_fun_assignment():
    states = parse_ts_program(
        "function a():void{if(2){2;}else{return;}return;}let b:()=>void = a;"
    )
    assert_weak_full(states)


def test_fun_assignment_primitive():
    states = parse_ts_program(
        "function a():void{if(2){2;}else{return;}return;}let b:number = a;"
    )
    assert_reject(states)


def test_rec_fun_call():
    states = parse_ts_program(
        "function a(b:number):()=>number{function c():number{return b;}return c;}a(2)();"
    )
    assert_weak_full(states)


def test_rec_fun_call_constrained():
    states = parse_ts_program(
        "function a(b:number):()=>number{function c():number{return b;}return c;}let x:number = a(2)();"
    )
    assert_weak_full(states)


def test_scope_identifiers():
    states = parse_ts_program("function a(b:number):number{return b;}let c:number = b;")
    assert_reject(states)


def test_scope_identifiers_let():
    states = parse_ts_program(
        "function a(b:number):number{let x:number = 2;return b;}let c:number = x;"
    )
    assert_reject(states)


@unittest.skip("does not work due to attributes")
def test_rec_fun_call_constrained_reject_early():
    assert_just_before_reject(
        "function a(b:string):()=>string{function c():string{return b;}return c;}let x:number = a"
    )


@unittest.skip("does not work due to attributes")
def test_rec_fun_call_constrained_reject_early_2():
    states = parse_ts_program(
        "function a(b:number):()=>number{function c():number{return b;}return c;}function y(b:string):()=>string{function c():string{return b;}return c;}let x:number = y"
    )
    assert_reject(states)


def test_comment():
    states = parse_ts_program("function a(b:number):()=>number{//helloworld")
    assert_strict_partial(states)


@unittest.skip
def test_comment_empty():
    states = parse_ts_program("function a(b:number):()=>number{//\nlet")
    assert_strict_partial(states)


@unittest.skip
def test_comment_2():
    states = parse_ts_program("function a//helloworld")
    assert_strict_partial(states)


def test_fun_call_reject():
    states = parse_ts_program("function abc(b:number):()=>number{return b;}ab(")
    assert_reject(states)


def test_arith_numerical():
    states = parse_ts_program("let a:number = 2;let b:number = a - 2;")
    assert_weak_full(states)


def test_arith_string():
    states = parse_ts_program('let a:string = "2";let b:string = a + "2";')
    assert_weak_full(states)


def test_arith_reject():
    states = parse_ts_program("let a:number = 2;let b:string = b - ")
    assert_reject(states)


def test_arith_reject_mul():
    states = parse_ts_program("let a:number = 2 * *")
    assert_reject(states)


def test_arith_reject_call():
    states = parse_ts_program("let a:number = 2 ( (")
    assert_reject(states)


def test_allow_nesting1():
    states = parse_ts_program("let a:number = ( ")
    assert_strict_partial(states)


def test_allow_nesting1_full():
    states = parse_ts_program("let a:number = ( 1 );")
    assert_weak_full(states)


def test_allow_nesting_2():
    states = parse_ts_program("let a:number = ( (")
    assert_strict_partial(states)


def test_allow_ws():
    states = parse_ts_program("let a:number = 2;  a;")
    assert_weak_full(states)


def test_allow_sub():
    states = parse_ts_program("let a:number = 2; let b:number = a - a;")
    assert_weak_full(states)


def test_allow_array_type():
    states = parse_ts_program("let a:number[] ")
    assert_strict_partial(states)
    states = parse_ts_program("let a:number[ ] ")
    assert_strict_partial(states)
    states = parse_ts_program("let a:number [ ] ")
    assert_strict_partial(states)
    states = parse_ts_program("let a:number [ ] ")
    assert_strict_partial(states)


def test_allow_array_value():
    states = parse_ts_program("let a:number[] = [1];")
    assert_weak_full(states)


def test_allow_array_many_value():
    states = parse_ts_program("let a:number[] = [1,2,3];")
    assert_weak_full(states)


def test_allow_array_reject_later_wrong():
    states = parse_ts_program('let a:number[] = [1,"hello"];')
    assert_reject(states)


@unittest.skip("does not work due to attributes")
def test_allow_array_reject_early_wrong():
    assert_just_before_reject('let a:number[] = [1,"')


def test_allow_array_reject_early_wrong_2():
    states = parse_ts_program('let a:number[] = [1,"hello",')
    assert_partial(states)


def test_allow_array_nested_value():
    states = parse_ts_program("let a:number[][] = [[1],[2],[3]];")
    assert_weak_full(states)


@unittest.skip("does not work due to attributes")
def test_reject_array_nested_invalid_value():
    assert_just_before_reject('let a:number[][] = [[1],[2],["')


@unittest.skip("can actually be saved via '? [1] : [2]]'")
def test_reject_array_nested_invalid_value_2():
    assert_just_before_reject('let a:number[][] = [[1],[2],["2"]')


def test_reject_array_value_invalid():
    states = parse_ts_program('let a:number[] = ["h"]; ')
    assert_reject(states)


@unittest.skip("does not work due to attributes")
def test_reject_array_value_invalid_early():
    assert_just_before_reject('let a:number[] = ["')


def test_reject_array_value_invalid_early_2():
    assert_just_before_reject('let a:number[] = [""];')


def test_allow_array_access():
    states = parse_ts_program("let a:number[] = [1,2]; let b:number = a[2];")
    assert_weak_full(states)


def test_allow_array_access_nested():
    states = parse_ts_program("let a:number[] = [1,2]; let b:number = a[a[2]];")
    assert_weak_full(states)


def test_allow_array_access_indirect():
    states = parse_ts_program(
        "let a:number[] = [1,2];let c:number=2; let b:number = a[c];"
    )
    assert_weak_full(states)


@unittest.skip("does not work due to attributes")
def test_reject_invalid_array_access():
    assert_just_before_reject('let a:string[] = ["1","2"]; let b:number = a[')


@unittest.skip("does not work due to attributes")
def test_reject_invalid_array_writing():
    assert_just_before_reject("let a:string[] = [1,")


def test_reject_invalid_array_access_index():
    assert_weak_full(
        parse_ts_program(
            'let a:number[] = [1,2]; let c:string[] = ["hello"]; let b:number = a[c[1].indexOf("l")];'
        )
    )


def test_low_ambiguity():
    states = parse_ts_program(
        """\
function parse_nested_parens(paren_string: string): number[] {
  let i: number = 0""",
        print_failure_point=True,
    )
    assert states, "Empty states"
    assert len(states) < 100, f"Too many states: {len(states)}"


def test_very_low_ambiguity():
    states = parse_ts_program(
        """\
function parse_nested_parens(paren_string: string): number[] {
  let stack: number[] = [0];
  let result: number[] = [0];
  let i: number = 0"""
    )
    assert states, "Empty states"
    assert len(states) < 100, f"Too many states: {len(states)}"


def test_for_loop():
    states = parse_ts_program("""for(let i:number = 0; i; i){i;}""")
    assert_weak_full(states)


def test_for_loop_reject_stmts():
    states = parse_ts_program("""for(let i:number = 0; i; let x:number = 5){i;}""")
    assert_reject(states)


def test_for_loop_reject_stmts_2():
    states = parse_ts_program("""for(let i:number = 0; let x:number = i; i){i;}""")
    assert_reject(states)


def test_for_loop_reject_ids():
    states = parse_ts_program("""for(let i:number = 0; j; i){i;}""")
    assert_reject(states)
    assert_just_before_reject("for(let i:number = 0; j")


def test_while_loop():
    states = parse_ts_program("""let i:number = 0; while(i){i;}""")
    assert_weak_full(states)


def test_do_while_loop():
    states = parse_ts_program("""let i:number = 0; do{i;}while(i)""")
    assert_weak_full(states)


def test_reachable_typ_analysis():
    # this demonstrates why functions should be considered to have a deeper depth
    states = parse_ts_program("function a():number[][]{return [[1]];} let b:number = a")
    assert_strict_partial(states)


def test_reach_str_from_number_toString():
    # this demonstrates why functions should be considered to have a deeper depth
    states = parse_ts_program("let a:number = 1; let b:string = a.toString();")
    assert_weak_full(states)


def test_bool_numerical():
    states = parse_ts_program("let a:number = 2;let b:boolean = a < 2;")
    assert_weak_full(states)


def test_bool_string():
    states = parse_ts_program('let a:string = "2";let b:boolean = a < "2";')
    assert_weak_full(states)


def test_bool_reject():
    states = parse_ts_program("let a:number = 2;let b:string = a < b;")
    assert_reject(states)


def test_const_assign():
    states = parse_ts_program("const a:number = 2;")
    assert_weak_full(states)


def test_multiline_comment():
    states = parse_ts_program(
        "function foo():void{/*some comment * */ const a:number = Math}"
    )
    assert_reject(states)


def test_untyped_const_assign():
    states = parse_ts_program("const a = 2; a + 5;")
    assert_weak_full(states)


def test_untyped_let_assign():
    states = parse_ts_program("let a = 2; a + 3;")
    assert_weak_full(states)


def test_reassignment():
    states = parse_ts_program("let a = 2;a")
    states = incremental_parse(states, " = a + 3;")
    assert_weak_full(states)


def test_reassignment_wrong_type():
    assert_just_before_reject('let a = 2;a = "hello";')


def test_reassignment_const():
    assert_just_before_reject("const a = 2;a = ")


def test_reassignment_nonexisting():
    assert_just_before_reject("let b = 2;a")


def test_fun_name_reassigment_disallowed():
    assert_just_before_reject("function a(b:number):void{return;} a = ")


def test_reassignment_prefix():
    states = parse_ts_program("let a = 2; ++a;")
    assert_weak_full(states)


def test_reassignment_prefix_wrong_type():
    # Note this is only true because attributes of a that result in int are read-only
    # Note extra tricky: it would be fine for arrays (++array.length) but not for strings
    assert_just_before_reject('let a = "2"; ++a')


@unittest.skip(
    "annoying, this can also be parsed as +(+a) -- issue of tokenization --> fixable but annoying"
)
def test_reassignment_prefix_const():
    assert_just_before_reject("const a = 2; ++a")


def test_reassignment_prefix_nonexisting():
    assert_just_before_reject("let b = 2; ++a")


def test_unop_plus_prefix():
    states = parse_ts_program("let a = 2; +a;")
    assert_weak_full(states)


def test_unop_minus_prefix():
    states = parse_ts_program("let a = 2; -a;")
    assert_weak_full(states)


def test_reassignment_postfix():
    states = parse_ts_program("let a = 2; a++;")
    assert_weak_full(states)


def test_reassignment_postfix_wrong_type():
    assert_just_before_reject('let a = "2"; a++')


def test_reassignment_postfix_const():
    assert_just_before_reject("const a = 2; a++")


def test_reassignment_postfix_nonexisting():
    assert_just_before_reject("let b = 2; a")


def test_reassignment_postfix_nonref():
    assert_just_before_reject("let a = 2; 2++")


def test_Math():
    states = parse_ts_program("let a = Math.trunc(3.2);")
    assert_weak_full(states)


def test_bool_literal():
    states = parse_ts_program("let a = true;")
    assert_weak_full(states)
    states = parse_ts_program("let a = false;")
    assert_weak_full(states)


def test_empty_string_array_literal_declaration():
    states = parse_ts_program("const result: string[] = [];")
    assert_weak_full(states)


def test_empty_string_array_literal_assignment():
    states = parse_ts_program("let result: string[] = []; result = [];")
    assert_weak_full(states)


def test_assignment_to_declaration_disallowed():
    states = parse_ts_program("(let a:number)++")
    assert_reject(states)


def test_assignment_to_array():
    states = parse_ts_program("let a: number[] = [1]; a[0] = 2;")
    assert_weak_full(states)


def test_assignment_to_array_literal():
    states = parse_ts_program("[1][0] = 2;")
    assert_weak_full(states)


def test_for_of_typed_arrayacc_loop():
    states = parse_ts_program("let a: number[] = [1, 2]; for (a[0] of a) {a[0]+1;}")
    assert_weak_full(states)


def test_for_of_typed_loop():
    states = parse_ts_program(
        "let a: number[] = [1, 2]; let b:number = 2; for (b of a) {b+1;}"
    )
    assert_weak_full(states)


def test_for_of_loop():
    states = parse_ts_program("let a: number[] = [1, 2]; for (let b of a) {b+1;}")
    assert_weak_full(states)


def test_add_assignment():
    states = parse_ts_program("let a: number = 1; a += 1;")
    assert_weak_full(states)


def test_exp_assignment():
    states = parse_ts_program("let a: number = 1; a **= 1;")
    assert_weak_full(states)


def test_update_assignment_declaration():
    assert_just_before_reject("let a: number +")


def test_exp_assignment_string():
    assert_just_before_reject('let a: string = "1"; a /')


def test_return_nested_fun():
    states = parse_ts_program(
        """
function foo(): (ar: number[]) => number[] {
    return (ar) => ar;
}""",
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_return_nested_typefun():
    states = parse_ts_program(
        """
function foo(): ((ar: number[]) => number)[] {
    let fo1: (ar:number[]) => number = (ar) => ar[1];
    return [fo1];
}
    """
    )
    assert_weak_full(states)


def test_lambdas():
    # we don't enforce the full three parameters
    states = parse_ts_program(
        """
let numbers: number[] = [1];
const mean = numbers.reduce((acc: number, val)"""
    )
    assert_strict_partial(states)


def test_lambdas_2():
    states = parse_ts_program(
        """
let numbers: number[] = [1];
const mean: number = numbers.reduce((acc: number, val, ind, arr) => acc + val, 0);"""
    )
    assert_weak_full(states)


def test_lambdas_3():
    # at least two params have to be given
    assert_just_before_reject(
        """
let numbers: number[] = [1];
const mean = numbers.reduce((acc)"""
    )


def test_return_empty_array():
    states = parse_ts_program("function foo(): number[] {return [];}")
    assert_weak_full(states)


def test_ws_at_end():
    states = parse_ts_program(
        """
    let a: number = 1;
"""
    )
    assert_weak_full(states)


def test_access_str():
    states = parse_ts_program(
        """
    let a: string = "1";
    let b: string = a[0];
"""
    )
    assert_weak_full(states)


def test_for_of_str():
    states = parse_ts_program(
        """
    let a: string = "1";
    for (let b of a) {b;}
"""
    )
    assert_weak_full(states)


@unittest.skip(
    "does actually not work without considering the identifiers for attributes"
)
def test_parse_parens():
    states = parse_ts_program(
        """
let parenString: string = "()";
let x: number[] = parenString.split(' ', 100000).filter((x, _, _1) => x).map((x,y,z) => [x.length]);
"""
    )
    assert_weak_full(states)


def test_no_overwrite_return_type():
    states = parse_ts_program(
        """
function parseNestedParens(parenString: string): number[] {

    function parseParenGroup(s: string): number {
        return 0;
    }

    return [0];
}
""",
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_precedence_arith():
    states = parse_ts_program(
        """
let a: number = 1 + 2 * 3;
"""
    )
    assert_weak_full(states)
    assert len([x for x in states if x.accept]) == 1


def test_type_search_member_access():
    states = parse_ts_program(
        """
function isPalindrome(str: string): boolean {
    /** Test if given string is a palindrome */
    return str === str.split('', 1000).reverse().join('');
}
"""
    )
    assert_weak_full(states)


def test_optional_arguments():
    states = parse_ts_program(
        """
    let str: string = "hello";
    str + str.substring(0, 100).substring(10).split('').reverse().join(''); 
     """
    )
    assert_weak_full(states)


def test_all_optional_arguments():
    states = parse_ts_program(
        """
    let str: string = "hello";
    str + str.concat(); 
     """
    )
    assert_weak_full(states)


def test_lambda_less_args():
    states = parse_ts_program(
        """
let parenString: string = "()";
let x: string[] = parenString.split(' ', 100000).filter((y) => y);
"""
    )
    assert_weak_full(states)


def test_empty_array_to_num_declaration():
    states = parse_ts_program("const result: number = [].length;")
    assert_weak_full(states)


def test_extreme_exclamation_1():
    states = parse_ts_program("let a: boolean = !!'hello';")
    assert_weak_full(states)


def test_number():
    states = parse_ts_program("let a: number = 1439")
    assert_strict_partial(states)


def test_extreme_exclamation():
    states = parse_ts_program("let a: boolean = !!!!!!!")
    assert_strict_partial(states)


def test_extreme_plus():
    assert_just_before_reject("let a: number = +++")


def test_member_access_empty_array():
    assert_weak_full(parse_ts_program("let a: number = [].length;"))


def test_crash():
    parse_ts_program(
        """
function is_sorted(numbers: number[]): boolean {
	return numbers.every(([].
"""
    )


def test_pop_empty_list():
    pop_states = parse_ts_program(
        """
//Given a positive integer n, you have to make a pile of n levels of stones.
// The first level has n stones.
// The number of stones in the next level is:
// - the next odd number if n is odd.
// - the next even number if n is even.
// Return the number of stones in each level in an array, where element at index
// i represents the number of stones in the level (i+1).
// Examples:
// >>> make_a_pile(3)
// [3, 5, 7]
function make_a_pile(n: number): number[] {
    let pile: number[] = [];
    let level: number = 0;
    let odd: number = 0;
    let even: number = 0;
    let i: number = 0;
    while (n > 0) {
        if (n % 2 === 0) {
            even++;
            n ==(n - even) / 2;
        } else {
            odd++;
            n ==(n - odd) / 2;
        }
        pile.push
"""
    )
    for v in starcoder_vocab:
        try:
            incremental_parse(pop_states, v)
        except Exception as e:
            print(f"Failed on {v}")
            raise e


def test_ternary():
    states = parse_ts_program(
        """
let a: number = 1;
let b: number = 2;
let c: number = a < b ? a : b;
"""
    )
    assert_weak_full(states)


def test_modify_fun_param():
    states = parse_ts_program(
        """
function modify(a: number): number {
    a = a + 1;
    return a;
}
"""
    )
    assert_weak_full(states)


def test_logic_ops():
    states = parse_ts_program(
        """
let a: boolean = true;
let b: boolean = false;
let c: boolean = a && b;
"""
    )
    assert_weak_full(states)


def test_lambda_stmts():
    states = parse_ts_program(
        """
let arr = [1, 2, 3];
arr.sort((a, b) => {     
    const aBinary = a.toString(2);     
    const bBinary = b.toString(2);     
    const aOnes = aBinary.split('').filter(x => x === '1').length;     
    const bOnes = bBinary.split('').filter(x => x === '1').length;     
    if (aOnes === bOnes) {     
      return a - b;     
    }     
    return aOnes - bOnes;     
});     
"""
    )
    assert_weak_full(states)


def test_else_if():
    states = parse_ts_program(
        """
let sum = 0;     
let lst = [1, 2, 3, 4, 5];
let i = 1;
if (i % 3 == 0) {     
    sum += lst[i] * lst[i];     
} else if (i % 4 == 0) {     
    sum += lst[i] * lst[i] * lst[i];     
}    
"""
    )
    assert_weak_full(states)


def test_else_if_else():
    states = parse_ts_program(
        """
let sum = 0;     
let lst = [1, 2, 3, 4, 5];
let i = 1;
if (i % 3 == 0) {     
    sum += lst[i] * lst[i];     
} else if (i % 4 == 0) {     
    sum += lst[i] * lst[i] * lst[i];     
} else {     
    sum += lst[i];     
}     
"""
    )
    assert_weak_full(states)


def test_arith_mixed_types():
    states = parse_ts_program(
        """
let i: string = "hello" + [1,2];
let l: string = "hello" + 1;
let k: number = 1 + 2;
let j: string = 1 + "hello";
let q: string = [1,2] + "hello";
"""
    )
    assert_weak_full(states)
    states = parse_ts_program(
        """
let k: number = 1 + true;
"""
    )
    assert_reject(states)
    states = parse_ts_program(
        """
let k: number = [1,2] + 1;
"""
    )
    assert_reject(states)
    states = parse_ts_program(
        """
let k: number = "hello" + 1;
"""
    )
    assert_reject(states)


def test_not_derivable():
    states = parse_ts_program("let a: number = !!'hello'")
    # definitely derivable through " ? 1 : 0"
    assert_strict_partial(states)


def test_not_derivable2():
    states = parse_ts_program("let a: number = 2; let b: number[] = a")
    # definitely reachable through " ? [1,2] : [3,4]"
    assert_strict_partial(states)


def test_not_derivable3():
    states = parse_ts_program("let b: number = 2 ? 'hi' : ")
    # definitely not derivable anymore, ? has the lowest precedence and union return type is string | number at best
    assert_reject(states)


def test_not_derivable4():
    states = parse_ts_program("let a: number = 2; let b: number = (a ? 'hi' : ")
    # definitely derivable through " 'hello').length"
    assert_strict_partial(states)


def test_not_derivable5():
    states = parse_ts_program("let a: number = 2; let b: number = a + ('hello' + ")
    # definitely derivable through " 'you' + ).length"
    assert_strict_partial(states)


def test_not_derivable6():
    states = parse_ts_program("let a: number = 2; let b: number = a + 'hello' + ")
    # definitely derivable through "  'x' ? 1 : 0"
    assert_strict_partial(states)


def test_not_derivable7():
    states = parse_ts_program("let a: number = 1 + !!'hello'")
    # not derivable
    assert_reject(states)


def test_not_derivable8():
    states = parse_ts_program("let a: number = 2; let b: number[] = [1].concat(a")
    # definitely reachable through " ? [1,2] : [3,4]"
    assert_strict_partial(states)


def test_not_derivable9():
    states = parse_ts_program("let a: number = 2; let b: number = 4 + (a ? 'hi' : ")
    # definitely derivable through " 'hello').length"
    assert_strict_partial(states)


def test_not_derivable10():
    states = parse_ts_program("let a: number = 2; let b: number = a * 'hello' + ")
    # definitely not derivable (cannot turn "hello" + into a number anymore, also a * 'hello' binds stronger)
    assert_reject(states)


def test_not_derivable11():
    states = parse_ts_program("let a: number = 2; let b: number = a && 'hello' + ")
    # definitely derivable through "  'x' ? 1 : 0"
    assert_strict_partial(states)


def test_not_derivable12():
    assert_just_before_reject("let b: number = 'hey' *")
    # definitely not derivable anymore, "string" * is not a valid operation


def test_not_derivable13():
    assert_just_before_reject("let b: boolean = 1 < 'hey' +")
    # definitely not derivable anymore, < binds weaker than + but strong than ? --> rhs can only be string and number < string is not allowed


def test_not_derivable_14():
    states = parse_ts_program(
        """\
let x = 100 *! 100"""
    )
    assert_reject(states)


def test_template_literal():
    states = parse_ts_program(
        """
let a: number = 1;
let b: string = `hello ${a}`;
"""
    )
    assert_weak_full(states)


def test_full_humaneval():
    states = parse_ts_program(
        """\
//Return a string containing space-delimited numbers starting from 0 upto n inclusive.
// >>> string_sequence(0)
// "0"
// >>> string_sequence(5)
// "0 1 2 3 4 5"
function string_sequence(n: number): string {
        let str = "";
        for (let i = 0; i <= n; i++) {
                str += i + " ";
        }
        return str.trim();
}
""",
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_full_humaneval_2():
    states = parse_ts_program(
        """\
//Return array with elements incremented by 1.
// >>> incr_list([1, 2, 3])
// [2, 3, 4]
// >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
// [6, 4, 6, 3, 4, 4, 10, 1, 124]
function incr_list(l: number[]): number[] {
        return l.map(x => x + 1);
}
"""
    )
    assert_weak_full(states)


def test_full_humaneval_3():
    states = parse_ts_program(
        """\
function total_match(lst1: string[], lst2: string[]): string[] {
        let sum1 = 0;
        let sum2 = 0;
        for (let i = 0; i < lst1.length; i++) {
                sum1 += lst1[i].length;
        }
        for (let i = 0; i < lst2.length; i++) {
                sum2 += lst2[i].length;
        }
        if (sum1 < sum2) {
                return lst1;
        } else if (sum1 > sum2) {
                return lst2;
        } else {
                return lst1;
        }
}
"""
    )
    assert_weak_full(states)


def test_for_of():
    states = parse_ts_program(
        """\
function total_match(lst1: string[], lst2: string[]): string[] {
        let sum1 = 0;
        let sum2 = 0;
        for (let i of lst1) {
                sum1 += i.length;
        }
        for (let i of lst2) {
                sum2 += i.length;
        }
        if (sum1 < sum2) {
                return lst1;
        } else if (sum1 > sum2) {
                return lst2;
        } else {
                return lst1;
        }
}
"""
    )
    assert_weak_full(states)


def test_illegal():
    states = parse_ts_program(
        """\
let x = 0; x < 100; x += 1.0 / 100.0; ++x
    | x <<- [0, 100]

[100] + 10

++ 100

* 100

*! 100"""
    )
    assert_reject(states)


def test_array_length_assign():
    states = parse_ts_program(
        """\
let result: number[] = [1, 2, 3];
result.length = 0;"""
    )
    assert_weak_full(states)


def test_int_fun_assign():
    states = parse_ts_program(
        """\
let result: number = 1;
result.valueOf = (10).valueOf;"""
    )
    assert_weak_full(states)


def test_array_length_assign_suff_op():
    states = parse_ts_program(
        """\
let result: number[] = [1];
result.length++;"""
    )
    assert_weak_full(states)


def test_array_length_assign_pref_op():
    states = parse_ts_program(
        """\
let result: number[] = [1, 2, 3];
++result.length;"""
    )
    assert_weak_full(states)


def test_array_index_assign_pref_op():
    states = parse_ts_program(
        """\
let result: number[] = [1];
++result[0];"""
    )
    assert_weak_full(states)


def test_array_index_assign_suff_op():
    states = parse_ts_program(
        """\
let result: number[] = [1, 2, 3];
result[0]++;"""
    )
    assert_weak_full(states)


def test_unassignable_member_assignment():
    states = parse_ts_program(
        """
let x = "hello";
x.length = 0;
"""
    )
    assert_reject(states)


def test_unassignable_member_post():
    states = parse_ts_program(
        """
let x = "hello";
x.length++;
"""
    )
    assert_reject(states)


def test_union_types1():
    states = parse_ts_program(
        """\
let b: number | string = 0;"""
    )
    assert_weak_full(states)


def test_union_types2():
    states = parse_ts_program(
        """\
let b: number | string = "";"""
    )
    assert_weak_full(states)


def test_union_types3():
    states = parse_ts_program(
        """\
let b: number | string = true;"""
    )
    assert_reject(states)


def test_union_types4():
    states = parse_ts_program(
        """\
let a: number | string = "";
let b: number = a;"""
    )
    assert_reject(states)


def test_union_types5():
    states = parse_ts_program(
        """\
# let a: number | string = 1;
# let b: number = a;"""
    )
    assert_reject(states)


def test_tuple_types1():
    states = parse_ts_program(
        """\
let b: [string, number] = ["", 0];"""
    )
    assert_weak_full(states)


def test_tuple_types2():
    states = parse_ts_program(
        """\
let b: [string, number] = [0, ""];"""
    )
    assert_reject(states)


def test_tuple_types3():
    states = parse_ts_program(
        """\
let b: [string] = [""];"""
    )
    assert_weak_full(states)


def test_tuple_types4():
    states = parse_ts_program(
        """\
let b: [] = [];"""
    )
    assert_weak_full(states)


def test_tuple_types5():
    states = parse_ts_program(
        """\
let b: [string] = [""];
let c: string[] = b;"""
    )
    assert_reject(states)


def test_tuple_types6():
    states = parse_ts_program(
        """\
let a:[number,number,number,number,number,number,number,number,number,number]"""
    )
    assert_partial(states)


def test_tuple_types7():
    states = parse_ts_program(
        """\
let a:[number,number,number,number,number,number,number,number,number,number,number]"""
    )
    assert_partial(states)


def test_tuple_types8():
    states = parse_ts_program(
        """\
let a = [1, "a"][0];"""
    )
    assert_weak_full(states)


def test_tuple_types9():
    states = parse_ts_program(
        """\
let a:number = [1, "a"][0];"""
    )
    assert_weak_full(states)


def test_tuple_types10():
    states = parse_ts_program(
        """\
let a = [1, "a"][1];"""
    )
    assert_weak_full(states)


def test_tuple_types11():
    states = parse_ts_program(
        """\
let a:number = [1, "a"][1];"""
    )
    assert_reject(states)


def test_tuple_types12():
    states = parse_ts_program(
        """\
let a:[number, string]= [1, "a"];
a[0] = 2;"""
    )
    assert_weak_full(states)


def test_tuple_types13():
    states = parse_ts_program(
        """\
let a:[number, string]= [1, "a"];
a[1] = 2;"""
    )
    assert_reject(states)


def test_tuple_types14():
    states = parse_ts_program(
        """\
let a:[number,number,number,number,number,number,number,number,number,number,string]= [1, 1,1,1,1,1,1,1,1,1,"a"];
let b:string = a[10];"""
    )
    assert_weak_full(states)


def test_reachable_paramtertype():
    assert reachable(
        FunctionPType(
            (TypeParameterPType("T"),),
            ArrayPType(TypeParameterPType("T")),
            0,
        ),
        ArrayPType(NumberPType()),
        (5, "left"),
        (100, "right"),
        in_array=[],
        in_nested_expression=[],
        in_pattern=[],
        max_depth=(0, 0),
    )


def test_parse_map_type_param():
    states = parse_ts_program(
        """
let parenString: string[] = ["("];
let x: number[] = parenString.map((x,y,z) => x.length);
"""
    )
    assert_weak_full(states)


def test_parse_map_no_type_param():
    states = parse_ts_program(
        """
let parenString: string[] = ["("];
let x: string[] = parenString.map((x,y,z) => x);"""
    )
    assert_weak_full(states)


def test_parse_map_wrong_type_param():
    assert_just_before_reject(
        """
let parenString: string[] = ["("];
let x: string[] = parenString.map((x,y,z) => x.length);"""
    )


@unittest.skip(
    "TODO does not work yet, somehow type passing into lambdas needs to be fixed"
)
def test_parse_reduce():
    states = parse_ts_program(
        """
let parenString: string[] = ["("];
let x: number = parenString.reduce((x: number, y) => y.length + x, 0);"""
    )
    assert_weak_full(states)


@unittest.skip("TODO re-enable parameterization for this")
def test_parse_reduce_explicit():
    states = parse_ts_program(
        """
function a(n: number, s: string): number {
  return s.length + n;
}
let parenString: string[] = ["("];
let x: number = parenString.reduce(a, 0);"""
    )
    assert_weak_full(states)


@unittest.skip(
    "TODO does not work yet, need to fix how followup parameters are changed after initialization"
)
def test_parse_reduce_explicit_wrong_type():
    states = parse_ts_program(
        """
function a(n: number, s: string): number {
  return s.length + n;
}
let parenString: string[] = ["("];
let x: number = parenString.reduce(a, "t")"""
    )
    assert_reject(states)


def test_parse_reduce_same_type():
    states = parse_ts_program(
        """
let parenString: string[] = ["("];
let x: string = parenString.reduce((x: string, y) => y + x);"""
    )
    assert_weak_full(states)


def test_parse_reduce_not_same_type():
    assert_just_before_reject(
        """
function a(n: number, s: string): number {
  return s.length + n;
}
let parenString: string[] = ["("];
let x: number = parenString.reduce(a)"""
    )


def test_not_derivable14():
    assert_just_before_reject("let b: boolean[] = ['h', 'y'].map((x) => 1 < 'hey' +")


def test_reachable_abstuple1():
    t1 = AbsTuplePType(types=[])
    t2 = TuplePType(types=[NumberPType()])
    assert t1 <= t2


def test_reachable_abstuple2():
    t1 = AbsTuplePType(types=[NumberPType()])
    t2 = TuplePType(types=[NumberPType()])
    assert t1 <= t2


def test_reachable_abstuple3():
    t1 = AbsTuplePType(types=[NumberPType(), NumberPType()])
    t2 = TuplePType(types=[NumberPType()])
    assert not (t1 <= t2)


def test_reachable_abstuple4():
    t1 = AbsTuplePType(types=[StringPType()])
    t2 = TuplePType(types=[NumberPType()])
    assert not (t1 <= t2)


def test_reachable_abstuple5():
    t1 = AbsTuplePType(types=[NumberPType(), AbsTuplePType(types=[StringPType()])])
    t2 = TuplePType(types=[NumberPType(), StringPType()])
    assert not (t1 <= t2)


def test_reachable_abstuple6():
    t1 = AbsTuplePType(types=[NumberPType(), AbsTuplePType(types=[StringPType()])])
    t2 = TuplePType(types=[NumberPType(), TuplePType(types=[StringPType()])])
    assert t1 <= t2


def test_reachable_abstuple7():
    t1 = AbsTuplePType(types=[NumberPType(), AbsTuplePType(types=[StringPType()])])
    t2 = TuplePType(
        types=[NumberPType(), TuplePType(types=[StringPType(), NumberPType()])]
    )
    assert t1 <= t2


def test_reachable_abstuple8():
    t1 = AbsTuplePType(
        types=[NumberPType(), AbsTuplePType(types=[StringPType(), NumberPType()])]
    )
    t2 = TuplePType(types=[NumberPType(), TuplePType(types=[StringPType()])])
    assert not (t1 <= t2)


def test_reachable_abstuple9():
    t1 = AbsTuplePType(types=[AnyPType()])
    t2 = TuplePType(types=[NumberPType()])
    assert t1 <= t2


def test_reachable_abstuple10():
    t1 = AbsTuplePType(types=[NumberPType(), AnyPType()])
    t2 = TuplePType(types=[NumberPType(), TuplePType(types=[StringPType()])])
    assert t1 <= t2


def test_spread_array_init1():
    states = parse_ts_program(
        """\
let a:number[] = [...[1, 2, 3]];"""
    )
    assert_weak_full(states)


def test_spread_array_init2():
    states = parse_ts_program(
        """\
let a:number[] = [0, ...[1, 2, 3], 4];"""
    )
    assert_weak_full(states)


def test_spread_array_init3():
    states = parse_ts_program(
        """\
let a:string[] = [...[1, 2, 3]];"""
    )
    assert_reject(states)


def test_array_from_string():
    states = parse_ts_program(
        """
let a: string[] = Array.from("123");
"""
    )
    assert_weak_full(states)


def test_array_from_array():
    states = parse_ts_program(
        """
let a: string[] = Array.from(["1", "2", "3"]);
"""
    )
    assert_weak_full(states)


def test_array_from_array_wrong_type():
    states = parse_ts_program(
        """
let a: string[] = Array.from([1]);
"""
    )
    assert_reject(states)


def test_array_is_array():
    states = parse_ts_program(
        """
let a: boolean = Array.isArray([1]);
"""
    )
    assert_weak_full(states)


def test_set():
    states = parse_ts_program(
        """
let a = new Set([1, 2, 3]);
"""
    )
    assert_weak_full(states)


def test_new_non_cons():
    assert_just_before_reject(
        """
let a = new p"""
    )


def test_set_derivable():
    states = parse_ts_program(
        """
let a: number = new Set([1, 2, 3]).size;
"""
    )
    assert_weak_full(states)


def test_set_2():
    states = parse_ts_program(
        """
let a = new Set([1, 2, 3]);
a.add(1);
"""
    )
    assert_weak_full(states)


def test_set_3():
    assert_just_before_reject(
        """
let a = new Set([1, 2, 3]);
a.add("1")"""
    )


def test_inf_args_toSpliced():
    states = parse_ts_program(
        """
let a: string[] = ["1"].toSpliced(0, 1, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10");
"""
    )
    assert_weak_full(states)


def test_inf_args_toSpliced2():
    assert_just_before_reject(
        """
let a: string[] = ["1"].toSpliced(0, "1","""
    )


def test_inf_args_toSpliced_spread():
    states = parse_ts_program(
        """
let b: string[] = ["1"];
let a: string[] = ["1"].toSpliced(0, 1, ...b);
"""
    )
    assert_weak_full(states)


def test_inf_args_toSpliced_spread_2():
    states = parse_ts_program(
        """
let b: string[] = ["1"];
let a: string[] = ["1"].toSpliced(0, 1, "h", ...b, "you");
"""
    )
    assert_weak_full(states)


def test_inf_args_toSpliced_no_opt_arg():
    states = parse_ts_program(
        """
let b: string[] = ["1"];
let a: string[] = ["1"].toSpliced(0, 1);
"""
    )
    assert_weak_full(states)


def test_inf_args():
    states = parse_ts_program("let a: number = 2; let b: number[] = [1].concat([a]);")
    assert_weak_full(states)


def test_map():
    states = parse_ts_program(
        """
let a = new Map<number, number>([[1, 2]]).size;
"""
    )
    assert_weak_full(states)


@unittest.skip("actually possible using ternary ops")
def test_map_2():
    assert_just_before_reject(
        """
let a = new Map<number, number>(2"""
    )


def test_map_3():
    assert_just_before_reject(
        """
let a = new Map<number, string>([[1, "2"]]);
a.set(1, 2)"""
    )


def test_map_4():
    assert_weak_full(
        parse_ts_program(
            """
let a = new Map<number, string>([[1, "2"]]);
a.set(1, "hello");
"""
        )
    )


def test_regexp_cons():
    states = parse_ts_program(
        """
let a = new RegExp("a");
"""
    )
    assert_weak_full(states)


def test_regexp_literal():
    states = parse_ts_program(
        """
let a = /a/;
"""
    )
    assert_weak_full(states)


def test_regexp_literal_flags():
    states = parse_ts_program(
        """
let a = /a/gi;
"""
    )
    assert_weak_full(states)


def test_regexp_literal_invalid_flags():
    assert_just_before_reject(
        """
let a = /a/gh"""
    )


def test_str_match():
    states = parse_ts_program(
        """
let a = "hellO".match(new RegExp("a"))!;
""",
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_break():
    states = parse_ts_program("""for(let i:number = 0; i; i){break;}""")
    assert_weak_full(states)


def test_continue():
    states = parse_ts_program("""for(let i:number = 0; i; i){continue;}""")
    assert_weak_full(states)


def test_no_loop_continue():
    states = parse_ts_program("""continue;""")
    assert_reject(states)


def test_no_loop_break():
    states = parse_ts_program("""break;""")
    assert_reject(states)


def test_two_loop_continue():
    states = parse_ts_program(
        """for(let i:number = 0; i; i){for(let j:number = 0; j; j){continue;}}"""
    )
    assert_weak_full(states)


def test_two_loop_break():
    states = parse_ts_program(
        """for(let i:number = 0; i; i){for(let j:number = 0; j; j){break;}}"""
    )
    assert_weak_full(states)


def test_loop_if_break():
    states = parse_ts_program("""for(let i:number = 0; i; i){if(true){break;}}""")
    assert_weak_full(states)


def test_function_loop_reset():
    states = parse_ts_program(
        """for(let i:number = 0; i; i){function foo():number{break;}}"""
    )
    assert_reject(states)


def test_stmt_after_return():
    states = parse_ts_program(
        """
function largest_divisor(n: number): number {
    for (let i = n - 1; i >= 1; i--) {     
        if (n % i === 0) {
            return i;     
        }     
    }     
    return 1; // If no divisor is found, return 1 as the largest possible divisor
}
""",
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_replace_accept_func():
    states = parse_ts_program(
        """
function flip_case(string: string): string {
    return string.toLocaleLowerCase().replace(/[A-Z]/g, letter => letter.toLocaleUpperCase());     
}
"""
    )
    assert_weak_full(states)


@unittest.skip("TODO")
def test_comment_in_bad_place():
    states = parse_ts_program(
        """
function flip_case(string: string): string {                
    return 2 // will be converted to string
    .toString();
}
"""
    )
    assert_weak_full(states)


@unittest.skip("TODO")
def test_comment_in_bad_place_2():
    states = parse_ts_program(
        """
function next_smallest(lst: number[]): number | undefined {                
    const uniqueLst = [...new Set(lst)               
                       .values()]               
                       .sort((a, b) => a - b);               
    return uniqueLst.length < 2 ? undefined : uniqueLst[1]               
                                   // TypeScript uses undefined"""
    )
    assert_reject(states)


@unittest.skip("TODO")
def test_union_from_ternary():
    states = parse_ts_program(
        """
const a: number | string = true ? 1 : "hello";"""
    )
    assert_weak_full(states)


def test_use_Number_as_mapping_fun():
    states = parse_ts_program(
        """
const digits = ["1", "2"].map(Number);"""
    )
    assert_weak_full(states)


def test_use_Number_as_mapping_fun2():
    states = parse_ts_program(
        """
function digits(n: number): number {                
    let product = 1;    
    for (let digit of n.toString().split('').map(Number)) {   
        if (digit % 2!== 0) {   
            product *= digit;    
        }   
    }   
    return product;    
}    
"""
    )
    assert_weak_full(states)


def test_assignment_is_expression():
    states = parse_ts_program(
        """
for(let i = 0; i < 10; i += 2) {
    let a = i;
    a;
}"""
    )
    assert_weak_full(states)


def test_string_as_str_array():
    states = parse_ts_program(
        """
const s = new Set("hello");
s.add("hi");
"""
    )
    assert_weak_full(states)


def test_short_stmt_no_bracket_if():
    states = parse_ts_program(
        """
if(2 == 3) 4;
"""
    )
    assert_weak_full(states)


def test_short_stmt_no_bracket_ite_return():
    states = parse_ts_program(
        """
function foo(): number {
    if(2 == 3) return 4;
    else return 5;
}
"""
    )
    assert_weak_full(states)


def test_short_stmt_no_bracket_for():
    states = parse_ts_program(
        """
for(let i = 0; i < 10; i++)
   2 + 3;
"""
    )
    assert_weak_full(states)


def test_parameterized_call_Set():
    states = parse_ts_program(
        """
let c = new Set<string>(["h", "i"]);
c.add("h");
"""
    )
    assert_weak_full(states)


def test_parameterized_call_Map():
    states = parse_ts_program(
        """
let c = new Map<string, number>();
c.set("hi", 2);
"""
    )
    assert_weak_full(states)


def test_parameterized_type_set():
    states = parse_ts_program(
        """
let c: Set<string> = new Set(["h", "i"]);
c.add("h");
"""
    )
    assert_weak_full(states)


def test_index_signature_type():
    states = parse_ts_program(
        """
let c : {
        [x: number]: string
} = {1: "hello", 2: "hi" + "me"};
"""
    )
    assert_weak_full(states)


def test_index_signature_type_reject():
    # it is not allowed to use expressions for the literal keys
    assert_just_before_reject(
        """
let c : {
        [x: number]: string
} = {1: "hello", 2+"""
    )


def test_index_signature_access():
    states = parse_ts_program(
        """
let c : {
        [x: string]: string
} = {"hi": "you"};
const y: string = c["hi"];
"""
    )
    assert_weak_full(states)


def test_index_signature_assign():
    states = parse_ts_program(
        """
let c : {
        [x: string]: string
} = {"ho": "you"};
c["hi"] = "what";
"""
    )
    assert_weak_full(states)


def test_array_from_length():
    states = parse_ts_program(
        """
let x = Array.from({length: 10 + 2}, (_, i) => i);
"""
    )
    assert_weak_full(states)


def test_array_from_length_wrong_typ():
    states = parse_ts_program(
        """
let x = Array.from({length: "hi"}, (_, i) => i);
"""
    )
    assert_reject(states)


def test_many_states():
    states = parse_ts_program(
        """
function numerical_letter_grade(grades: number[]): string[] {                
    let letterGrade: string[] = [];               
    for (const gpa of grades) {
        if (gpa === 4.0) {
            letterGrade.push("A+");
        } else if (gpa > 3.7) {
            letterGrade.push("A");
        } else if (gpa > 3.3) {
            letterGrade.push("A-");
        } else if (gpa > 3.0) {
            letterGrade.push("B+");
        } else if (gpa > 2.7) {
            letterGrade.push("B");
        } else if (gpa > 2.3) {"""
    )
    assert_strict_partial(states)
    assert len(states) <= 100, "too many states"


@unittest.skip("not fixed yet")
def test_state_leaking():
    states = parse_ts_program(
        """
function check_dict_case(dict: {[key: string]: string}): boolean {                
    if (Object.keys(dict).length === 0) {                
        return false;                
    } else {               
        let state: string = "start";               
        for (const key of Object.keys(dict)) {               
            if (typeof key !== 'string') {                
                state = "mixed";                
                break;                
            }                
            if (state === "start") {                
                if (key.toUpperCase() === key) {                
                    state = "upper";                
                } else if (key.toLowerCase() === key) {                
                    state = "lower";                
                } else {                
                    break;                
                }                
            } else {                 
                break;                 
            }                 
        }                 
        return state === "upper" || state === "lower";                  
    }                  
} """
    )
    assert_weak_full(states)
    assert len(states) <= 10, "too many states"


@unittest.skip("not fixed yet")
def test_state_leaking2():
    states = parse_ts_program(
        """
let state = "hi";
let key = "you";
(state === "upper" && !key.toUpperCase() === key);"""
    )
    assert assert_weak_full(states)
    assert len(states) <= 10, "too many states"


def test_lambdas_type_args():
    states = parse_ts_program(
        """
let numbers: number[] = [1];
const mean: number = numbers.reduce((acc: number, val, ind, arr) => acc + val, 0);"""
    )
    assert_weak_full(states)


def test_array_from_with_map():
    states = parse_ts_program(
        """
let a: number[] = Array.from(["h", "i"], (x) => 1);
"""
    )
    assert_weak_full(states)


def test_array_from_with_map_wrong():
    states = parse_ts_program(
        """
let a: number[] = Array.from(["h", "i"], (x) => "1");
"""
    )
    assert_reject(states)


def test_array_from_with_map_2():
    states = parse_ts_program(
        """
let a: number[] = Array.from([1, 2], (x) => 1 + x);
"""
    )
    assert_weak_full(states)


def test_typeof():
    states = parse_ts_program(
        """
let a: string = typeof 1;
"""
    )
    assert_weak_full(states)


def test_bigint():
    states = parse_ts_program(
        """
let a: bigint = 10n;
let b: bigint = BigInt("5");
if(a < 5) { - a; }
a + b;"""
    )
    assert_weak_full(states)


def test_bigint_not_number():
    states = parse_ts_program(
        """
let a: bigint = 10;
"""
    )
    assert_reject(states)


def test_ternary_varying_return():
    states = parse_ts_program(
        """
let a: number | string = true ? 1 : "hello";
"""
    )
    assert_weak_full(states)


def test_new_array():
    states = parse_ts_program(
        """
let a = new Array(10);
"""
    )
    assert_weak_full(states)


def test_array_constr():
    states = parse_ts_program(
        """
let a = Array(10);
"""
    )
    assert_weak_full(states)


def test_keyword():
    states = parse_ts_program(
        """
let switch: number = 1;
"""
    )
    assert_reject(states)


def test_logic1():
    states = parse_ts_program(
        """
let a: number = 1 && 2;
"""
    )
    assert_weak_full(states)


def test_logic2():
    states = parse_ts_program(
        """
let a: string = "a" || "b";
"""
    )
    assert_weak_full(states)


def test_logic3():
    states = parse_ts_program(
        """
let a: number = 1 || "";
"""
    )
    assert_reject(states)


def test_logic4():
    states = parse_ts_program(
        """
let a: number | string = 1 || "";
"""
    )
    assert_weak_full(states)


def test_logic5():
    states = parse_ts_program(
        """
let a = 1 || "";
"""
    )
    assert_weak_full(states)


def test_index_sig_end_comma():
    states = parse_ts_program(
        """
const dic: { [key: number]: string } = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
};
"""
    )
    assert_weak_full(states)


def test_set_equality():
    states = parse_ts_program(
        """
Set([1,2,4]).size == Set([1,2]).size;
"""
    )
    assert_weak_full(states)


def test_set_inequality():
    states = parse_ts_program(
        """
Set([1,2,4]).size <= Set([1,2]).size;
"""
    )
    assert_weak_full(states)


def test_equality_overlap():
    states = parse_ts_program(
        """
let a = false ? 1n: "hi";
let b = 1n;
if (b == a) {
  let y = 1;
}
"""
    )
    assert_weak_full(states)


def test_spread_Set():
    states = parse_ts_program(
        """
let x: number[] = [...Set([1,2])];
    """
    )
    assert_weak_full(states)


def test_spread_Object_keys():
    states = parse_ts_program(
        """
let x: string[] = Object.keys([1,2]);
    """
    )
    assert_weak_full(states)


def test_try_catch():
    states = parse_ts_program(
        """
try {
 let x = 1;
} catch (error) {
  // bla
}
"""
    )
    assert_weak_full(states)


def test_union_array_number_type():
    states = parse_ts_program(
        """
let x: string[] | number = 1;
"""
    )
    assert_weak_full(states)


def test_Number_type():
    states = parse_ts_program(
        """
let x: boolean = Number.isFinite("hi");
"""
    )
    assert_weak_full(states)


def test_allow_any_type_written():
    states = parse_ts_program(
        """
function filter_integers(values: any[]): number[] {
  return values.filter((value) => Number.isInteger(value)).map((value) => Number.parseInt(value.toString()));
}"""
    )
    assert_weak_full(states)


def test_Set_from_string_expression():
    states = parse_ts_program(
        """
function count_distinct_characters(string: string): number {
    return new Set(string.toLowerCase()).size;
}"""
    )
    assert_weak_full(states)


def test_unassigned_declaration():
    states = parse_ts_program(
        """
let x: number;
x = 5;
"""
    )
    assert_weak_full(states)


def test_index_sig_literal_no_context():
    assert_just_before_reject(
        """
  const numberToName = {"""
    )


def test_crypto_objects():
    states = parse_ts_program(
        """
let text = "hi";
// Create a new instance of the Crypto library     
const crypto = require('crypto');     
 
// Create a hash object using the MD5 algorithm     
const hash = crypto.createHash('md5');     
 
// Update the hash object with the input text     
hash.update(text);     
 
// Get the hex digest of the hash     
const hexDigest: string = hash.digest('hex');     
"""
    )
    assert_weak_full(states)


def test_Array_fill():
    states = parse_ts_program(
        """
let x: number[] = Array(10).fill(0);
"""
    )
    assert_weak_full(states)


def test_lambda_with_type_annotation():
    states = parse_ts_program(
        """
const isPrime = (num: number): boolean => {
    return true;
};"""
    )
    assert_weak_full(states)


def test_map2():
    states = parse_ts_program(
        """
function by_length(arr: number[]): string[] {
    const result = arr.map(num => {
        return "hi";
    })"""
    )
    states = incremental_parse(
        states,
        """;
     return result;    
}    
""",
    )
    assert_weak_full(states)


def test_map3():
    states = parse_ts_program(
        """
function by_length(arr: number[]): string[] {
    const result = arr.map(num => "hi") """
    )
    states = incremental_parse(
        states,
        """;
     return result;    
}    
""",
    )
    assert_weak_full(states)


def test_map4():
    states = parse_ts_program(
        """
let x:string[] = [1].map(s => {   
        return "hello";
    });"""
    )
    assert_weak_full(states)


def test_map5():
    states = parse_ts_program(
        """
let x:string[] = [1].map(s => {
        let count = 0;
        for (let c of s.toString()) {
            if (parseInt(c) % 2!== 0) {
                count++;
            }
        }
        return `the number of odd elements ${count}n the str${s} of the ${s}nput.`;
    });"""
    )
    assert_weak_full(states)


def test_anonymous_function():
    states = parse_ts_program(
        """
let text = "hello";
let result = text.replace(/\s+/g, function(match) {   
    if (match.length > 2) {   
        return '-';
    } else {   
        return '_';   
    }   
});"""
    )
    assert_weak_full(states)


def test_anonymous_function_2():
    # the function potentially returns void type which does not match the required type
    states = parse_ts_program(
        """
let text = "hello";
let result = text.replace(/\s+/g, function(match) {   
    if (match.length > 2) {   
        return '-';
    }   
});"""
    )
    assert_reject(states)


def test_tuple_assignment():
    states = parse_ts_program(
        """
        let a = 2;
        let b = 3;
        [a, b] = [1, 2];
        """
    )
    assert_weak_full(states)


def test_single_tuple_assignment():
    states = parse_ts_program(
        """
        let b = 3;
        [b] = [2];
        """
    )
    assert_weak_full(states)


def test_single_tuple_assignment_wrong():
    states = parse_ts_program(
        """
        let b = 3;
        [b] = ["2"];
        """
    )
    assert_reject(states)


def test_disallow_multiline_string():
    states = parse_ts_program(
        """
let x = " 
let"""
    )
    assert_reject(states)


def test_comparison_completable():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num: number = 2;
num >"""
    )
    states = incremental_parse(states, "largestNegative")
    assert_reject(states)


def test_comparison_completable_2():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num: number = 2;
largestNegative """
    )
    states = incremental_parse(states, "<")
    assert_reject(states)


def test_number_undefined_comparison():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num = 2; 
largestNegative === undefined || num > largestNegative;
"""
    )
    assert_weak_full(states)


def test_number_undefined_comparison_2():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num = 2; 
largestNegative !== undefined && num > largestNegative;
"""
    )
    assert_weak_full(states)


def test_number_undefined_comparison_3():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num = 2; 
largestNegative === undefined && num > largestNegative;
"""
    )
    assert_reject(states)


def test_number_undefined_comparison_4():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num = 2; 
largestNegative !== undefined || num > largestNegative;
"""
    )
    assert_reject(states)


def test_number_undefined_ite():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;
let num = 2; 
if(largestNegative === undefined){
   num += 1;
} else {
   largestNegative += 1;
}
"""
    )
    assert_weak_full(states)


def test_number_undefined_ite_2():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num = 2; 
if(largestNegative !== undefined){
   largestNegative += 1;
} else {
   num += 1;
}
"""
    )
    assert_weak_full(states)


@unittest.skip("Currently disallowed but in principle ok (with ASI)")
def test_comments_inside_expressions():
    states = parse_ts_program(
        """
function count_nums(arr: number[]): number {
  return arr.reduce((count: number, num) => {               
    const digits = num < 0               
      // If the number is"""
    )
    assert_strict_partial(states)


def test_set_array_type_push():
    states = parse_ts_program(
        """
let x = [];
x.push(1);
x[0] > 2;
"""
    )
    assert_weak_full(states)


def test_set_array_type_assign():
    states = parse_ts_program(
        """
const fib = new Array(2);     
fib[0] = 0;     
fib[1] = 0;   
"""
    )
    assert_weak_full(states)


def test_no_crash_2():
    states = parse_ts_program("""
function has_close_elements(numbers: number[], threshold: number): boolean {
  // Sort the array in ascending order               
  numbers.sort();               

  // Iterate over the array               
  for (let i = 0; i 
""")
    incremental_parse(states, "< numbers")


def test_no_crash_3():
    states = parse_ts_program("""
let x = "
""")
    assert_reject(states)


def test_reject_numeric_literal_member_access():
    # TODO can we test for early rejection, i.e. that access to fields is not even attempted in type search?
    states = parse_ts_program("""
let x = 0.toString();""")
    assert_reject(states)


def test_accept_map_lambda_fun():
    states = parse_ts_program("""
function odd_count(lst: string[]): string[] {
  return lst.map((str, index) => """)
    states = incremental_parse(states, "{")
    assert_strict_partial(states)


def test_fix_push_type_propagation():
    states = parse_ts_program("""
function foo(x: number[]): number[] {
  const result = [];
  while(x[0] == 0){
    if (true){
        result.push(x.shift());
    } else {
        result.push(0);
    }
  }
  return result;
}
""")
    assert_reject(states)


def test_assign_const_array():
    states = parse_ts_program("""
let n = 0;
let arr = [1,2,3];
const counter = new Array(n + 1).fill(0);
arr.forEach(num => counter[num]++);
""")
    assert_weak_full(states)


def test_throw_error():
    states = parse_ts_program(
        """
throw new Error("Array must contain at least two elements."); 
""",
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_logical_or_undefined_union():
    states = parse_ts_program("""
let x: number | undefined = 2;
let y: number = x || 0;
""")
    assert_weak_full(states)


def test_unary_not_plus_num():
    assert_just_before_reject("""let x = !2 + 1;""")


def test_number_undefined_comparison_5():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num = 2; 
!largestNegative || num > largestNegative;
"""
    )
    assert_weak_full(states)


def test_number_undefined_comparison_6():
    states = parse_ts_program(
        """
let largestNegative: number | undefined = undefined;     
let num = 2; 
largestNegative && num > largestNegative;
"""
    )
    assert_weak_full(states)


def test_disallowed_bool_plus_op():
    states = parse_ts_program(
        """
const lastCharNotPartOfWord = false + 1;
"""
    )
    assert_reject(states)


def test_disallowed_bool_plus_op_2():
    states = parse_ts_program(
        """
const txt = "hi";
const lastChar = txt.charAt(txt.length - 1);     
 
const lastCharNotPartOfWord = !txt.lastIndexOf(lastChar) + 1 === txt.length;    
"""
    )
    assert_reject(states)


def test_no_crash_10():
    states = parse_ts_program("""
function max_fill(grid: number[][], capacity: number): number {
  const m = grid.length;
  const n = grid[0].length;
  let counter = 0;
  let filled_grid = [];

  // Initialize the filled grid
  for (let i = 0; i < m; i++) {
    filled_grid""")
    states = incremental_parse(states, "\n")


def test_disallow_double_decrement():
    states = parse_ts_program("""
const n = 1;
let i = 1;
n--i--;""")
    assert_reject(states)


@unittest.skip("currently not supported")
def test_create_typed_set():
    states = parse_ts_program(
        """
function common(l1: number[], l2: number[]): number[] {
  const uniqueCommon: Set<number> = new Set();""",
        print_failure_point=True,
    )
    assert_partial(states)


def test_reduce_different_type():
    states = parse_ts_program(
        """
let numbers: string[] = ["1"];
const mean: number = numbers.reduce((acc: number, val, ind, arr) => acc + parseInt(val), 0);"""
    )
    assert_weak_full(states)


def test_no_crash_100():
    parse_ts_program("""
function make_a_pile(n: number): number[] {
  // Function to check if a number is even or odd
  function isEven(num: number): boolean {
    return num % 2 === 0;
  }

  // Iterate from 1 to n (inclusive) and build the pile
  const pile: number[] = new Array(n).fill(0).map((_, i) => {
    const numStones = i + n;
    // Use number and its parity to determine the belonging to a line
    let belongingLine = (i + 1) % 2 === 0? "even" : "odd";
    // Add number of stones based on parity
    return belongingLine """)


def test_accept_forEach():
    states = parse_ts_program("""
const testCases = ["hi", "you"];
testCases.forEach(x => {
        x.length;
});
""")
    assert_weak_full(states)


def test_use_scoped_var_outside():
    states = parse_ts_program("""
if(1 == 5){
  let x = 1;
}
x += 2;
""")
    assert_reject(states)


def test_use_scoped_var_outside_2():
    states = parse_ts_program("""
for(let x of [1,2,3]){
  x += 2;
}
x += 2;
""")
    assert_reject(states)


def test_disallow_octal():
    states = parse_ts_program(
        (
            """
let x:number = 022;
"""
        )
    )
    assert_reject(states)


@unittest.skip("not implemented yet")
def test_nullish_coersion():
    states = parse_ts_program(
        (
            """
let y: number | undefined = 0;
let x:number = y ?? 2;
"""
        ),
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_disallow_qmdot():
    states = parse_ts_program(
        (
            """
let y: number | undefined = 0;
let x:number = y?.1 : 10;
"""
        ),
    )
    assert_reject(states)


def test_fromCharCode():
    states = parse_ts_program(
        (
            """
String.fromCharCode(189, 43, 190, 61);
"""
        ),
    )
    assert_weak_full(states)


def test_optional_chaining_non_call():
    states = parse_ts_program(
        (
            """
let y: number[] | undefined = [0];
let x:number  | undefined = y?.length;
"""
        ),
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_optional_chaining():
    states = parse_ts_program(
        (
            """
let y: number | undefined = 0;
let x:string  | undefined = y?.toString();
"""
        ),
        print_failure_point=True,
    )
    assert_weak_full(states)


def test_optional_chaining_wrong_res_type():
    states = parse_ts_program(
        (
            """
let y: number | undefined = 0;
let x:string = y?.toString();
"""
        ),
        print_failure_point=True,
    )
    assert_reject(states)


def test_optional_chaining_no_undefined():
    # In theory this is ok, but most likely an error - only allow if the type is potentially undefined
    states = parse_ts_program(
        (
            """
let y: number = 0;
let x:string = y?.toString();
"""
        ),
        print_failure_point=True,
    )
    assert_reject(states)


def test_lambda_in_if():
    states = parse_ts_program(
        """
        if ((a: number): number => a) {}
        """
    )
    assert_reject(states)


def test_recursive_lambda_expr():
    states = parse_ts_program(
        """
        const gcd = (a: number, b: number): number => gcd(a, b);
        """
    )
    assert_weak_full(states)


def test_recursive_lambda_func():
    states = parse_ts_program(
        """
        const gcd = (a: number, b: number): number => {
            return gcd(a, b);
        };
        """
    )
    assert_weak_full(states)


def test_forof_tuple1():
    states = parse_ts_program(
        """
        const map = new Map<number, string>([[1000, 'M']]);
        for (let [key, val] of map.entries()) { key = 1; }
        """
    )
    assert_weak_full(states)


def test_forof_tuple2():
    states = parse_ts_program(
        """
        const map = new Map<number, string>([[1000, 'M']]);
        for (const [key, val] of map.entries()) { let a = 1; }
        """
    )
    assert_weak_full(states)


def test_forof_tuple3():
    states = parse_ts_program(
        """
        const map = new Map<number, string>([[1000, 'M']]);
        for (const [key, val] of map.entries()) { key = 1; }
        """
    )
    assert_reject(states)


def test_reject_ternary_early():
    states = parse_ts_program(
        """
const res: number = true ? [1] :""",
        print_failure_point=True,
    )
    assert_reject(states)


def test_reject_ternary_on_truthy():
    # tests both whether truthy values can have ternary op and optional chaining
    states = parse_ts_program(
        """
const res: number = "".toString ?""",
        print_failure_point=True,
    )
    assert_reject(states)


def test_tuple_and_array1():
    states = parse_ts_program(
        """
function sort_numbers(numbers: string): string {
  const numberMap = new Map<string, number>([
    ['zero', 0],
    ['one', 1],
  ]);
        """
    )
    assert_partial(states)


def test_tuple_and_array2():
    states = parse_ts_program(
        """
function sort_numbers(numbers: string): string {
  const numberMap = new Map<string, number>([
    ['zero', 0],
    ['one', 1],
  ]);

  return numbers.split(' ')
    .map(number => [number, numberMap.get(number)!])
    .sort((a, b) => a[1] - b[1])
    .map(pair => pair[0])
    .join(' ');
}
        """
    )
    assert_reject(states)


def test_no_newline_in_return():
    code = """
function check_if_last_char_is_a_letter(txt: string): boolean {                
  if (txt.length === 0) {     
    return false;     
  }     
  const lastChar = txt.slice(-1);     
  const previousChar = txt.slice(-2, -1);     
       
  return      
    lastChar.match(/[a-zA-Z]/) &&      
    (previousChar === ' ' || previousChar === undefined);     
}"""
    states = parse_ts_program(code)
    assert_reject(states)


def test_tuple_lambda():
    code = """
function sort_numbers(numbers: string): string {
  const numberMap = new Map<string, number>([
    ["zero", 0],
    ["one", 1],
    ["two", 2],
  ]);

  const numberStrings = numbers.split(" ");
  const sortedNumbers = numberStrings
    .map(number => numberMap.get(number)!)
    .sort((a, b) => a - b)
    .map(number => numberStrings[numberStrings.indexOf(Object.entries(numberMap).find(([key, value]) => value === number)![0])]);

  return sortedNumbers.join(" ");
}
"""
    states = parse_ts_program(code)
    assert_weak_full(states)


def test_non_break_whitespace():
    parser_state = IncrementalParserState.from_state(
        TerminalParserState(target_value="hi\byou")
    )
    for code in (
        "hi you",
        "hi    you",
    ):
        assert_weak_full(incremental_parse(parser_state, code))
    for code in (
        "hi\nyou",
        "hi \n you",
        "hi\n you",
        "hi\r\nyou",
    ):
        assert_reject(incremental_parse(parser_state, code))


def test_asi_2():
    code = """
function iscube(a: number): boolean {                
  if (a < 0) {     
    return false     
  }     
      
  return Math.cbrt(a) % 1 === 0;     
}
"""
    assert_weak_full(parse_ts_program(code))


def test_float_parser():
    code = ".;"
    states = parse_ts_program(code)
    assert_reject(states)


def test_set_weirdness():
    code = """function unique_product(list_data: number[]): number {
  const temp = [... new Set("""
    states = parse_ts_program(code)
    code2 = "list_data"
    states = incremental_parse(states, code2)
    assert_partial(states)
