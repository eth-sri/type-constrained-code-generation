from collections import defaultdict
from dataclasses import field
from typing import List, Dict, Self, Set, FrozenSet, Tuple, Literal, Optional

from functools import lru_cache

from .types_base import PType, AnyPType, OperatorPrecedence
from .util import fnr_dataclass, union_dict


@fnr_dataclass
class BaseTsObject(PType):
    """
    Base class for all TypeScript objects, synonymous for "any"
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
        Get all attributes of the object
        """
        return {
            "hasOwnProperty": FunctionPType([StringPType()], BooleanPType()),
            "toString": FunctionPType([], StringPType()),
            "toLocaleString": FunctionPType([], StringPType()),
            "valueOf": FunctionPType([], self),
        }

    @property
    def attributes(self) -> Dict[str, Tuple[Self, bool]]:
        return {
            k: (v, isinstance(v, FunctionPType)) for k, v in self._attributes.items()
        }

    @property
    def nesting_depth(self) -> Tuple[int, int]:
        return (0, 0)

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return isinstance(other, BaseTsObject)

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(self.__class__)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "any"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class TypeParameterPType(AnyPType, BaseTsObject):
    """
    Type parameter type
    """

    name: str

    def __str__(self):
        return self.name

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return type_params.get(self.name, self)

    def type_params(self) -> Set[str]:
        return {self.name}


@fnr_dataclass
class FunctionPType(BaseTsObject):
    """
    Function type
    """

    call_signature: List[PType]
    return_type: PType
    # How many of the arguments are optional (counted from the end)
    optional_args: int = 0
    # do we allow the last arg to be repeated infinitely?
    # Note: to allow 0 or more repetitions, set optional args to 1 - i.e. the last element is optional but can be repeated infinitely
    inf_args: bool = False
    # Whether this function is a constructor (i.e. can be called with new)
    is_constructor: bool = False
    # Whether this function should always be called with new
    # TODO this is not enforced yet
    force_new: bool = False

    def __post_init__(self):
        object.__setattr__(self, "call_signature", tuple(self.call_signature))

    @property
    def nesting_depth(self):
        return_type_nesting_depth_fun = self.return_type.nesting_depth
        return (return_type_nesting_depth_fun[0], return_type_nesting_depth_fun[1] + 1)

    @property
    def root_values(self):
        return self.return_type.root_values

    def __ge__(self, other):
        if not isinstance(other, FunctionPType):
            return False
        if len(self.call_signature) - self.optional_args > len(other.call_signature):
            return False
        for self_arg, other_arg in zip(self.call_signature, other.call_signature):
            if not other_arg >= self_arg:
                return False
        if isinstance(self.return_type, VoidPType):
            return True
        return self.return_type >= other.return_type

    def __eq__(self, other):
        if not isinstance(other, FunctionPType):
            return False
        if len(self.call_signature) != len(other.call_signature):
            return False
        for self_arg, other_arg in zip(self.call_signature, other.call_signature):
            if not other_arg == self_arg:
                return False
        return self.return_type == other.return_type

    def __hash__(self):
        return hash((tuple(self.call_signature), self.return_type))

    def __str__(self):
        len_call_signature = len(self.call_signature)
        return (
            (f"<{','.join(self.type_params())}>" if self.type_params() else "")
            + f"({','.join(f'v{i}: {x}' + ('?' if len_call_signature - i <= self.optional_args else '') for i, x in enumerate(self.call_signature))}) => {self.return_type}"
        )

    def instantiate_type_params(self, type_params: Dict[str, PType]) -> Self:
        return FunctionPType(
            [x.instantiate_type_params(type_params) for x in self.call_signature],
            self.return_type.instantiate_type_params(type_params),
            self.optional_args,
        )

    def type_params(self) -> Set[str]:
        return (
            set().union(*(x.type_params() for x in self.call_signature))
            | self.return_type.type_params()
        )


@fnr_dataclass
class PrimitivePType(BaseTsObject):
    """
    Primitive type
    """

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return issubclass(other.__class__, self.__class__)

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class StringPType(PrimitivePType):
    """
    String type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
                String
        Constructor

            String() constructor

        Static methods

            String.fromCharCode()
            String.fromCodePoint()
            String.raw()

        Instance methods

            String.prototype.anchor() Deprecated
            String.prototype.at()
            String.prototype.big() Deprecated
            String.prototype.blink() Deprecated
            String.prototype.bold() Deprecated
            String.prototype.charAt()
            String.prototype.charCodeAt()
            String.prototype.codePointAt()
            String.prototype.concat()
            String.prototype.endsWith()
            String.prototype.fixed() Deprecated
            String.prototype.fontcolor() Deprecated
            String.prototype.fontsize() Deprecated
            String.prototype.includes()
            String.prototype.indexOf()
            String.prototype.isWellFormed()
            String.prototype.italics() Deprecated
            String.prototype.lastIndexOf()
            String.prototype.link() Deprecated
            String.prototype.localeCompare()
            String.prototype.match()
            String.prototype.matchAll()
            String.prototype.normalize()
            String.prototype.padEnd()
            String.prototype.padStart()
            String.prototype.repeat()
            String.prototype.replace()
            String.prototype.replaceAll()
            String.prototype.search()
            String.prototype.slice()
            String.prototype.small() Deprecated
            String.prototype.split()
            String.prototype.startsWith()
            String.prototype.strike() Deprecated
            String.prototype.sub() Deprecated
            String.prototype.substr() Deprecated
            String.prototype.substring()
            String.prototype.sup() Deprecated
            String.prototype[Symbol.iterator]()
            String.prototype.toLocaleLowerCase()
            String.prototype.toLocaleUpperCase()
            String.prototype.toLowerCase()
            String.prototype.toString()
            String.prototype.toUpperCase()
            String.prototype.toWellFormed()
            String.prototype.trim()
            String.prototype.trimEnd()
            String.prototype.trimStart()
            String.prototype.valueOf()

        Instance properties

            String: length
        """
        res = {
            "fromCharCode": FunctionPType(
                [NumberPType()],
                self,
                1,
                inf_args=True,
            ),
            "fromCodePoint": FunctionPType(
                [NumberPType()],
                self,
                1,
                inf_args=True,
            ),
            "at": FunctionPType([NumberPType()], self),
            "charAt": FunctionPType([NumberPType()], self),
            "charCodeAt": FunctionPType([NumberPType()], NumberPType()),
            "codePointAt": FunctionPType([NumberPType()], NumberPType()),
            "concat": FunctionPType([self], self, 1, inf_args=True),
            "endsWith": FunctionPType([self, NumberPType()], BooleanPType(), 1),
            "includes": FunctionPType([self, NumberPType()], BooleanPType(), 1),
            "indexOf": FunctionPType([self, NumberPType()], NumberPType(), 1),
            "isWellFormed": FunctionPType([], BooleanPType()),
            "lastIndexOf": FunctionPType([self, NumberPType()], NumberPType(), 1),
            "localeCompare": FunctionPType([self, StringPType()], NumberPType(), 1),
            # TODO return type is special (if g flag not used...) - lets see if we need to model this
            "match": FunctionPType(
                [RegExpPType()],
                ArrayPType(StringPType()),
                1,
            ),
            "matchAll": FunctionPType([RegExpPType()], ArrayPType(ArrayPType(self)), 1),
            "normalize": FunctionPType([StringPType()], self, 1),
            "padEnd": FunctionPType([NumberPType(), self], self, 1),
            "padStart": FunctionPType([NumberPType(), self], self, 1),
            "repeat": FunctionPType(
                [NumberPType()],
                self,
            ),
            "replace": FunctionPType(
                [
                    UnionPType([StringPType(), RegExpPType()]),
                    UnionPType(
                        [StringPType(), FunctionPType([StringPType()], StringPType())]
                    ),
                ],
                self,
            ),
            "replaceAll": FunctionPType(
                [
                    UnionPType([StringPType(), RegExpPType()]),
                    UnionPType(
                        [StringPType(), FunctionPType([StringPType()], StringPType())]
                    ),
                ],
                self,
            ),
            "search": FunctionPType(
                [UnionPType([StringPType(), RegExpPType()])],
                NumberPType(),
            ),
            "slice": FunctionPType(
                [
                    NumberPType(),
                    NumberPType(),
                ],
                self,
                1,
            ),
            "split": FunctionPType(
                [
                    UnionPType([StringPType(), RegExpPType()]),
                    NumberPType(),
                ],
                ArrayPType(self),
                1,
            ),
            "startsWith": FunctionPType(
                [
                    StringPType(),
                    NumberPType(),
                ],
                BooleanPType(),
                1,
            ),
            "substring": FunctionPType(
                [
                    NumberPType(),
                    NumberPType(),
                ],
                self,
                1,
            ),
            "toLocaleLowerCase": FunctionPType(
                [ArrayPType(StringPType())],
                self,
                1,
            ),
            "toLocaleUpperCase": FunctionPType(
                [ArrayPType(StringPType())],
                self,
                1,
            ),
            "toLowerCase": FunctionPType(
                [],
                self,
            ),
            "toUpperCase": FunctionPType(
                [],
                self,
            ),
            "toWellFormed": FunctionPType([], self),
            "trim": FunctionPType(
                [],
                self,
            ),
            "trimEnd": FunctionPType(
                [],
                self,
            ),
            "trimRight": FunctionPType(
                [],
                self,
            ),
            "trimStart": FunctionPType(
                [],
                self,
            ),
            "trimLeft": FunctionPType(
                [],
                self,
            ),
            "length": NumberPType(),
        }
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "string"


@fnr_dataclass
class NumberPType(PrimitivePType):
    """
    Number type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = super()._attributes
        res.update(
            {
                # "EPSILON": NumberPType(),
                # "MAX_SAFE_INTEGER": NumberPType(),
                # "MAX_VALUE": NumberPType(),
                # "MIN_SAFE_INTEGER": NumberPType(),
                # "MIN_VALUE": NumberPType(),
                # "NEGATIVE_INFINITY": NumberPType(),
                # "NaN": NumberPType(),
                # "POSITIVE_INFINITY": NumberPType(),
                # "isFinite": FunctionPType([], BooleanPType()),
                # "isInteger": FunctionPType([], BooleanPType()),
                # "isNaN": FunctionPType([], BooleanPType()),
                # "isSafeInteger": FunctionPType([], BooleanPType()),
                # "parseFloat": FunctionPType([StringPType()], NumberPType(), 0),
                # "parseInt": FunctionPType(
                #     [StringPType(), NumberPType()], NumberPType(), 1
                # ),
                "toExponential": FunctionPType([NumberPType()], StringPType(), 1),
                "toFixed": FunctionPType([NumberPType()], StringPType(), 1),
                "toLocaleString": FunctionPType([StringPType()], StringPType(), 1),
                "toPrecision": FunctionPType([NumberPType()], StringPType(), 1),
                "toString": FunctionPType([NumberPType()], StringPType(), 1),
            }
        )
        return res

    def __str__(self):
        return "number"


@fnr_dataclass
class BooleanPType(PrimitivePType):
    """
    Boolean type
    """

    def __str__(self):
        return "boolean"


@fnr_dataclass
class VoidPType(PrimitivePType):
    """
    Void type
    """

    def __str__(self):
        return "void"

    @property
    def _attributes(self) -> Dict[str, Self]:
        return {}


@fnr_dataclass
class GenericPType(BaseTsObject):
    """
    Special type that signifies a generic type
    i.e. ArrayPType(GenericPType()) is an array that fits into any array
    NOTE: this may only be used for
    - empty arrays
    - searching for any array type (no matter which)
    """

    def __str__(self):
        return "T"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


# @fnr_dataclass
# class NeverPType(BaseTsObject):
#     def __str__(self):
#         return "never"
#
#     def __ge__(self, other):
#         return False


@fnr_dataclass
class NullPType(PrimitivePType):
    def __str__(self):
        return "null"

    @property
    def _attributes(self) -> Dict[str, Self]:
        return {}


@fnr_dataclass
class UndefinedPType(PrimitivePType):
    def __str__(self):
        return "undefined"

    @property
    def _attributes(self) -> Dict[str, Self]:
        return {}


@fnr_dataclass
class ReduceFunctionPType(FunctionPType):
    pass


@fnr_dataclass
class ArrayPType(BaseTsObject):
    """
    Array type
    """

    element_type: PType

    @property
    def _attributes(self) -> Dict[str, Self]:
        res_no_elements = {
            "length": NumberPType(),
            "concat": FunctionPType([self], self, 1, inf_args=True),
            "copyWithin": FunctionPType(
                [NumberPType(), NumberPType(), NumberPType()], self, 1
            ),
            "fill": FunctionPType(
                [TypeParameterPType("T"), NumberPType(), NumberPType()],
                ArrayPType(TypeParameterPType("T")),
                2,
            ),
            "join": FunctionPType([StringPType()], StringPType(), 1),
            "reverse": FunctionPType([], self),
            "slice": FunctionPType([NumberPType(), NumberPType()], self, 2),
            "push": FunctionPType([AnyPType()], NumberPType(), 1, inf_args=True),
            "toReversed": FunctionPType([], self),
        }
        res_elements = {
            "at": FunctionPType([NumberPType()], self.element_type),
            "every": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BaseTsObject(), 2
                    ),
                    self,
                ],
                BooleanPType(),
                1,
            ),
            "fill": FunctionPType(
                [self.element_type, NumberPType(), NumberPType()], self, 2
            ),
            "filter": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BaseTsObject(), 2
                    ),
                    self,
                ],
                self,
                1,
            ),
            "find": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BaseTsObject(), 2
                    ),
                    self,
                ],
                self.element_type,
                1,
            ),
            "findIndex": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BaseTsObject(), 2
                    ),
                    self,
                ],
                NumberPType(),
                1,
            ),
            "findLast": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BaseTsObject(), 2
                    ),
                    self,
                ],
                NumberPType(),
                1,
            ),
            "findLastIndex": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BaseTsObject(), 2
                    ),
                    self,
                ],
                NumberPType(),
                1,
            ),
            "forEach": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BaseTsObject(), 2
                    ),
                    self,
                ],
                VoidPType(),
                1,
            ),
            "includes": FunctionPType(
                [self.element_type, NumberPType()], BooleanPType(), 1
            ),
            "indexOf": FunctionPType(
                [self.element_type, NumberPType()], NumberPType(), 1
            ),
            "lastIndexOf": FunctionPType(
                [self.element_type, NumberPType()], NumberPType(), 1
            ),
            "map": FunctionPType(
                # this function allows mapping any array to an array of a return type for any function in scope
                # that matches the signature
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self],
                        TypeParameterPType("T"),
                        2,
                    ),
                    self,
                ],
                ArrayPType(TypeParameterPType("T")),
                1,
            ),
            "pop": FunctionPType(
                [],
                self.element_type,
            ),
            "push": FunctionPType([self.element_type], NumberPType(), 1, inf_args=True),
            "reduce": UnionPType(
                [
                    FunctionPType(
                        [
                            ReduceFunctionPType(
                                [
                                    self.element_type,
                                    self.element_type,
                                    NumberPType(),
                                    self,
                                ],
                                self.element_type,
                                2,
                            ),
                            self.element_type,
                        ],
                        self.element_type,
                        1,
                    ),
                    FunctionPType(
                        [
                            ReduceFunctionPType(
                                [
                                    NumberPType(),
                                    StringPType(),
                                    NumberPType(),
                                    self,
                                ],
                                NumberPType(),
                                2,
                            ),
                            NumberPType(),
                        ],
                        NumberPType(),
                    ),
                    FunctionPType(
                        [
                            ReduceFunctionPType(
                                [
                                    ArrayPType(NumberPType()),
                                    NumberPType(),
                                    NumberPType(),
                                    self,
                                ],
                                ArrayPType(NumberPType()),
                                2,
                            ),
                            ArrayPType(NumberPType()),
                        ],
                        ArrayPType(NumberPType()),
                    ),
                ]
            ),
            "reduceRight":  # UnionPType(
            # [
            # FunctionPType(
            #     [
            #         FunctionPType(
            #             [
            #                 TypeParameterPType("T"),
            #                 self.element_type,
            #                 NumberPType(),
            #                 self,
            #             ],
            #             TypeParameterPType("T"),
            #             2,
            #         ),
            #         TypeParameterPType("T"),
            #     ],
            #     TypeParameterPType("T"),
            # ),
            FunctionPType(
                [
                    FunctionPType(
                        [
                            self.element_type,
                            self.element_type,
                            NumberPType(),
                            self,
                        ],
                        self.element_type,
                        2,
                    ),
                ],
                self.element_type,
            ),
            # ],
            # ),
            "shift": FunctionPType(
                [],
                self.element_type,
            ),
            "some": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, NumberPType(), self], BooleanPType(), 2
                    ),
                    self,
                ],
                BooleanPType(),
                1,
            ),
            "sort": FunctionPType(
                [FunctionPType([self.element_type, self.element_type], NumberPType())],
                self,
                1,
            ),
            "splice": FunctionPType(
                [NumberPType(), NumberPType(), self.element_type],
                self,
                2,
                inf_args=True,
            ),
            "toSorted": FunctionPType(
                [FunctionPType([self.element_type, self.element_type], NumberPType())],
                self,
                1,
            ),
            "toSpliced": FunctionPType(
                [NumberPType(), NumberPType(), self.element_type],
                self,
                1,
                inf_args=True,
            ),
            "unshift": FunctionPType(
                [self.element_type],
                NumberPType(),
                1,
                inf_args=True,
            ),
            "with": FunctionPType([NumberPType(), self.element_type], self),
        }
        res_no_elements.update(super()._attributes)
        if not isinstance(self.element_type, GenericPType):
            res_no_elements.update(res_elements)
        return res_no_elements

    @property
    def attributes(self) -> Dict[str, Tuple[Self, bool]]:
        sup = super().attributes
        sup["length"] = (NumberPType(), True)
        return sup

    @property
    def nesting_depth(self):
        (
            element_nesting_depth_array,
            element_nesting_depth_fun,
        ) = (
            self.element_type.nesting_depth
            if not isinstance(self.element_type, GenericPType)
            else (0, 0)
        )
        return (
            element_nesting_depth_array + 1,
            element_nesting_depth_fun,
        )

    def __ge__(self, other):
        if not isinstance(other, ArrayPType):
            return False
        return (
            self.element_type == other.element_type
            or (
                self.element_type == GenericPType()
                or other.element_type == GenericPType()
            )
            or self.element_type >= other.element_type
        )

    @property
    def root_values(self) -> Set[Self]:
        return (
            self.element_type.root_values
            if self.element_type != GenericPType()
            else {AnyPType()}
        )

    def __eq__(self, other):
        if not isinstance(other, ArrayPType):
            return False
        return self.element_type == other.element_type

    def __hash__(self):
        return hash((self.element_type, self.__class__))

    def __str__(self):
        return f"{self.element_type}[]"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return ArrayPType(self.element_type.instantiate_type_params(type_params))

    def type_params(self) -> Set[str]:
        return self.element_type.type_params()


@fnr_dataclass
class MathPType(BaseTsObject):
    """
    Type of the global object Math
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
        from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math

        Math.E

            Euler's number and the base of natural logarithms; approximately 2.718.
        Math.LN10

            Natural logarithm of 10; approximately 2.303.
        Math.LN2

            Natural logarithm of 2; approximately 0.693.
        Math.LOG10E

            Base-10 logarithm of E; approximately 0.434.
        Math.LOG2E

            Base-2 logarithm of E; approximately 1.443.
        Math.PI

            Ratio of a circle's circumference to its diameter; approximately 3.14159.
        Math.SQRT1_2

            Square root of ½; approximately 0.707.
        Math.SQRT2

            Square root of 2; approximately 1.414.
        Math[Symbol.toStringTag]

            The initial value of the [Symbol.toStringTag] property is the string "Math". This property is used in Object.prototype.toString().

        Static methods

        Math.abs()

            Returns the absolute value of the input.
        Math.acos()

            Returns the arccosine of the input.
        Math.acosh()

            Returns the hyperbolic arccosine of the input.
        Math.asin()

            Returns the arcsine of the input.
        Math.asinh()

            Returns the hyperbolic arcsine of a number.
        Math.atan()

            Returns the arctangent of the input.
        Math.atan2()

            Returns the arctangent of the quotient of its arguments.
        Math.atanh()

            Returns the hyperbolic arctangent of the input.
        Math.cbrt()

            Returns the cube root of the input.
        Math.ceil()

            Returns the smallest integer greater than or equal to the input.
        Math.clz32()

            Returns the number of leading zero bits of the 32-bit integer input.
        Math.cos()

            Returns the cosine of the input.
        Math.cosh()

            Returns the hyperbolic cosine of the input.
        Math.exp()

            Returns ex, where x is the argument, and e is Euler's number (2.718…, the base of the natural logarithm).
        Math.expm1()

            Returns subtracting 1 from exp(x).
        Math.floor()

            Returns the largest integer less than or equal to the input.
        Math.f16round()

            Returns the nearest half precision float representation of the input.
        Math.fround()

            Returns the nearest single precision float representation of the input.
        Math.hypot()

            Returns the square root of the sum of squares of its arguments.
        Math.imul()

            Returns the result of the 32-bit integer multiplication of the inputs.
        Math.log()

            Returns the natural logarithm (㏒e; also, ㏑) of the input.
        Math.log10()

            Returns the base-10 logarithm of the input.
        Math.log1p()

            Returns the natural logarithm (㏒e; also ㏑) of 1 + x for the number x.
        Math.log2()

            Returns the base-2 logarithm of the input.
        Math.max()

            Returns the largest of zero or more numbers.
        Math.min()

            Returns the smallest of zero or more numbers.
        Math.pow()

            Returns base x to the exponent power y (that is, xy).
        Math.random()

            Returns a pseudo-random number between 0 and 1.
        Math.round()

            Returns the value of the input rounded to the nearest integer.
        Math.sign()

            Returns the sign of the input, indicating whether it is positive, negative, or zero.
        Math.sin()

            Returns the sine of the input.
        Math.sinh()

            Returns the hyperbolic sine of the input.
        Math.sqrt()

            Returns the positive square root of the input.
        Math.tan()

            Returns the tangent of the input.
        Math.tanh()

            Returns the hyperbolic tangent of the input.
        Math.trunc()

            Returns the integer portion of the input, removing any fractional digits.
        """
        res = {
            "E": NumberPType(),
            "LN10": NumberPType(),
            "LN2": NumberPType(),
            "LOG10E": NumberPType(),
            "LOG2E": NumberPType(),
            "PI": NumberPType(),
            "SQRT1_2": NumberPType(),
            "SQRT2": NumberPType(),
            "abs": FunctionPType([NumberPType()], NumberPType()),
            "acos": FunctionPType([NumberPType()], NumberPType()),
            "acosh": FunctionPType([NumberPType()], NumberPType()),
            "asin": FunctionPType([NumberPType()], NumberPType()),
            "asinh": FunctionPType([NumberPType()], NumberPType()),
            "atan": FunctionPType([NumberPType()], NumberPType()),
            "atan2": FunctionPType([NumberPType(), NumberPType()], NumberPType()),
            "atanh": FunctionPType([NumberPType()], NumberPType()),
            "cbrt": FunctionPType([NumberPType()], NumberPType()),
            "ceil": FunctionPType([NumberPType()], NumberPType()),
            "clz32": FunctionPType([NumberPType()], NumberPType()),
            "cos": FunctionPType([NumberPType()], NumberPType()),
            "cosh": FunctionPType([NumberPType()], NumberPType()),
            "exp": FunctionPType([NumberPType()], NumberPType()),
            "expm1": FunctionPType([NumberPType()], NumberPType()),
            "floor": FunctionPType([NumberPType()], NumberPType()),
            "f16round": FunctionPType([NumberPType()], NumberPType()),
            "fround": FunctionPType([NumberPType()], NumberPType()),
            "hypot": FunctionPType(
                [NumberPType()],
                NumberPType(),
                1,
                inf_args=True,
            ),
            "imul": FunctionPType([NumberPType(), NumberPType()], NumberPType()),
            "log": FunctionPType([NumberPType()], NumberPType()),
            "log10": FunctionPType([NumberPType()], NumberPType()),
            "log1p": FunctionPType([NumberPType()], NumberPType()),
            "log2": FunctionPType([NumberPType()], NumberPType()),
            "max": FunctionPType(
                [NumberPType()],
                NumberPType(),
                1,
                inf_args=True,
            ),
            "min": FunctionPType(
                [NumberPType()],
                NumberPType(),
                1,
                inf_args=True,
            ),
            "pow": FunctionPType([NumberPType(), NumberPType()], NumberPType()),
            "random": FunctionPType([], NumberPType()),
            "round": FunctionPType([NumberPType()], NumberPType()),
            "sign": FunctionPType([NumberPType()], NumberPType()),
            "sin": FunctionPType([NumberPType()], NumberPType()),
            "sinh": FunctionPType([NumberPType()], NumberPType()),
            "sqrt": FunctionPType([NumberPType()], NumberPType()),
            "tan": FunctionPType([NumberPType()], NumberPType()),
            "tanh": FunctionPType([NumberPType()], NumberPType()),
            "trunc": FunctionPType([NumberPType()], NumberPType()),
        }
        res.update(super()._attributes)
        return res

    @property
    def nesting_depth(self):
        return (0, 0)

    def __ge__(self, other):
        return isinstance(other, MathPType)

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __eq__(self, other):
        return isinstance(other, MathPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "Math"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class ObjectPType(BaseTsObject):
    """
    Type of the global object Object
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
                from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object

                    Static methods

        Object.assign()

            Copies the values of all enumerable own properties from one or more source objects to a target object.
        Object.create()

            Creates a new object with the specified prototype object and properties.
        Object.defineProperties()

            Adds the named properties described by the given descriptors to an object.
        Object.defineProperty()

            Adds the named property described by a given descriptor to an object.
        Object.entries()

            Returns an array containing all of the [key, value] pairs of a given object's own enumerable string properties.
        Object.freeze()

            Freezes an object. Other code cannot delete or change its properties.
        Object.fromEntries()

            Returns a new object from an iterable of [key, value] pairs. (This is the reverse of Object.entries).
        Object.getOwnPropertyDescriptor()

            Returns a property descriptor for a named property on an object.
        Object.getOwnPropertyDescriptors()

            Returns an object containing all own property descriptors for an object.
        Object.getOwnPropertyNames()

            Returns an array containing the names of all of the given object's own enumerable and non-enumerable properties.
        Object.getOwnPropertySymbols()

            Returns an array of all symbol properties found directly upon a given object.
        Object.getPrototypeOf()

            Returns the prototype (internal [[Prototype]] property) of the specified object.
        Object.groupBy()

            Groups the elements of a given iterable according to the string values returned by a provided callback function. The returned object has separate properties for each group, containing arrays with the elements in the group.
        Object.hasOwn()

            Returns true if the specified object has the indicated property as its own property, or false if the property is inherited or does not exist.
        Object.is()

            Compares if two values are the same value. Equates all NaN values (which differs from both IsLooselyEqual used by == and IsStrictlyEqual used by ===).
        Object.isExtensible()

            Determines if extending of an object is allowed.
        Object.isFrozen()

            Determines if an object was frozen.
        Object.isSealed()

            Determines if an object is sealed.
        Object.keys()

            Returns an array containing the names of all of the given object's own enumerable string properties.
        Object.preventExtensions()

            Prevents any extensions of an object.
        Object.seal()

            Prevents other code from deleting properties of an object.
        Object.setPrototypeOf()

            Sets the object's prototype (its internal [[Prototype]] property).
        Object.values()

            Returns an array containing the values that correspond to all of a given object's own enumerable string properties.

        """
        res = {
            "freeze": FunctionPType([TypeParameterPType("T")], TypeParameterPType("T")),
            "hasOwn": FunctionPType([AnyPType(), StringPType()], BooleanPType()),
            "is": FunctionPType(
                [TypeParameterPType("T"), TypeParameterPType("T")], BooleanPType()
            ),
            "isExtensible": FunctionPType([AnyPType()], BooleanPType()),
            "isFrozen": FunctionPType([AnyPType()], BooleanPType()),
            "isSealed": FunctionPType([AnyPType()], BooleanPType()),
            "keys": FunctionPType([AnyPType()], ArrayPType(StringPType())),
            "preventExtensions": FunctionPType(
                [TypeParameterPType("T")], TypeParameterPType("T")
            ),
            "seal": FunctionPType([TypeParameterPType("T")], TypeParameterPType("T")),
            "entries": FunctionPType(
                [MapPType(key_type=StringPType(), value_type=NumberPType())],
                ArrayPType(TuplePType(types=[StringPType(), NumberPType()])),
            ),
            # TODO if used likely need to make more precise
            # "values": FunctionPType([AnyPType()], ArrayPType(BaseTsObject())),
        }
        res.update(super()._attributes)
        return res

    @property
    def nesting_depth(self):
        return (0, 0)

    def __ge__(self, other):
        return isinstance(other, MathPType)

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __eq__(self, other):
        return isinstance(other, MathPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "Math"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class UnionPType(BaseTsObject):
    """
    Union type
    """

    types: FrozenSet[PType]

    def __post_init__(self):
        object.__setattr__(self, "types", frozenset(self.types))

    @property
    def nesting_depth(self):
        if len(self.types) == 0:
            return (0, 0)
        else:
            return (
                max(x.nesting_depth[0] for x in self.types),
                max(x.nesting_depth[1] for x in self.types),
            )

    @property
    def attributes(self) -> Dict[str, Tuple[Self, bool]]:
        res = intersection_attribute_union_dict(*(x.attributes for x in self.types))
        if "valueOf" in res:
            res["valueOf"] = (FunctionPType([], self), True)
        return res

    @property
    def root_values(self) -> Set[Self]:
        return set().union(*(x.root_values for x in self.types))

    def __ge__(self, other):
        if not isinstance(other, UnionPType):
            other_types = {other}
        else:
            other_types = other.types
        return all(any(x >= y for x in self.types) for y in other_types)

    def __eq__(self, other):
        if not isinstance(other, UnionPType):
            return False
        return set(self.types) == set(other.types)

    def __hash__(self):
        return hash((tuple(self.types), self.__class__))

    def __str__(self):
        return f'({" | ".join(sorted(str(x) for x in self.types))})'

    def type_params(self) -> Set[str]:
        return set().union(*(x.type_params() for x in self.types))

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return merge_typs(*(x.instantiate_type_params(type_params) for x in self.types))


@fnr_dataclass
class OverlapsWith(BaseTsObject):
    """
    Type that expresses that it overlaps with a given type
    """

    type: PType

    @property
    def nesting_depth(self):
        return self.type.nesting_depth

    @property
    def attributes(self) -> Dict[str, Tuple[Self, bool]]:
        return super().attributes

    @property
    def root_values(self) -> Set[Self]:
        return self.type.root_values

    def __ge__(self, other):
        if not isinstance(other, UnionPType):
            other_types = {other}
        else:
            other_types = other.types
        return any(self.type >= y for y in other_types)

    def __eq__(self, other):
        if not isinstance(other, OverlapsWith):
            return False
        return self.type == other.type

    def __hash__(self):
        return hash((self.type, self.__class__))

    def __str__(self):
        return f"OverlapsWith({self.type})"


@fnr_dataclass
class TuplePType(BaseTsObject):
    """
    Tuple type
    """

    types: List[PType]

    @property
    def nesting_depth(self):
        if len(self.types) == 0:
            return (1, 0)
        else:
            return (
                max(x.nesting_depth[0] for x in self.types) + 1,
                max(x.nesting_depth[1] for x in self.types),
            )

    @property
    def _attributes(self) -> Dict[str, Self]:
        return dict()

    @property
    def root_values(self) -> Set[Self]:
        return set().union(*(x.root_values for x in self.types))

    def __ge__(self, other):
        if isinstance(other, TuplePType):
            if len(self.types) != len(other.types):
                return False
            return all(self.types[i] >= other.types[i] for i in range(len(self.types)))

        if isinstance(other, AbsTuplePType):
            if len(self.types) < len(other.types):
                return False
            for t, o_t in zip(self.types, other.types):
                if isinstance(o_t, AnyPType):
                    continue
                if not (t >= o_t):
                    return False
            return True

        return False

    def __eq__(self, other):
        if not isinstance(other, TuplePType):
            return False
        return self.types == other.types

    def __hash__(self):
        return hash((tuple(self.types), self.__class__))

    def __str__(self):
        return "[" + ", ".join(str(t) for t in self.types) + "]"

    def type_params(self) -> Set[str]:
        return set().union(*(t.type_params() for t in self.types))

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return TuplePType([t.instantiate_type_params(type_params) for t in self.types])


@fnr_dataclass
class AbsTuplePType(BaseTsObject):
    """
    Tuple type
    """

    types: List[PType]

    @property
    def nesting_depth(self):
        if len(self.types) == 0:
            return (1, 0)
        else:
            return (
                max(x.nesting_depth[0] for x in self.types) + 1,
                max(x.nesting_depth[1] for x in self.types),
            )

    @property
    def _attributes(self) -> Dict[str, Self]:
        return dict()

    @property
    def root_values(self) -> Set[Self]:
        return set().union(*(x.root_values for x in self.types)) | {AnyPType()}

    def __ge__(self, other):
        if isinstance(other, TuplePType):
            if len(self.types) > len(other.types):
                return False
            for t, o_t in zip(self.types, other.types):
                if not (t >= o_t):
                    return False
            return True

        if isinstance(other, AbsTuplePType):
            for t, o_t in zip(self.types, other.types):
                if not (t >= o_t):
                    return False
            return True
            pass

        return False

    def __eq__(self, other):
        if not isinstance(other, AbsTuplePType):
            return False
        return self.types == other.types

    def __hash__(self):
        return hash((tuple(self.types), self.__class__))

    def __str__(self):
        return "[" + ", ".join(list(map(lambda t: str(t), self.types)) + ["..."]) + "]"


@fnr_dataclass
class LengthPType(BaseTsObject):
    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def _attributes(self) -> Dict[str, BaseTsObject]:
        own = {
            "length": NumberPType(),
        }
        own.update(super()._attributes)
        return own

    @property
    def root_values(self) -> Set[BaseTsObject]:
        return {self}

    def __ge__(self, other):
        return isinstance(other, LengthPType)

    def __eq__(self, other):
        return isinstance(other, LengthPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "{length : number }"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class AbsArrayPType(FunctionPType):
    """
    Array static type, doubles as a constructor
    """

    call_signature: List[PType] = field(default_factory=lambda: [NumberPType()])
    return_type: PType = ArrayPType(GenericPType())
    optional_args: int = 0
    inf_args: bool = False
    is_constructor: bool = True
    force_new: bool = False

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def _attributes(self) -> Dict[str, Self]:
        return {
            "from": UnionPType(
                [
                    FunctionPType(
                        [ArrayPType(TypeParameterPType("T"))],
                        ArrayPType(TypeParameterPType("T")),
                    ),
                    FunctionPType([StringPType()], ArrayPType(StringPType())),
                    FunctionPType(
                        [
                            LengthPType(),
                            FunctionPType(
                                [AnyPType(), NumberPType()],
                                TypeParameterPType("T"),
                                optional_args=1,
                            ),
                        ],
                        ArrayPType(TypeParameterPType("T")),
                    ),
                    FunctionPType(
                        [
                            ArrayPType(TypeParameterPType("T")),
                            FunctionPType(
                                [TypeParameterPType("T"), NumberPType()],
                                TypeParameterPType("S"),
                                optional_args=1,
                            ),
                        ],
                        ArrayPType(TypeParameterPType("S")),
                    ),
                ]
            ),
            "isArray": FunctionPType([AnyPType()], BooleanPType()),
            # "of": FunctionPType(
            #     [TypeParameterPType("T")] * (INF_ELEMS_UNROLL + 1),
            #     ArrayPType(TypeParameterPType("T")),
            #     INF_ELEMS_UNROLL,
            # ),
        }

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, AbsArrayPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "Array"


@fnr_dataclass
class AbsNumberPType(FunctionPType):
    """
    Number static type, doubles as a constructor
    """

    call_signature: List[PType] = field(default_factory=lambda: [AnyPType()])
    return_type: PType = NumberPType()
    optional_args: int = 0
    inf_args: bool = False
    is_constructor: bool = True
    force_new: bool = False

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def _attributes(self) -> Dict[str, Self]:
        """

        Number.EPSILON

            The smallest interval between two representable numbers.
        Number.MAX_SAFE_INTEGER

            The maximum safe integer in JavaScript (253 - 1).
        Number.MAX_VALUE

            The largest positive representable number.
        Number.MIN_SAFE_INTEGER

            The minimum safe integer in JavaScript (-(253 - 1)).
        Number.MIN_VALUE

            The smallest positive representable number—that is, the positive number closest to zero (without actually being zero).
        Number.NaN

            Special "Not a Number" value.
        Number.NEGATIVE_INFINITY

            Special value representing negative infinity. Returned on overflow.
        Number.POSITIVE_INFINITY

            Special value representing infinity. Returned on overflow.

        Static methods

        Number.isFinite()

            Determine whether the passed value is a finite number.
        Number.isInteger()

            Determine whether the passed value is an integer.
        Number.isNaN()

            Determine whether the passed value is NaN.
        Number.isSafeInteger()

            Determine whether the passed value is a safe integer (number between -(253 - 1) and 253 - 1).
        Number.parseFloat()

            This is the same as the global parseFloat() function.
        Number.parseInt()

            This is the same as the global parseInt() function.

        """
        return {
            "EPSILON": NumberPType(),
            "MAX_SAFE_INTEGER": NumberPType(),
            "MAX_VALUE": NumberPType(),
            "MIN_SAFE_INTEGER": NumberPType(),
            "MIN_VALUE": NumberPType(),
            "NaN": NumberPType(),
            "NEGATIVE_INFINITY": NumberPType(),
            "POSITIVE_INFINITY": NumberPType(),
            "isFinite": FunctionPType([AnyPType()], BooleanPType()),
            "isInteger": FunctionPType([AnyPType()], BooleanPType()),
            "isNaN": FunctionPType([AnyPType()], BooleanPType()),
            "isSafeInteger": FunctionPType([AnyPType()], BooleanPType()),
            "parseFloat": FunctionPType([StringPType()], NumberPType()),
            "parseInt": FunctionPType([StringPType(), NumberPType()], NumberPType(), 1),
        }

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, AbsArrayPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "Number"


@fnr_dataclass
class AbsStringPType(FunctionPType):
    """
    String static type, doubles as a constructor
    """

    call_signature: List[PType] = field(default_factory=lambda: [AnyPType()])
    return_type: PType = StringPType()
    optional_args: int = 0
    inf_args: bool = False
    is_constructor: bool = True
    force_new: bool = False

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def _attributes(self) -> Dict[str, Self]:
        """

        String.fromCharCode()

            Returns a string created by using the specified sequence of Unicode values.
        String.fromCodePoint()

            Returns a string created by using the specified sequence of code points.
        String.raw()

            Returns a string created from a raw template string.

        """
        return {
            "fromCharCode": FunctionPType(
                [NumberPType()],
                StringPType(),
                optional_args=1,
                inf_args=True,
            ),
            "fromCodePoint": FunctionPType(
                [NumberPType()],
                StringPType(),
                optional_args=1,
                inf_args=True,
            ),
        }

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, AbsArrayPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "Number"


@fnr_dataclass
class CryptoPType(BaseTsObject):
    """
    crypto static type, acquired by running "require('crypto')"
    """

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
        NOTE: so far we only model md5, which is required by humaneval. other classes are trivial to add
        """
        return {
            "createHash": FunctionPType([StringPType()], HashPType(), 1),
        }

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "crypto"


@fnr_dataclass
class HashPType(BaseTsObject):
    """
    crypto static type, acquired by running "require('crypto')"
    """

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
        NOTE: so far we only model md5, which is required by humaneval. other classes are trivial to add
        """
        return {
            "update": FunctionPType([StringPType()], HashPType(), 1),
            "digest": FunctionPType([StringPType()], StringPType(), 1),
        }

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "Hash"


@fnr_dataclass
class SetPType(BaseTsObject):
    """
    Set type
    """

    element_type: PType

    @property
    def _attributes(self) -> Dict[str, Self]:
        res_no_elements = {
            "size": NumberPType(),
            "clear": FunctionPType([], VoidPType()),
        }
        res_elements = {
            "add": FunctionPType([self.element_type], self),
            "delete": FunctionPType([self.element_type], BooleanPType()),
            "difference": FunctionPType([self], self),
            # This would lead to infinitely more tuple types.. lets try to avoid it
            # "entries": FunctionPType([], ArrayPType(TuplePType([self.element_type, self.element_type]))),
            "forEach": FunctionPType(
                [
                    FunctionPType(
                        [self.element_type, self.element_type, self], BaseTsObject(), 2
                    )
                ],
                VoidPType(),
            ),
            "has": FunctionPType([self.element_type], BooleanPType()),
            "intersection": FunctionPType([self], self),
            "isDisjointFrom": FunctionPType([self], BooleanPType()),
            "isSubsetOf": FunctionPType([self], BooleanPType()),
            "isSupersetOf": FunctionPType([self], BooleanPType()),
            "keys": FunctionPType([], ArrayPType(self.element_type)),
            "symmetricDifference": FunctionPType([self], self),
            "union": FunctionPType([self], self),
            "values": FunctionPType([], ArrayPType(self.element_type)),
        }
        res_no_elements.update(super()._attributes)
        if not isinstance(self.element_type, GenericPType):
            res_no_elements.update(res_elements)
        return res_no_elements

    @property
    def nesting_depth(self):
        (
            element_nesting_depth_array,
            element_nesting_depth_fun,
        ) = (
            self.element_type.nesting_depth
            if not isinstance(self.element_type, GenericPType)
            else (0, 0)
        )
        return (
            element_nesting_depth_array,
            element_nesting_depth_fun,
        )

    def __ge__(self, other):
        if not isinstance(other, SetPType):
            return False
        return self.element_type == other.element_type or (
            self.element_type == GenericPType() or other.element_type == GenericPType()
        )

    @property
    def root_values(self) -> Set[Self]:
        return (
            self.element_type.root_values
            if self.element_type != GenericPType()
            else {AnyPType()}
        )

    def __eq__(self, other):
        if not isinstance(other, SetPType):
            return False
        return self.element_type == other.element_type

    def __hash__(self):
        return hash((self.element_type, self.__class__))

    def __str__(self):
        return f"Set<{self.element_type}>"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return SetPType(self.element_type.instantiate_type_params(type_params))

    def type_params(self) -> Set[str]:
        return self.element_type.type_params()


@fnr_dataclass
class MapPType(BaseTsObject):
    """
    Map type
    """

    key_type: PType
    value_type: PType

    @property
    def _attributes(self) -> Dict[str, Self]:
        res_no_elements = {
            "size": NumberPType(),
            "clear": FunctionPType([], VoidPType()),
        }
        res_elements = {
            "delete": FunctionPType([self.key_type], BooleanPType()),
            # This would lead to infinitely more tuple types.. lets try to avoid it
            "entries": FunctionPType(
                [], ArrayPType(TuplePType([self.key_type, self.value_type]))
            ),
            "forEach": FunctionPType(
                [
                    FunctionPType(
                        [self.value_type, self.key_type, self], BaseTsObject(), 2
                    )
                ],
                VoidPType(),
            ),
            "get": FunctionPType(
                [self.key_type],
                self.value_type,
            ),
            "has": FunctionPType([self.key_type], BooleanPType()),
            "keys": FunctionPType([], ArrayPType(self.key_type)),
            "set": FunctionPType([self.key_type, self.value_type], self),
            "values": FunctionPType([], ArrayPType(self.value_type)),
        }
        res_no_elements.update(super()._attributes)
        if not isinstance(self.key_type, GenericPType) and not isinstance(
            self.value_type, GenericPType
        ):
            res_no_elements.update(res_elements)
        return res_no_elements

    @property
    def nesting_depth(self):
        (
            element_nesting_depth_array,
            element_nesting_depth_fun,
        ) = (
            (
                max(self.key_type.nesting_depth[0], self.value_type.nesting_depth[0]),
                max(self.key_type.nesting_depth[1], self.value_type.nesting_depth[1]),
            )
            if not isinstance(self.key_type, GenericPType)
            and not isinstance(self.value_type, GenericPType)
            else (0, 0)
        )
        return (
            element_nesting_depth_array,
            element_nesting_depth_fun,
        )

    def __ge__(self, other):
        if not isinstance(other, MapPType):
            return False
        return (
            self.key_type == other.key_type
            and self.value_type == other.value_type
            or (
                self.key_type == GenericPType()
                or other.key_type == GenericPType()
                or self.value_type == GenericPType()
                or other.value_type == GenericPType()
            )
        )

    @property
    def root_values(self) -> Set[Self]:
        return (
            self.key_type.root_values
            if self.key_type != GenericPType()
            else {AnyPType()}
        ) | (
            self.value_type.root_values
            if self.value_type != GenericPType()
            else {AnyPType()}
        )

    def __eq__(self, other):
        if not isinstance(other, MapPType):
            return False
        return self.key_type == other.key_type and self.value_type == other.value_type

    def __hash__(self):
        return hash((self.key_type, self.value_type, self.__class__))

    def __str__(self):
        return f"Map<{self.key_type},{self.value_type}>"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return MapPType(
            self.key_type.instantiate_type_params(type_params),
            self.value_type.instantiate_type_params(type_params),
        )

    def type_params(self) -> Set[str]:
        return self.key_type.type_params() | self.value_type.type_params()


@fnr_dataclass
class IndexSignaturePType(BaseTsObject):
    """
    Type of objects described by { [key: string]: value }
    """

    key_type: PType
    value_type: PType

    @property
    def nesting_depth(self):
        (
            element_nesting_depth_array,
            element_nesting_depth_fun,
        ) = (
            (
                max(self.key_type.nesting_depth[0], self.value_type.nesting_depth[0]),
                max(self.key_type.nesting_depth[1], self.value_type.nesting_depth[1]),
            )
            if not isinstance(self.key_type, GenericPType)
            and not isinstance(self.value_type, GenericPType)
            else (0, 0)
        )
        return (
            element_nesting_depth_array,
            element_nesting_depth_fun,
        )

    def __ge__(self, other):
        if not isinstance(other, IndexSignaturePType):
            return False
        return self.key_type == other.key_type and self.value_type == other.value_type

    @property
    def root_values(self) -> Set[Self]:
        return self.value_type.root_values

    def __eq__(self, other):
        if not isinstance(other, IndexSignaturePType):
            return False
        return self.value_type == other.value_type

    def __hash__(self):
        return hash((self.key_type, self.value_type, self.__class__))

    def __str__(self):
        return f"{{[index: {self.key_type}]:{self.value_type}}}"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return IndexSignaturePType(
            self.key_type.instantiate_type_params(type_params),
            self.value_type.instantiate_type_params(type_params),
        )

    def type_params(self) -> Set[str]:
        return self.key_type.type_params() | self.value_type.type_params()


@fnr_dataclass
class RegExpPType(BaseTsObject):
    """
    A regular expression type
    """

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
        RegExp.prototype.dotAll
        RegExp.prototype.flags
        RegExp.prototype.global
        RegExp.prototype.hasIndices
        RegExp.prototype.ignoreCase
        RegExp: lastIndex
        RegExp.prototype.multiline
        RegExp.prototype.source
        RegExp.prototype.sticky
        RegExp.prototype.unicode
        RegExp.prototype.unicodeSets
        """
        return {
            # TODO the return type is actually special, lets see if we need to model this
            "exec": FunctionPType(
                [StringPType()], UnionPType([ArrayPType(StringPType()), NullPType()])
            ),
            "test": FunctionPType([StringPType()], BooleanPType()),
            "dotAll": BooleanPType(),
            "flags": StringPType(),
            "global": BooleanPType(),
            "hasIndices": BooleanPType(),
            "ignoreCase": BooleanPType(),
            "lastIndex": NumberPType(),
            "multiline": BooleanPType(),
            "source": StringPType(),
            "sticky": BooleanPType(),
            "unicode": BooleanPType(),
            "unicodeSets": BooleanPType(),
        }

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return isinstance(other, RegExpPType)

    def __eq__(self, other):
        return isinstance(other, RegExpPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "RegExp"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class BigIntPType(BaseTsObject):
    """
    A big int type
    """

    @property
    def nesting_depth(self):
        return (0, 0)

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return isinstance(other, BigIntPType)

    def __eq__(self, other):
        return isinstance(other, BigIntPType)

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "bigint"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


def any_bigger(xs: List[int], ys: List[int]):
    return any(x > y for x, y in zip(xs, ys))


OPERATOR_PRECEDENCES = {
    "p.x": 17,
    "p[]": 17,
    "p()": 17,
    "p++": 15,
    # "++p": 14,
    # "!p": 14, are prepended, not relevant
    # "~p": 14, are prepended, not relevant
    # "+p": 14, are prepended, not relevant
    "**": 13,
    "*": 12,
    "/": 12,
    "%": 12,
    "+": 11,
    "-": 11,
    "<<": 10,
    "<<<": 10,
    ">>": 10,
    ">>>": 10,
    "<": 9,
    "<=": 9,
    ">": 9,
    ">=": 9,
    "==": 8,
    "!=": 8,
    "===": 8,
    "!==": 8,
    "&": 7,
    "^": 6,
    "|": 5,
    "&&": 4,
    "||": 3,
    "?": 2,
}
OPERATOR_ASSOCIATIVITY = {
    "p.x": "left",
    "p[]": "left",
    "p()": "left",
    "p++": "right",
    "++p": "right",
    "!p": "right",
    "~p": "right",
    "+p": "right",
    "**": "right",
    "*": "left",
    "/": "left",
    "%": "left",
    "+": "left",
    "-": "left",
    "<<": "left",
    ">>": "left",
    ">>>": "left",
    "<<<": "left",
    "<": "left",
    "<=": "left",
    ">": "left",
    ">=": "left",
    "==": "left",
    "!=": "left",
    "===": "left",
    "!==": "left",
    "&": "left",
    "^": "left",
    "|": "left",
    "&&": "left",
    "||": "left",
    "?": "right",
}
OPERATOR_SORTED_BY_PRECEDENCE = list(
    sorted(
        OPERATOR_PRECEDENCES.keys(), key=lambda x: OPERATOR_PRECEDENCES[x], reverse=True
    )
)


def member_access_reachable_types(typ: PType):
    """
    Get the reachable types from a given type
    by applying member access
    :param typ: type to explore
    """
    if not isinstance(typ, AnyPType):
        return set(x[0] for x in typ.attributes.values())
    return set()


def computed_member_access_reachable_types(typ: PType):
    """
    Get the reachable types from a given type
    by applying computed member access
    :param typ: type to explore
    """
    if isinstance(typ, ArrayPType) and not isinstance(typ.element_type, GenericPType):
        return {typ.element_type}
    if isinstance(typ, TuplePType) and typ.types:
        return set(typ.types)
    elif isinstance(typ, StringPType):
        return {StringPType()}
    if isinstance(typ, IndexSignaturePType):
        return {typ.value_type}
    elif isinstance(typ, UnionPType) and typ.types:
        return set.intersection(
            *(computed_member_access_reachable_types(x) for x in typ.types)
        )
    return set()


# this function covers addition which is special: it can cast anything to string (when one operand is string) and is otherwise exclusive to number/bigint
# further all that can be reached by other operator is included in addition
def addition_reachable_types(typ: PType):
    if isinstance(typ, NumberPType):
        return {NumberPType(), StringPType()}
    if isinstance(typ, BigIntPType):
        return {BigIntPType(), StringPType()}
    return {StringPType()}


def call_reachable_types(typ: PType):
    if isinstance(typ, FunctionPType):
        return {typ.return_type}
    return {}


# what about string comparison? --> part of equality. this is for <= etc
def relational_comparison_reachable_types(typ: PType):
    if isinstance(typ, NumberPType):
        return {BooleanPType()}
    return {}


def equality_reachable_types(typ: PType):
    return {BooleanPType()}


def logical_and_reachable_types(typ: PType):
    # these are values that are potentially false, they can be overwritten by the rhs
    if any(
        isinstance(typ, S)
        for S in (UnionPType, NumberPType, StringPType, BooleanPType, ArrayPType)
    ):
        return {AnyPType()}
    # otherwise we obtain nothing interesting and reject this completion
    return {}


def ternary_reachable_types(typ: PType):
    # these are values that are potentially false, they can be overwritten by the rhs
    if any(
        isinstance(typ, S)
        for S in (UnionPType, NumberPType, StringPType, BooleanPType, ArrayPType)
    ):
        return {AnyPType()}
    # otherwise we obtain nothing interesting (always the lhs) -> forbid using ternary
    return {}


OPERATOR_REACHABLE_TYPE_MAP = {
    "p.x": member_access_reachable_types,
    "p[]": computed_member_access_reachable_types,
    "p()": call_reachable_types,
    "+": addition_reachable_types,
    # enough to have one of the comparison operators
    "<": relational_comparison_reachable_types,
    # enough to have one of the equality operators
    "==": equality_reachable_types,
    # enough to have one of the logical operators
    "&&": logical_and_reachable_types,
    "?": ternary_reachable_types,
}

MAX_OPERATOR_PRECEDENCE: OperatorPrecedence = (
    max(OPERATOR_PRECEDENCES.values()) + 1,
    "right",
)
MIN_OPERATOR_PRECEDENCE: OperatorPrecedence = (
    min(OPERATOR_PRECEDENCES.values()) - 1,
    "left",
)

ReachableState = Literal["REACHABLE", "NOT_REACHABLE", "UNKNOWN"]
GLOBAL_REACHABLE_CACHE: Dict[..., ReachableState] = {}


def _reachable_bfs(
    t: PType,
    goal_t: PType,
    min_operator_prec: OperatorPrecedence,
    max_operator_prec: OperatorPrecedence,
    in_array: List[OperatorPrecedence],
    in_nested_expression: List[OperatorPrecedence],
    in_pattern: List[Tuple[PType, PType, OperatorPrecedence]],
    max_depth=(0, 0),
    root_types: FrozenSet[PType] = frozenset(),
    max_steps=20,
) -> ReachableState:
    # track which results become reachable once a certain state was reached
    preds_map = defaultdict(set)
    init_key_t = (
        t,
        goal_t,
        min_operator_prec,
        max_operator_prec,
        in_array,
        in_nested_expression,
        in_pattern,
        max_depth,
        root_types,
        max_steps,
    )
    # track which sates need to be explored next
    visited = set()
    queue = [init_key_t]

    def exec_reached(init_key_t, res):
        if res != "REACHABLE":
            return
        all_preds = {init_key_t}
        new_preds = [init_key_t]
        while new_preds:
            key_t = new_preds.pop()
            new_preds.extend(set(preds_map[key_t]).difference(all_preds))
            all_preds.update(preds_map[key_t])
        for pred in all_preds:
            GLOBAL_REACHABLE_CACHE[pred] = "REACHABLE"

    def insert(key_t, new_key_t):
        preds_map[new_key_t].add(key_t)
        if new_key_t in visited or new_key_t in queue:
            return
        queue.append(new_key_t)

    i = 0
    while queue and i < 1000:
        i += 1
        key_t = queue.pop(0)
        (
            t,
            goal_t,
            min_operator_prec,
            max_operator_prec,
            in_array,
            in_nested_expression,
            in_pattern,
            max_depth,
            root_types,
            max_steps,
        ) = key_t

        if key_t in GLOBAL_REACHABLE_CACHE:
            exec_reached(key_t, GLOBAL_REACHABLE_CACHE[key_t])
            return GLOBAL_REACHABLE_CACHE[key_t]
        visited.add(key_t)
        if in_array:
            # if in_array we have to reach the goal type from the array version of the expression
            insert(
                key_t,
                (
                    ArrayPType(t),
                    goal_t,
                    in_array[0],
                    MAX_OPERATOR_PRECEDENCE,
                    in_array[1:],
                    in_nested_expression,
                    in_pattern,
                    max_depth,
                    root_types,
                    max_steps,
                ),
            )
        elif in_nested_expression:
            # if in_nested_expression we have to reach the goal type from the nested expression version of the expression - i.e. we get to override the operator precedence once
            insert(
                key_t,
                (
                    t,
                    goal_t,
                    in_nested_expression[0],
                    MAX_OPERATOR_PRECEDENCE,
                    in_array,
                    in_nested_expression[1:],
                    in_pattern,
                    max_depth,
                    root_types,
                    max_steps,
                ),
            )
        elif in_pattern:
            # if in pattern we have to reach the goal type from the pattern where
            # the parameter type is replaced by the current type
            matches, type_params = extract_type_params(in_pattern[0][0], t)
            if matches:
                insert(
                    key_t,
                    (
                        in_pattern[0][1].instantiate_type_params(type_params),
                        goal_t,
                        in_pattern[0][2],
                        max_operator_prec,
                        in_array,
                        in_nested_expression,
                        in_pattern[1:],
                        max_depth,
                        root_types,
                        max_steps,
                    ),
                )
            # if t == AnyPType():
            #     exec_reached(key_t, "REACHABLE")
            #     GLOBAL_REACHABLE_CACHE[key_t] = "REACHABLE"
            #     return "REACHABLE"
        else:
            # if the goal type is already reached, we are done
            if goal_t >= t:
                exec_reached(key_t, "REACHABLE")
                return "REACHABLE"
            if t == AnyPType():
                # we can reach all types from AnyPType
                exec_reached(key_t, "REACHABLE")
                return "REACHABLE"

        root_types = frozenset(t.root_values | root_types)
        allowed_ops = OPERATOR_SORTED_BY_PRECEDENCE
        if max_operator_prec[1] == "right":
            # only strictly weaker operations leave the tree
            allowed_ops = [
                op
                for op in allowed_ops
                if OPERATOR_PRECEDENCES[op] < max_operator_prec[0]
            ]
        else:
            # weaker or equal operations leave the tree
            allowed_ops = [
                op
                for op in allowed_ops
                if OPERATOR_PRECEDENCES[op] <= max_operator_prec[0]
            ]
        if min_operator_prec[1] == "left":
            # only strictly stronger operations stay in the tree
            allowed_ops = [
                op
                for op in allowed_ops
                if OPERATOR_PRECEDENCES[op] > min_operator_prec[0]
            ]
        else:
            # stronger or equal operations stay in the tree
            allowed_ops = [
                op
                for op in allowed_ops
                if OPERATOR_PRECEDENCES[op] >= min_operator_prec[0]
            ]
        reachable_typs = set(
            (typ, op)
            for op in allowed_ops
            if op in OPERATOR_REACHABLE_TYPE_MAP
            for typ in OPERATOR_REACHABLE_TYPE_MAP[op](t)
            # keep only types that either introduce new root values or do not unnecessarily expand the nesting depth
            if (
                typ.nesting_depth[0] <= max_depth[0]
                and typ.nesting_depth[1] <= max_depth[1]
            )
            or (typ.root_values - root_types)
        )
        # can we reach the goal type from any of the reachable types?
        # we have to stay within the current tree but can easily exit the subtree, hence the precedence stays the same
        if max_steps > 0:
            for rt_op in reachable_typs:
                for rt, op in [rt_op]:
                    insert(
                        key_t,
                        (
                            rt,
                            goal_t,
                            min_operator_prec,
                            (OPERATOR_PRECEDENCES[op], OPERATOR_ASSOCIATIVITY[op]),
                            in_array,
                            in_nested_expression,
                            in_pattern,
                            (
                                max(rt.nesting_depth[0], max_depth[0]),
                                max(rt.nesting_depth[1], max_depth[1]),
                            ),
                            root_types,
                            max_steps - 1,
                        ),
                    )
            # prevent recursing indefinitely!
            if min_operator_prec <= (17, "left") <= max_operator_prec and (
                not in_pattern
                or not in_pattern[0][1]
                == UnionPType([TypeParameterPType("T"), UndefinedPType()])
            ):
                # Allow ?. ... which shortcircuits wherever we arrive by taking optional chaining
                for rt in member_access_reachable_types(t):
                    insert(
                        key_t,
                        (
                            rt,
                            goal_t,
                            (17, "left"),
                            max_operator_prec,
                            in_array,
                            in_nested_expression,
                            (
                                (
                                    TypeParameterPType("T"),
                                    UnionPType(
                                        [TypeParameterPType("T"), UndefinedPType()]
                                    ),
                                    min_operator_prec,
                                ),
                            )
                            + in_pattern,
                            max_depth,
                            root_types,
                            max_steps - 1,
                        ),
                    )
    return "NOT_REACHABLE"


def _reachable(
    t: PType,
    goal_t: PType,
    min_operator_prec: OperatorPrecedence,
    max_operator_prec: OperatorPrecedence,
    in_array: List[OperatorPrecedence],
    in_nested_expression: List[OperatorPrecedence],
    in_pattern: List[Tuple[PType, PType, OperatorPrecedence]],
    max_depth=(0, 0),
    root_types: FrozenSet[PType] = frozenset(),
    max_steps=20,
) -> ReachableState:
    """
    Get whether an expression fitting into the goal type can be reached
    by applying any of the allowed operators, staying within the current expression
    :param t: type to explore
    :param goal_t: type to reach
    :param min_operator_prec: operator precedence of lower bound expression -> guarantees we are not exiting enclosing expression
    :param max_operator_prec: operator precedence of upper bound expression -> guarantees we are not binding to a subexpression
    :param in_array: indicates we are inside an array literal. the integer indicates the min_operator_prec of the expression surrounding the array literal
    :param in_nested_expression: indicates we are inside a nested expression. the integer indicates the min_operator_prec of the expression surrounding the nested expression
    """
    if max_steps == 0:
        return "NOT_REACHABLE"
    key_t = (
        t,
        goal_t,
        min_operator_prec,
        max_operator_prec,
        in_array,
        in_nested_expression,
        in_pattern,
        max_depth,
        root_types,
        max_steps,
    )
    if key_t in GLOBAL_REACHABLE_CACHE:
        return GLOBAL_REACHABLE_CACHE[key_t]
    else:
        GLOBAL_REACHABLE_CACHE[key_t] = "UNKNOWN"
    if in_array:
        # if in_array we have to reach the goal type from the array version of the expression
        if (
            _reachable(
                ArrayPType(t),
                goal_t,
                in_array[0],
                MAX_OPERATOR_PRECEDENCE,
                in_array[1:],
                in_nested_expression,
                in_pattern,
                max_depth,
                root_types,
                max_steps=max_steps,
            )
            == "REACHABLE"
        ):
            GLOBAL_REACHABLE_CACHE[key_t] = "REACHABLE"
            return "REACHABLE"
    elif in_nested_expression:
        # if in_nested_expression we have to reach the goal type from the nested expression version of the expression - i.e. we get to override the operator precedence once
        if (
            _reachable(
                t,
                goal_t,
                in_nested_expression[0],
                MAX_OPERATOR_PRECEDENCE,
                in_array,
                in_nested_expression[1:],
                in_pattern,
                max_depth,
                root_types,
                max_steps=max_steps,
            )
            == "REACHABLE"
        ):
            GLOBAL_REACHABLE_CACHE[key_t] = "REACHABLE"
            return "REACHABLE"
    elif in_pattern:
        # if in pattern we have to reach the goal type from the pattern where
        # the parameter type is replaced by the current type
        matches, type_params = extract_type_params(in_pattern[0][0], t)
        if (
            matches
            and _reachable(
                in_pattern[0][1].instantiate_type_params(type_params),
                goal_t,
                in_pattern[0][2],
                max_operator_prec,
                in_array,
                in_nested_expression,
                in_pattern[1:],
                max_depth,
                root_types,
                max_steps=max_steps,
            )
            == "REACHABLE"
        ) or t == AnyPType():
            GLOBAL_REACHABLE_CACHE[key_t] = "REACHABLE"
            return "REACHABLE"
    else:
        # if the goal type is already reached, we are done
        if goal_t >= t:
            GLOBAL_REACHABLE_CACHE[key_t] = "REACHABLE"
            return "REACHABLE"
        if t == AnyPType():
            # we can reach all types from AnyPType
            GLOBAL_REACHABLE_CACHE[key_t] = "REACHABLE"
            return "REACHABLE"

    root_types = frozenset(t.root_values | root_types)
    allowed_ops = OPERATOR_SORTED_BY_PRECEDENCE
    if max_operator_prec[1] == "right":
        # only strictly weaker operations leave the tree
        allowed_ops = [
            op for op in allowed_ops if OPERATOR_PRECEDENCES[op] < max_operator_prec[0]
        ]
    else:
        # weaker or equal operations leave the tree
        allowed_ops = [
            op for op in allowed_ops if OPERATOR_PRECEDENCES[op] <= max_operator_prec[0]
        ]
    if min_operator_prec[1] == "left":
        # only strictly stronger operations stay in the tree
        allowed_ops = [
            op for op in allowed_ops if OPERATOR_PRECEDENCES[op] > min_operator_prec[0]
        ]
    else:
        # stronger or equal operations stay in the tree
        allowed_ops = [
            op for op in allowed_ops if OPERATOR_PRECEDENCES[op] >= min_operator_prec[0]
        ]
    reachable_typs = set(
        (typ, op)
        for op in allowed_ops
        if op in OPERATOR_REACHABLE_TYPE_MAP
        for typ in OPERATOR_REACHABLE_TYPE_MAP[op](t)
        # keep only types that either introduce new root values or do not unnecessarily expand the nesting depth
        if (
            typ.nesting_depth[0] <= max_depth[0]
            and typ.nesting_depth[1] <= max_depth[1]
        )
        or (typ.root_values - root_types)
    )
    # can we reach the goal type from any of the reachable types?
    # we have to stay within the current tree but can easily exit the subtree, hence the precedence stays the same
    res: ReachableState = (
        "REACHABLE"
        if any(
            _reachable(
                rt,
                goal_t,
                min_operator_prec,
                (OPERATOR_PRECEDENCES[op], OPERATOR_ASSOCIATIVITY[op]),
                in_array,
                in_nested_expression,
                in_pattern,
                (
                    max(rt.nesting_depth[0], max_depth[0]),
                    max(rt.nesting_depth[1], max_depth[1]),
                ),
                root_types,
                max_steps=max_steps - 1,
            )
            == "REACHABLE"
            for rt_op in reachable_typs
            for rt, op in [rt_op]
        )
        else "NOT_REACHABLE"
    )
    # prevent recursing indefinitely!
    if (
        res == "NOT_REACHABLE"
        and min_operator_prec <= (17, "left") <= max_operator_prec
        and (
            not in_pattern
            or not in_pattern[0][1]
            == UnionPType([TypeParameterPType("T"), UndefinedPType()])
        )
        and isinstance(t, UnionPType)
        and any(s in FALSEY_TYPES for s in t.types)
    ):
        # Allow ?. ... which shortcircuits wherever we arrive by taking optional chaining
        res = (
            "REACHABLE"
            if any(
                _reachable(
                    rt,
                    goal_t,
                    (17, "left"),
                    max_operator_prec,
                    in_array,
                    in_nested_expression,
                    (
                        (
                            TypeParameterPType("T"),
                            UnionPType([TypeParameterPType("T"), UndefinedPType()]),
                            min_operator_prec,
                        ),
                    )
                    + in_pattern,
                    max_depth,
                    root_types,
                    max_steps=max_steps - 1,
                )
                == "REACHABLE"
                for rt in member_access_reachable_types(t)
            )
            else "NOT_REACHABLE"
        )

    GLOBAL_REACHABLE_CACHE[key_t] = res
    return res


def any_reachable(
    typs: Set[PType],
    goal_t: PType,
    min_operator_prec: OperatorPrecedence,
    max_operator_prec: OperatorPrecedence,
    in_array: List[OperatorPrecedence],
    in_nested_expression: List[OperatorPrecedence],
    in_pattern: List[Tuple[PType, PType, OperatorPrecedence]],
    max_steps=5,
    max_depth=(0, 0),
):
    return any(
        reachable(
            t,
            goal_t,
            min_operator_prec,
            max_operator_prec,
            in_array,
            in_nested_expression,
            in_pattern,
            max_steps,
            max_depth,
        )
        for t in typs
    )


def reachable(
    t: PType,
    goal_t: PType,
    min_operator_prec: OperatorPrecedence,
    max_operator_prec: OperatorPrecedence,
    in_array: List[OperatorPrecedence],
    in_nested_expression: List[OperatorPrecedence],
    in_pattern: List[Tuple[PType, PType, OperatorPrecedence]],
    max_steps=5,
    max_depth=(0, 0),
):
    max_depth = tuple(
        max(xs)
        for xs in zip(
            max_depth,
            t.nesting_depth,
            goal_t.nesting_depth,
            *[p[0].nesting_depth for p in in_pattern],
        )
    )
    res = _reachable_bfs(
        t,
        goal_t,
        min_operator_prec,
        max_operator_prec,
        tuple(in_array),
        tuple(in_nested_expression),
        tuple(in_pattern),
        max_depth,
        max_steps=max_steps,
    )
    # print(t, goal_t, res)
    return res == "REACHABLE"


VARIANCE_MAP = {
    "invariant": lambda x, y: x == y,
    "covariant": lambda x, y: x >= y,
    "contravariant": lambda x, y: y >= x,
}


@lru_cache(maxsize=1000)
def extract_type_params(
    pattern: PType, typ: PType, variance="covariant"
) -> Tuple[bool, Dict[str, PType]]:
    """
    Extract the type parameter from a given type by matching the type against the pattern
    """
    if isinstance(pattern, TypeParameterPType):
        if not isinstance(typ, TypeParameterPType):
            return True, {pattern.name: typ}
        else:
            # in the case T[] matches T[] we dont actually extract a new type but it still matches
            return True, {}
    if VARIANCE_MAP[variance](pattern, typ) and not pattern.type_params():
        return True, {}
    elif isinstance(pattern, UnionPType):
        for p in pattern.types:
            matches, res = extract_type_params(p, typ, variance)
            if matches:
                return True, res
        return False, {}
    elif isinstance(pattern, FunctionPType) and isinstance(typ, FunctionPType):
        # first try to match the parameters
        res = {}
        if len(pattern.call_signature) - pattern.optional_args > len(
            typ.call_signature
        ):
            return False, {}
        for p_p, t_p in zip(pattern.call_signature, typ.call_signature):
            local_matches, local_res = extract_type_params(p_p, t_p, "contravariant")
            if not local_matches:
                return False, {}
            res.update(local_res)
            pattern = pattern.instantiate_type_params(local_res)

        # then try to match the return type
        ret_matches, ret_res = extract_type_params(
            pattern.return_type, typ.return_type, "covariant"
        )
        if not ret_matches:
            return False, {}
        res.update(ret_res)
        return True, res
    elif isinstance(pattern, ArrayPType) and isinstance(typ, ArrayPType):
        return extract_type_params(pattern.element_type, typ.element_type, "invariant")
    elif isinstance(pattern, SetPType) and isinstance(typ, SetPType):
        return extract_type_params(pattern.element_type, typ.element_type, "invariant")
    elif isinstance(pattern, MapPType) and isinstance(typ, MapPType):
        key_match, key_pat = extract_type_params(
            pattern.key_type, typ.key_type, "invariant"
        )
        val_match, val_pat = extract_type_params(
            pattern.value_type, typ.value_type, "invariant"
        )
        if key_match and val_match:
            return True, union_dict(key_pat, val_pat)
        else:
            return False, {}
    elif (
        isinstance(pattern, ArrayPType)
        and isinstance(pattern.element_type, TypeParameterPType)
        and isinstance(typ, StringPType)
    ):
        return True, {pattern.element_type.name: StringPType()}
    elif isinstance(pattern, TuplePType) and isinstance(typ, AbsTuplePType):
        if len(pattern.types) < len(typ.types):
            return False, {}
        res = {}
        for typ_a, typ_b in zip(pattern.types, typ.types):
            int_matches, int_res = extract_type_params(typ_a, typ_b, "invariant")
            if not int_matches:
                return False, {}
            pattern = pattern.instantiate_type_params(int_res)
            res.update(int_res)
        return True, res
    elif isinstance(pattern, TuplePType) and isinstance(typ, TuplePType):
        if len(pattern.types) != len(typ.types):
            return False, {}
        res = {}
        for typ_a, typ_b in zip(pattern.types, typ.types):
            int_matches, int_res = extract_type_params(typ_a, typ_b, "invariant")
            if not int_matches:
                return False, {}
            pattern = pattern.instantiate_type_params(int_res)
            res.update(int_res)
        return True, res
    return False, {}


@fnr_dataclass
class CommandPType(PrimitivePType):
    @property
    def _attributes(self):
        return {
            "addHelpCommand": FunctionPType([CommandPType()], NullPType()),
            "argument": FunctionPType([StringPType()], self),
            "description": FunctionPType([StringPType()], self),
            **super()._attributes,
        }

    def __str__(self):
        return "command"

    @property
    def nesting_depth(self):
        return (0, 0)


@fnr_dataclass
class JSONPType(PrimitivePType):
    @property
    def _attributes(self):
        return {
            **super()._attributes,
            "stringify": FunctionPType([AnyPType()], StringPType()),
            "parse": FunctionPType([StringPType()], BaseTsObject()),
        }

    def __str__(self):
        return "JSON"

    @property
    def nesting_depth(self):
        return (0, 0)


def merge_typs(*args: PType) -> Optional[PType]:
    if not args:
        return None
    typ1, args = args[0], args[1:]
    if not args:
        return typ1
    typ2, args = args[0], args[1:]
    if typ1 >= typ2:
        merged = typ1
    elif typ2 >= typ1:
        merged = typ2
    elif isinstance(typ1, UnionPType):
        merged = UnionPType(typ1.types | {typ2})
    elif isinstance(typ2, UnionPType):
        merged = UnionPType(typ2.types | {typ1})
    else:
        merged = UnionPType({typ1, typ2})
    if args:
        return merge_typs(merged, *args)
    return merged


FALSEY_TYPES = (
    UndefinedPType(),
    NullPType(),
    VoidPType(),
)


def intersection_attribute_union_dict(*args):
    if len(args) == 0:
        return {}
    if len(args) == 1:
        return args[0]

    d1, d2, *args = args
    if args:
        return intersection_attribute_union_dict(
            d1, intersection_attribute_union_dict(d2, *args)
        )
    return {
        k: (merge_typs(*(v[0], d2[k][0])), v[1] and d2[k][1])
        for k, v in d1.items()
        if k in d2
    }
