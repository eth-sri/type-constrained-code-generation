import re
from typing import Dict

from typesafe_llm.parser.types_base import PType
from typesafe_llm.parser.types_go import (
    FloatPType,
    IntPType,
    UIntPType,
    PRIMITIVE_TYPE_MAP,
    FunctionPType,
)

_defs = """
const (
	E   = 2.71828182845904523536028747135266249775724709369995957496696763 // https://oeis.org/A001113
	Pi  = 3.14159265358979323846264338327950288419716939937510582097494459 // https://oeis.org/A000796
	Phi = 1.61803398874989484820458683436563811772030917980576286213544862 // https://oeis.org/A001622

	Sqrt2   = 1.41421356237309504880168872420969807856967187537694807317667974 // https://oeis.org/A002193
	SqrtE   = 1.64872127070012814684865078781416357165377610071014801157507931 // https://oeis.org/A019774
	SqrtPi  = 1.77245385090551602729816748334114518279754945612238712821380779 // https://oeis.org/A002161
	SqrtPhi = 1.27201964951406896425242246173749149171560804184009624861664038 // https://oeis.org/A139339

	Ln2    = 0.693147180559945309417232121458176568075500134360255254120680009 // https://oeis.org/A002162
	Log2E  = 1 / Ln2
	Ln10   = 2.30258509299404568401799145468436420760110148862877297603332790 // https://oeis.org/A002392
	Log10E = 1 / Ln10
)

// Floating-point limit values.
// Max is the largest finite value representable by the type.
// SmallestNonzero is the smallest positive, non-zero value representable by the type.
const (
	MaxFloat32             = 0x1p127 * (1 + (1 - 0x1p-23)) // 3.40282346638528859811704183484516925440e+38
	SmallestNonzeroFloat32 = 0x1p-126 * 0x1p-23            // 1.401298464324817070923729583289916131280e-45

	MaxFloat64             = 0x1p1023 * (1 + (1 - 0x1p-52)) // 1.79769313486231570814527423731704356798070e+308
	SmallestNonzeroFloat64 = 0x1p-1022 * 0x1p-52            // 4.9406564584124654417656879286822137236505980e-324
)

// Integer limit values.
const (
	intSize = 32 << (^uint(0) >> 63) // 32 or 64

	MaxInt    = 1<<(intSize-1) - 1  // MaxInt32 or MaxInt64 depending on intSize.
	MinInt    = -1 << (intSize - 1) // MinInt32 or MinInt64 depending on intSize.
	MaxInt8   = 1<<7 - 1            // 127
	MinInt8   = -1 << 7             // -128
	MaxInt16  = 1<<15 - 1           // 32767
	MinInt16  = -1 << 15            // -32768
	MaxInt32  = 1<<31 - 1           // 2147483647
	MinInt32  = -1 << 31            // -2147483648
	MaxInt64  = 1<<63 - 1           // 9223372036854775807
	MinInt64  = -1 << 63            // -9223372036854775808
	MaxUint   = 1<<intSize - 1      // MaxUint32 or MaxUint64 depending on intSize.
	MaxUint8  = 1<<8 - 1            // 255
	MaxUint16 = 1<<16 - 1           // 65535
	MaxUint32 = 1<<32 - 1           // 4294967295
	MaxUint64 = 1<<64 - 1           // 18446744073709551615
)

func Abs(x float64) float64
func Acos(x float64) float64
func Acosh(x float64) float64
func Asin(x float64) float64
func Asinh(x float64) float64
func Atan(x float64) float64
func Atan2(y, x float64) float64
func Atanh(x float64) float64
func Cbrt(x float64) float64
func Ceil(x float64) float64
func Copysign(f, sign float64) float64
func Cos(x float64) float64
func Cosh(x float64) float64
func Dim(x, y float64) float64
func Erf(x float64) float64
func Erfc(x float64) float64
func Erfcinv(x float64) float64
func Erfinv(x float64) float64
func Exp(x float64) float64
func Exp2(x float64) float64
func Expm1(x float64) float64
func FMA(x, y, z float64) float64
func Float32bits(f float32) uint32
func Float32frombits(b uint32) float32
func Float64bits(f float64) uint64
func Float64frombits(b uint64) float64
func Floor(x float64) float64
func Frexp(f float64) (frac float64, exp int)
func Gamma(x float64) float64
func Hypot(p, q float64) float64
func Ilogb(x float64) int
func Inf(sign int) float64
func IsInf(f float64, sign int) bool
func IsNaN(f float64) (is bool)
func J0(x float64) float64
func J1(x float64) float64
func Jn(n int, x float64) float64
func Ldexp(frac float64, exp int) float64
func Lgamma(x float64) (lgamma float64, sign int)
func Log(x float64) float64
func Log10(x float64) float64
func Log1p(x float64) float64
func Log2(x float64) float64
func Logb(x float64) float64
func Max(x, y float64) float64
func Min(x, y float64) float64
func Mod(x, y float64) float64
func Modf(f float64) (int float64, frac float64)
func NaN() float64
func Nextafter(x, y float64) (r float64)
func Nextafter32(x, y float32) (r float32)
func Pow(x, y float64) float64
func Pow10(n int) float64
func Remainder(x, y float64) float64
func Round(x float64) float64
func RoundToEven(x float64) float64
func Signbit(x float64) bool
func Sin(x float64) float64
func Sincos(x float64) (sin, cos float64)
func Sinh(x float64) float64
func Sqrt(x float64) float64
func Tan(x float64) float64
func Tanh(x float64) float64
func Trunc(x float64) float64
func Y0(x float64) float64
func Y1(x float64) float64
func Yn(n int, x float64) float64
        """


def extract_defs(defs: str) -> Dict[str, PType]:
    res = {}
    const_pattern = re.compile(r"^\s*(\w+)\s*=\s*.+")
    func_pattern = re.compile(r"^\s*func\s+(\w+)\s*\(([^)]*)\)\s*([\w\[\]]+)")

    for line in defs.splitlines():
        const_match = const_pattern.match(line)
        func_match = func_pattern.match(line)

        if const_match:
            varname = const_match.group(1)
            typ = FloatPType()
            if "Int" in varname:
                typ = IntPType()
            elif "Uint" in varname:
                typ = UIntPType()
            res[varname] = typ
        elif func_match:
            func_name = func_match.group(1)
            args = func_match.group(2).split(",") if func_match.group(2) else []
            return_type = PRIMITIVE_TYPE_MAP[func_match.group(3)]
            # manage the "multiple args one type annotation" go madness
            untyped_args = 0
            arg_types = []
            for arg in args:
                split_arg = arg.strip().split(" ")
                if len(split_arg) < 2:
                    untyped_args += 1
                else:
                    typ = PRIMITIVE_TYPE_MAP[split_arg[1]]
                    arg_types.extend((untyped_args + 1) * [typ])
            res[func_name] = FunctionPType(arg_types, return_type)

    return res


res = extract_defs(_defs)
formatted_res = {f'"{k}"': repr(v) for k, v in res.items()}
for k, v in formatted_res.items():
    print(k, ":", v, ",")
