#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent binary64 directed-rounding reference for tests/numerics_f64_directed.rs (ROADMAP U43).

Exact rational arithmetic (fractions.Fraction, math.isqrt) rounded to binary64 in a directed mode by an implementation that shares no
code with crates/unpopped/src/numerics_f64.rs (which works in a 256-bit integer frame with a sticky remainder; this one compares
exact rationals against the binary64 grid). Provenance: written by the unpopped lane for this test (no third-party code, no copied
vectors; the Python standard library only), under the repository's licence. Seeded (its own splitmix64), so the output is
reproducible:
    python tools/gen_f64_directed_vectors.py > crates/unpopped/tests/data/numerics_f64_directed.txt
One row: `op mode a b c result`, every field the hex bits of a binary64 except op and mode (`add sub mul div fma sqrt`, `rz rd ru`).
"""
import math
from fractions import Fraction

MASK = (1 << 64) - 1
MAXF = (2**53 - 1) * Fraction(2) ** 971
BIAS_EMIN = -1022


class Rng:
    """splitmix64, the generator's only source of variety."""

    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)


def value(bits):
    """The exact value of finite binary64 bits."""
    sign = -1 if bits >> 63 else 1
    e = (bits >> 52) & 0x7FF
    f = bits & ((1 << 52) - 1)
    if e == 0:
        return sign * Fraction(f) * Fraction(2) ** -1074
    return sign * Fraction(f | (1 << 52)) * Fraction(2) ** (e - 1075)


def bits_of(x, neg_zero=False):
    """The bits of x, which must be exactly a binary64 (or zero, signed by neg_zero)."""
    if x == 0:
        return (1 << 63) if neg_zero else 0
    sign = (1 << 63) if x < 0 else 0
    ax = abs(x)
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if Fraction(2) ** e > ax:
        e -= 1
    e = max(e, BIAS_EMIN)
    m = ax / Fraction(2) ** (e - 52)
    assert m.denominator == 1, "not a binary64"
    m = int(m)
    if e == BIAS_EMIN and m < (1 << 52):
        return sign | m
    return sign | ((e + 1023) << 52) | (m - (1 << 52))


def grid_floor(x):
    """The largest binary64 grid point <= x for x > 0 (capped at MAX)."""
    if x >= MAXF:
        return MAXF
    e = max(x.numerator.bit_length() - x.denominator.bit_length(), BIAS_EMIN)
    if Fraction(2) ** e > x:
        e -= 1
    e = max(e, BIAS_EMIN)
    ulp = Fraction(2) ** (e - 52)
    return (x // ulp) * ulp


def round_pos(x, inexact_above, mode_up):
    """For x >= 0 plus an infinitesimal when inexact_above: the grid point (or None for infinity) toward +inf if mode_up, else toward 0."""
    lo = grid_floor(x)
    if not mode_up:
        return lo
    if lo == x and not inexact_above:
        return lo
    if lo >= MAXF:
        return None
    if lo == 0:
        return Fraction(2) ** -1074
    e = max(lo.numerator.bit_length() - lo.denominator.bit_length(), BIAS_EMIN)
    if Fraction(2) ** e > lo:
        e -= 1
    e = max(e, BIAS_EMIN)
    nxt = lo + Fraction(2) ** (e - 52)
    return None if nxt > MAXF else nxt


def directed(x, mode, inexact_above=False, sign_hint=False):
    """Bits of the directed rounding of the exact real x (nonzero, or zero with sign_hint = negative zero).

    inexact_above: the true value is |x| plus an infinitesimal (an irrational root); only used with x > 0.
    """
    if x == 0 and not inexact_above:
        return bits_of(x, sign_hint)
    neg = x < 0
    ax = -x if neg else x
    if mode == "rz":
        up_mag = False
    elif mode == "ru":
        up_mag = not neg
    else:
        up_mag = neg
    r = round_pos(ax, inexact_above, up_mag)
    if r is None:
        return (0xFFF0000000000000 if neg else 0x7FF0000000000000)
    return bits_of(-r if neg else r, neg_zero=neg)


def zero_sign(mode, same_sign_zero, negs):
    """The sign of an exact-zero sum: like-signed zero terms keep their sign, else +0 except rd."""
    if same_sign_zero:
        return negs
    return mode == "rd"


def op_add(mode, a, b):
    x, y = value(a), value(b)
    s = x + y
    if s == 0:
        both = x == 0 and y == 0 and (a >> 63) == (b >> 63)
        return bits_of(0, zero_sign(mode, both, bool(a >> 63)))
    return directed(s, mode)


def op_mul(mode, a, b):
    p = value(a) * value(b)
    if p == 0:
        return bits_of(0, bool((a ^ b) >> 63))
    return directed(p, mode)


def op_div(mode, a, b):
    x, y = value(a), value(b)
    if x == 0:
        return bits_of(0, bool((a ^ b) >> 63))
    return directed(x / y, mode)


def op_fma(mode, a, b, c):
    p, z = value(a) * value(b), value(c)
    s = p + z
    if s == 0:
        pneg = bool((a ^ b) >> 63)
        both = p == 0 and z == 0 and pneg == bool(c >> 63)
        return bits_of(0, zero_sign(mode, both, pneg))
    return directed(s, mode)


def op_sqrt(mode, a):
    x = value(a)
    k = 1300
    n = x * Fraction(2) ** (2 * k)
    r = math.isqrt(n.numerator // n.denominator)
    exact = Fraction(r * r) == n
    return directed(Fraction(r, 2**k), mode, inexact_above=not exact)


def normals(rng, count):
    """Finite binary64 bit patterns: random, and with the exponent drawn so the whole range gets visited."""
    out = []
    for _ in range(count):
        w = rng.next()
        e = (rng.next() % 2046) + 1 if rng.next() % 3 else rng.next() % 2047
        if e == 2047:
            e = 2046
        out.append((w & ~(0x7FF << 52) & MASK) | (e << 52))
    return out


SPECIAL = [
    0x0000000000000000, 0x8000000000000000, 0x0000000000000001, 0x8000000000000001, 0x000FFFFFFFFFFFFF, 0x0010000000000000,
    0x8010000000000000, 0x3FF0000000000000, 0xBFF0000000000000, 0x3FF0000000000001, 0x3FEFFFFFFFFFFFFF, 0x7FEFFFFFFFFFFFFF,
    0xFFEFFFFFFFFFFFFF, 0x4000000000000000, 0x4008000000000000, 0x3FD5555555555555, 0x7FE0000000000000, 0x0360000000000000,
]


def pick(rng, pool):
    return pool[rng.next() % len(pool)]


def operands(rng):
    """The operand pool: specials, near-ties (1 +- 2^-53), and random values."""
    pool = SPECIAL + normals(rng, 60)
    pool += [0x3CA0000000000000, 0x3C90000000000000, 0xBCA0000000000000, 0x3FF0000000000002]
    return pool


def rows(rng):
    pool = operands(rng)
    out = []
    for mode in ("rz", "rd", "ru"):
        for _ in range(26):
            a, b, c = pick(rng, pool), pick(rng, pool), pick(rng, pool)
            out.append(("add", mode, a, b, 0, op_add(mode, a, b)))
            nb = b ^ (1 << 63)
            out.append(("sub", mode, a, b, 0, op_add(mode, a, nb)))
            out.append(("mul", mode, a, b, 0, op_mul(mode, a, b)))
            if value(b) != 0:
                out.append(("div", mode, a, b, 0, op_div(mode, a, b)))
            out.append(("fma", mode, a, b, c, op_fma(mode, a, b, c)))
            if a >> 63 == 0 and value(a) != 0:
                out.append(("sqrt", mode, a, 0, 0, op_sqrt(mode, a)))
        out += edge_rows(rng, mode)
        out += cancel_rows(rng, mode)
    return out


def with_exp(rng, lo, hi, neg=None):
    """A binary64 with a random mantissa and a biased exponent field in lo..hi (0 is subnormal); a random sign unless neg is given."""
    e = lo + rng.next() % (hi - lo + 1)
    sign = (rng.next() & 1) if neg is None else int(neg)
    return (sign << 63) | (e << 52) | (rng.next() & ((1 << 52) - 1))


def edge_rows(rng, mode):
    """Operands aimed at the subnormal range and at overflow, where a naive reference or implementation goes wrong."""
    out = []
    for _ in range(10):
        a, b = with_exp(rng, 300, 700), with_exp(rng, 300, 700)
        out.append(("mul", mode, a, b, 0, op_mul(mode, a, b)))
        d, n = with_exp(rng, 1, 500), with_exp(rng, 1030, 1560)
        out.append(("div", mode, d, n, 0, op_div(mode, d, n)))
        c = with_exp(rng, 0, 3)
        out.append(("fma", mode, a, b, c, op_fma(mode, a, b, c)))
        x, y = with_exp(rng, 0, 2), with_exp(rng, 0, 2)
        out.append(("add", mode, x, y, 0, op_add(mode, x, y)))
        out.append(("sub", mode, x, y, 0, op_add(mode, x, y ^ (1 << 63))))
        z = with_exp(rng, 0, 40, neg=False)
        if value(z) != 0:
            out.append(("sqrt", mode, z, 0, 0, op_sqrt(mode, z)))
        big, two, bigger = with_exp(rng, 2000, 2046), with_exp(rng, 1030, 1040), with_exp(rng, 1990, 2046)
        out.append(("mul", mode, big, two, 0, op_mul(mode, big, two)))
        out.append(("add", mode, big, bigger, 0, op_add(mode, big, bigger)))
    return out


def cancel_rows(rng, mode):
    """Exact cancellations and signed-zero sums, where the sign of the zero depends on the mode (IEEE 754 6.3)."""
    out = []
    zeros = (0x0000000000000000, 0x8000000000000000)
    for a in zeros:
        for b in zeros:
            out.append(("add", mode, a, b, 0, op_add(mode, a, b)))
            out.append(("sub", mode, a, b, 0, op_add(mode, a, b ^ (1 << 63))))
    for _ in range(3):
        a = with_exp(rng, 1, 2046)
        out.append(("add", mode, a, a ^ (1 << 63), 0, op_add(mode, a, a ^ (1 << 63))))
        out.append(("sub", mode, a, a, 0, op_add(mode, a, a ^ (1 << 63))))
    two, three, six = 0x4000000000000000, 0x4008000000000000, 0x4018000000000000
    for (a, b, c) in [(two, three, six ^ (1 << 63)), (two ^ (1 << 63), three, six), (two, three, 0x8000000000000000)]:
        out.append(("fma", mode, a, b, c, op_fma(mode, a, b, c)))
    for c in zeros:
        out.append(("fma", mode, 0, three, c, op_fma(mode, 0, three, c)))
        out.append(("fma", mode, 0x8000000000000000, three, c, op_fma(mode, 0x8000000000000000, three, c)))
    return out


def main():
    rng = Rng(0x0F64D1EC7ED)
    for op, mode, a, b, c, r in rows(rng):
        print(f"{op} {mode} {a:016x} {b:016x} {c:016x} {r:016x}")


if __name__ == "__main__":
    main()
