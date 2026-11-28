#!/usr/bin/env python3
"""Independent binary128 reference for tests/soft_f128_vectors.rs (ROADMAP E3).

Exact rational arithmetic (fractions.Fraction) rounded to binary128 by a separate implementation (round to nearest, ties to even),
sharing no code with crates/unpopped/src/soft_f128.rs. UNVERIFIED against a third-party binary128 library (none is installed
here): the check is two independent implementations agreeing, not a vendor's vectors. Seeded, so the output is reproducible:
    python tools/gen_f128_vectors.py > crates/unpopped/tests/data/soft_f128_vectors.txt
"""
import random
from fractions import Fraction

EXP_BITS, FRAC_BITS, BIAS = 15, 112, 16383
EMIN = 1 - BIAS            # exponent of the smallest normal
INF = (2**EXP_BITS - 1) << FRAC_BITS
QNAN = INF | (1 << (FRAC_BITS - 1))
MAXFIN = ((2**EXP_BITS - 2) << FRAC_BITS) | (2**FRAC_BITS - 1)


def decode(bits):
    sign = -1 if bits >> 127 else 1
    e = (bits >> FRAC_BITS) & (2**EXP_BITS - 1)
    f = bits & (2**FRAC_BITS - 1)
    if e == 2**EXP_BITS - 1:
        return ("inf", sign) if f == 0 else ("nan", 0)
    if e == 0:
        return ("fin", sign * Fraction(f, 2**FRAC_BITS) * Fraction(2) ** EMIN)
    return ("fin", sign * (1 + Fraction(f, 2**FRAC_BITS)) * Fraction(2) ** (e - BIAS))


def encode(x):
    """The binary128 bits of the exact rational x, round to nearest even."""
    if x == 0:
        return 0
    sign = 1 << 127 if x < 0 else 0
    ax = abs(x)
    # find e with 2^e <= ax < 2^(e+1)
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if Fraction(2) ** e > ax:
        e -= 1
    e = max(e, EMIN)
    q = ax / Fraction(2) ** (e - FRAC_BITS)       # in units of the last place
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n >= 2 ** (FRAC_BITS + 1):                  # carried into the next binade
        n //= 2
        e += 1
    if e > BIAS:
        return sign | INF
    if n < 2**FRAC_BITS:                           # subnormal (only when e == EMIN)
        return sign | n
    return sign | ((e + BIAS) << FRAC_BITS) | (n - 2**FRAC_BITS)


def op(name, a, b):
    da, db = decode(a), decode(b)
    if da[0] == "nan" or db[0] == "nan":
        return QNAN
    inf = lambda s: INF | (1 << 127 if s < 0 else 0)
    if name in ("add", "sub"):
        if name == "sub":
            db = (db[0], -db[1])
        if da[0] == "inf" and db[0] == "inf":
            return inf(da[1]) if da[1] == db[1] else QNAN
        if da[0] == "inf":
            return inf(da[1])
        if db[0] == "inf":
            return inf(db[1])
        s = da[1] + db[1]
        if s == 0:
            # an exact zero sum is +0 unless both operands are -0 (RNE)
            negz = (a >> 127 == 1 and ((b >> 127 == 1) if name == "add" else (b >> 127 == 0)))
            return (1 << 127) if negz and da[1] == 0 and db[1] == 0 else 0
        return encode(s)
    if name == "mul":
        if (da[0] == "inf" and db[0] == "fin" and db[1] == 0) or (db[0] == "inf" and da[0] == "fin" and da[1] == 0):
            return QNAN
        neg = (a >> 127) ^ (b >> 127)
        if da[0] == "inf" or db[0] == "inf":
            return INF | (neg << 127)
        p = da[1] * db[1]
        return (neg << 127) if p == 0 else encode(p)
    if name == "div":
        neg = (a >> 127) ^ (b >> 127)
        if da[0] == "inf" and db[0] == "inf":
            return QNAN
        if da[0] == "inf":
            return INF | (neg << 127)
        if db[0] == "inf":
            return neg << 127
        if db[1] == 0:
            return QNAN if da[1] == 0 else INF | (neg << 127)
        q = da[1] / db[1]
        return (neg << 127) if q == 0 else encode(q)
    raise ValueError(name)


def rand_bits(rng):
    kind = rng.random()
    if kind < 0.10:   # subnormal
        return (rng.getrandbits(1) << 127) | rng.getrandbits(FRAC_BITS)
    if kind < 0.15:   # near the top
        return (rng.getrandbits(1) << 127) | (rng.randrange(2**EXP_BITS - 4, 2**EXP_BITS - 1) << FRAC_BITS) | rng.getrandbits(FRAC_BITS)
    if kind < 0.40:   # small exponent spread around 1
        return (rng.getrandbits(1) << 127) | (rng.randrange(BIAS - 8, BIAS + 8) << FRAC_BITS) | rng.getrandbits(FRAC_BITS)
    if kind < 0.45:   # a short mantissa (exact sums and ties)
        return (rng.getrandbits(1) << 127) | (rng.randrange(BIAS - 130, BIAS + 130) << FRAC_BITS) | (rng.getrandbits(3) << (FRAC_BITS - 3))
    return (rng.getrandbits(1) << 127) | (rng.randrange(1, 2**EXP_BITS - 1) << FRAC_BITS) | rng.getrandbits(FRAC_BITS)


SPECIALS = [0, 1 << 127, 1, 2**FRAC_BITS, (BIAS << FRAC_BITS), (BIAS << FRAC_BITS) | (1 << 127), INF, INF | (1 << 127), QNAN, MAXFIN,
            (BIAS + 1) << FRAC_BITS, ((BIAS - 1) << FRAC_BITS), 2**FRAC_BITS - 1]


def main():
    rng = random.Random(0xE3128)
    rows = []
    pairs = [(a, b) for a in SPECIALS for b in SPECIALS]
    pairs += [(rand_bits(rng), rand_bits(rng)) for _ in range(160)]
    # near-cancellation and tie cases: a and b of the same binade
    for _ in range(40):
        e = rng.randrange(1, 2**EXP_BITS - 1)
        f = rng.getrandbits(FRAC_BITS)
        a = (e << FRAC_BITS) | f
        b = (1 << 127) | (e << FRAC_BITS) | (f ^ rng.getrandbits(8))
        pairs.append((a, b))
    for a, b in pairs:
        for name in ("add", "sub", "mul", "div"):
            rows.append((name, a, b, op(name, a, b)))
    print("# Generated by tools/gen_f128_vectors.py (seed 0xE3128): an independent Python reference, UNVERIFIED against a third-party library.")
    print("# op a b expected: binary128 bit patterns in hex; an expected NaN is the canonical quiet NaN (0x7fff8000...0).")
    for name, a, b, r in rows:
        print(f"{name} {a:032x} {b:032x} {r:032x}")

if __name__ == "__main__":
    main()
