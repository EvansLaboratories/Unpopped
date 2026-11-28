#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent binary128 reference for tests/soft_f128_vectors.rs (ROADMAP E3).

Exact rational arithmetic (fractions.Fraction) rounded to binary128 by a separate implementation (round to nearest, ties to even),
sharing no code with crates/unpopped/src/soft_f128.rs. Provenance: written by the unpopped lane for this test (no third-party code, no copied vectors; the only dependency is the
Python standard library, `fractions` only), under the repository's licence. UNVERIFIED against a third-party binary128 library
(none is installed here): the check is two independent implementations agreeing, not a vendor's vectors. Seeded, so the output is reproducible:
    python tools/gen_f128_vectors.py > crates/unpopped/tests/data/soft_f128_vectors.txt
"""
from fractions import Fraction

EXP_BITS, FRAC_BITS, BIAS = 15, 112, 16383
EMIN = 1 - BIAS            # exponent of the smallest normal
INF = (2**EXP_BITS - 1) << FRAC_BITS
QNAN = INF | (1 << (FRAC_BITS - 1))
MAXFIN = ((2**EXP_BITS - 2) << FRAC_BITS) | (2**FRAC_BITS - 1)


def decode(bits, ebits=EXP_BITS, fbits=FRAC_BITS):
    bias = 2 ** (ebits - 1) - 1
    sign = -1 if bits >> (ebits + fbits) else 1
    e = (bits >> fbits) & (2**ebits - 1)
    f = bits & (2**fbits - 1)
    if e == 2**ebits - 1:
        return ("inf", sign) if f == 0 else ("nan", 0)
    if e == 0:
        return ("fin", sign * Fraction(f, 2**fbits) * Fraction(2) ** (1 - bias))
    return ("fin", sign * (1 + Fraction(f, 2**fbits)) * Fraction(2) ** (e - bias))


def encode(x, ebits=EXP_BITS, fbits=FRAC_BITS):
    """The IEEE bits of (ebits, fbits) of the exact rational x, round to nearest even."""
    bias = 2 ** (ebits - 1) - 1
    emin = 1 - bias
    inf = (2**ebits - 1) << fbits
    if x == 0:
        return 0
    sign = 1 << (ebits + fbits) if x < 0 else 0
    ax = abs(x)
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if Fraction(2) ** e > ax:
        e -= 1
    e = max(e, emin)
    q = ax / Fraction(2) ** (e - fbits)       # in units of the last place
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n >= 2 ** (fbits + 1):                  # carried into the next binade
        n //= 2
        e += 1
    if e > bias:
        return sign | inf
    if n < 2**fbits:                           # subnormal (only when e == emin)
        return sign | n
    return sign | ((e + bias) << fbits) | (n - 2**fbits)


def enc_signed(neg, d, ebits, fbits):
    """Encode a decoded finite value, keeping the sign of a zero (the rational 0 has none)."""
    if d[1] == 0:
        return (1 << (ebits + fbits)) if neg else 0
    return encode(d[1], ebits, fbits)


def to_int(bits, mode):
    """The 256-bit two's-complement of the f128 value rounded to an integer under `mode`, or None (NaN, infinity, |x| >= 2^255)."""
    d = decode(bits)
    if d[0] != "fin":
        return None
    x = d[1]
    fl = x.numerator // x.denominator
    frac = x - fl
    if mode == "trunc":
        n = fl + 1 if (x < 0 and frac != 0) else fl
    elif mode == "floor":
        n = fl
    elif mode == "ceil":
        n = fl + (1 if frac != 0 else 0)
    elif mode == "rne":
        n = fl + (1 if (frac > Fraction(1, 2) or (frac == Fraction(1, 2) and fl % 2 == 1)) else 0)
    elif mode == "rna":
        # ties away from zero
        if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and x > 0):
            n = fl + 1
        else:
            n = fl
    else:
        raise ValueError(mode)
    if abs(n) >= 2**255:
        return None
    return n % 2**256


def _inf(sign):
    return INF | (1 << 127 if sign < 0 else 0)


def _add(a, b, da, db, subtract):
    if subtract:
        db = (db[0], -db[1])
    if da[0] == "inf" and db[0] == "inf":
        return _inf(da[1]) if da[1] == db[1] else QNAN
    if da[0] == "inf":
        return _inf(da[1])
    if db[0] == "inf":
        return _inf(db[1])
    s = da[1] + db[1]
    if s == 0:
        # an exact zero sum is +0 unless both operands are -0 (RNE); `b` is the second operand as given
        b_neg = (b >> 127 == 1) != subtract
        return (1 << 127) if (a >> 127 == 1 and b_neg and da[1] == 0 and db[1] == 0) else 0
    return encode(s)


def _mul(a, b, da, db):
    zero = lambda d: d[0] == "fin" and d[1] == 0
    if (da[0] == "inf" and zero(db)) or (db[0] == "inf" and zero(da)):
        return QNAN
    neg = (a >> 127) ^ (b >> 127)
    if da[0] == "inf" or db[0] == "inf":
        return INF | (neg << 127)
    p = da[1] * db[1]
    return (neg << 127) if p == 0 else encode(p)


def _div(a, b, da, db):
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


def op(name, a, b):
    da, db = decode(a), decode(b)
    if da[0] == "nan" or db[0] == "nan":
        return QNAN
    table = {
        "add": lambda: _add(a, b, da, db, False),
        "sub": lambda: _add(a, b, da, db, True),
        "mul": lambda: _mul(a, b, da, db),
        "div": lambda: _div(a, b, da, db),
    }
    return table[name]()


class Rng:
    """splitmix64, seeded: a reproducible test-data generator (not for security); the `random` module is not used."""

    def __init__(self, seed):
        self.state = seed & (2**64 - 1)

    def _next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & (2**64 - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
        return z ^ (z >> 31)

    def getrandbits(self, k):
        out, got = 0, 0
        while got < k:
            out |= self._next() << got
            got += 64
        return out & (2**k - 1)

    def randrange(self, lo, hi=None):
        if hi is None:
            lo, hi = 0, lo
        return lo + self.getrandbits((hi - lo).bit_length() + 64) % (hi - lo)

    def random(self):
        return self.getrandbits(53) / 2**53

    def choice(self, seq):
        return seq[self.randrange(len(seq))]


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


SPECIALS = [0, 1 << 127, 1, 2**FRAC_BITS - 1, (BIAS << FRAC_BITS), (BIAS << FRAC_BITS) | (1 << 127), INF, INF | (1 << 127), QNAN, MAXFIN]


def main():
    rng = Rng(0xE3128)
    rows = []
    pairs = [(a, b) for a in SPECIALS for b in SPECIALS]
    pairs += [(rand_bits(rng), rand_bits(rng)) for _ in range(70)]
    # near-cancellation and tie cases: a and b of the same binade
    for _ in range(20):
        e = rng.randrange(1, 2**EXP_BITS - 1)
        f = rng.getrandbits(FRAC_BITS)
        a = (e << FRAC_BITS) | f
        b = (1 << 127) | (e << FRAC_BITS) | (f ^ rng.getrandbits(8))
        pairs.append((a, b))
    for a, b in pairs:
        for name in ("add", "sub", "mul", "div"):
            rows.append((name, a, b, op(name, a, b)))
    conv = []
    for a, _ in pairs[:45]:
        d = decode(a)
        conv.append(("tof64", a, 0, 0xFFFFFFFFFFFFFFFF if False else (0x7FF8000000000000 if d[0] == "nan" else (enc_signed(a >> 127, d, 11, 52) if d[0] == "fin" else (0x7FF0000000000000 | (1 << 63 if d[1] < 0 else 0))))))
        conv.append(("tof32", a, 0, 0x7FC00000 if d[0] == "nan" else (enc_signed(a >> 127, d, 8, 23) if d[0] == "fin" else (0x7F800000 | (1 << 31 if d[1] < 0 else 0)))))
        for mode in ("trunc", "floor", "ceil", "rne", "rna"):
            r = to_int(a, mode)
            if r is not None:
                conv.append(("int_" + mode, a, 0, r))
    # integer roundings need exact ties (n + 1/2) and values just either side of one, which random patterns almost never are
    halves = [Fraction(2 * k + 1, 2) for k in range(-6, 6)] + [Fraction(2 * k + 1, 2) + eps for k in (-3, 2) for eps in (Fraction(1, 2**100), -Fraction(1, 2**100))]
    for x in halves:
        bits = encode(x)
        for mode in ("trunc", "floor", "ceil", "rne", "rna"):
            conv.append(("int_" + mode, bits, 0, to_int(bits, mode)))
    # f32 rounding must be ONE rounding from the exact binary128 value: a value a hair either side of an f32 tie is a tie in f64, so a path
    # through f64 would round it by the tie rule (to even) instead of by the side it is on
    for _ in range(40):
        f32 = (rng.getrandbits(1) << 31) | (rng.randrange(1, 254) << 23) | rng.getrandbits(23)
        lo = decode(f32, 8, 23)[1]
        hi = decode(f32 + 1 if f32 & 0x7FFFFFFF < 0x7F7FFFFF else f32, 8, 23)[1]
        mid = (lo + hi) / 2
        for eps in (Fraction(1, 2**100), -Fraction(1, 2**100)):
            x = mid * (1 + eps)
            conv.append(("tof32", encode(x), 0, enc_signed(1 if x < 0 else 0, ("fin", x), 8, 23)))
    for _ in range(50):
        x = rng.choice([rng.getrandbits(64), rng.getrandbits(52), rng.getrandbits(23)])
        sign = rng.getrandbits(1)
        e = rng.randrange(0, 2047) if rng.random() < 0.8 else rng.choice([0, 2046, 2047])
        bits64 = (sign << 63) | (e << 52) | (x & (2**52 - 1))
        d = decode(bits64, 11, 52)
        res = 0x7FFF8000000000000000000000000000 if d[0] == "nan" else (enc_signed(bits64 >> 63, d, 15, 112) if d[0] == "fin" else (INF | (1 << 127 if d[1] < 0 else 0)))
        conv.append(("fromf64", bits64, 0, res))
        bits32 = (sign << 31) | (rng.randrange(0, 256) << 23) | (x & (2**23 - 1))
        d = decode(bits32, 8, 23)
        res = 0x7FFF8000000000000000000000000000 if d[0] == "nan" else (enc_signed(bits32 >> 31, d, 15, 112) if d[0] == "fin" else (INF | (1 << 127 if d[1] < 0 else 0)))
        conv.append(("fromf32", bits32, 0, res))
    print("# Generated by tools/gen_f128_vectors.py (seed 0xE3128): an independent Python reference, UNVERIFIED against a third-party library.")
    print("# op a b expected: bit patterns in hex without leading zeros. add/sub/mul/div: binary128 (an expected NaN is the canonical quiet NaN 0x7fff8000...0).")
    print("# tof64/tof32: the f64/f32 bits of a (RNE; NaN is the canonical quiet NaN); int_<mode>: the 256-bit two's complement of a rounded to an")
    print("# integer under trunc/floor/ceil/rne/rna (only rows with |a| < 2^255); fromf64/fromf32: a is the f64/f32 bits, the expected is binary128.")
    for name, a, b, r in rows + conv:
        print(f"{name} {a:x} {b:x} {r:x}")

if __name__ == "__main__":
    main()
