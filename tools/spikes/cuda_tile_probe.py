#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""THROWAWAY SPIKE (ROADMAP: CUDA Tile IR reader spike, PM [TASK] 2026-11-05). Parse and validate only; no lowering.

    python -I tools/spikes/cuda_tile_probe.py <path to a checkout of NVIDIA/cuda-tile>

What it does, standard library only:
 1. Reads the dialect's TableGen files for the op, type, attribute and bytecode-section inventory.
 2. Reads every bytecode file of the suite (`test/Bytecode/**/*.tileirbc|*.bc`): the 12-byte header (magic, major, minor,
    tag), then the section table (id, length, optional alignment and 0xCB padding), the end marker, the string section.
    A file the suite expects the reader to REJECT must be rejected here for the same class of reason where that class is
    structural (magic, version, section id, section length); payload-level rejections need the full reader and are
    reported as "not checked".
 3. Counts which `cuda_tile.*` operations and tile element types the suite's `.mlir` tests use.

Provenance: nothing is copied from the cuda-tile repository (Apache-2.0 WITH LLVM-exception). The file format facts used
(magic bytes, header layout, section ids, the 0xCB alignment byte, the supported version range) are read from its
`lib/Bytecode/*` sources and written here independently; the repository is only ever read, never executed.
"""
import glob
import os
import re
import sys

MAGIC = b"\x7fTileIR\x00"
ALIGN = 0xCB
SECTIONS = {0: "end", 1: "string", 2: "func", 3: "debug", 4: "constant", 5: "type", 6: "global", 7: "producer"}
SUPPORTED = [(13, 1), (13, 2), (13, 3), (13, 4)]


class Reject(Exception):
    def __init__(self, cls, msg):
        super().__init__(msg)
        self.cls = cls


def varint(b, i):
    r = shift = 0
    while True:
        if i >= len(b) or shift > 63:
            raise Reject("truncated", "truncated varint")
        x = b[i]
        i += 1
        r |= (x & 0x7F) << shift
        shift += 7
        if not x & 0x80:
            return r, i


def header(b):
    """The (major, minor, tag) of the 12-byte header, or the reason it is rejected."""
    if b[:len(MAGIC)] != MAGIC[:len(b)]:
        raise Reject("magic", "invalid magic number")
    if len(b) < 12:
        raise Reject("truncated", "shorter than the 12-byte header")
    ver = (b[8], b[9], b[10] | b[11] << 8)
    if (ver[0], ver[1]) not in SUPPORTED:
        raise Reject("version", "unsupported Tile version %d.%d.%d" % ver)
    return ver


def skip_alignment(b, i):
    """Step over a section's alignment field and its 0xCB padding."""
    align, i = varint(b, i)
    if align < 2:
        raise Reject("section", "alignment flag with alignment < 2")
    while i % align:
        if i >= len(b) or b[i] != ALIGN:
            raise Reject("section", "alignment padding is not 0xCB")
        i += 1
    return i


def section(b, i, sid, aligned):
    """Body (offset, length) of a non-end section whose id byte was just read, and the offset after it."""
    if sid not in SECTIONS:
        raise Reject("section_id", "unknown section ID: %d" % sid)
    length, i = varint(b, i)
    if aligned:
        i = skip_alignment(b, i)
    if i + length > len(b):
        raise Reject("section_length", "section %s runs past the end" % SECTIONS[sid])
    return i, length


def probe(b):
    """Structure of one bytecode buffer: (version, [(section, offset, length)], strings)."""
    ver = header(b)
    i, secs, strings = 12, [], None
    while True:
        if i >= len(b):
            raise Reject("truncated", "no end-of-bytecode section")
        sid, aligned = b[i] & 0x7F, bool(b[i] & 0x80)
        i += 1
        if sid == 0:
            if aligned:
                raise Reject("section", "end section should not have the alignment flag")
            if i != len(b):
                raise Reject("section_length", "end section is not the last section")
            return ver, secs, strings
        i, length = section(b, i, sid, aligned)
        secs.append((SECTIONS[sid], i, length))
        if sid == 1:
            strings = decode_strings(b[i:i + length])
        i += length


def decode_strings(d):
    n, i = varint(d, 0)
    while i % 4:
        i += 1
    offs = [int.from_bytes(d[i + 4 * k:i + 4 * k + 4], "little") for k in range(n)]
    data = d[i + 4 * n:]
    out = []
    for k, o in enumerate(offs):
        e = offs[k + 1] if k + 1 < n else len(data)
        out.append(data[o:e].decode("utf-8", "replace"))
    return out


def td_inventory(root):
    ir = os.path.join(root, "include", "cuda_tile", "Dialect", "CudaTile", "IR")
    read = lambda n: open(os.path.join(ir, n), encoding="utf-8").read()
    ops = re.findall(r'^def CudaTile_(\w+?)Op\s*:\s*\w+<"([a-z_0-9]+)",\s*"(\d+\.\d+)"', read("Ops.td"), re.M)
    types = re.findall(r"^def CudaTile_(\w+)\b", read("Types.td"), re.M)
    attrs = re.findall(r"^def CudaTile_(\w+)\b", read("AttrDefs.td"), re.M)
    opcodes = re.findall(r"PublicOpcode<CudaTile_(\w+?)Op,\s*(0x[0-9A-Fa-f]+|\d+)>", read("BytecodeOpcodes.td"))
    return ops, types, attrs, opcodes


def mlir_census(root):
    names, types, files = {}, {}, 0
    for f in glob.glob(os.path.join(root, "test", "**", "*.mlir"), recursive=True):
        t = open(f, encoding="utf-8", errors="replace").read()
        files += 1
        for m in re.findall(r"(?<![\w.])cuda_tile\.([a-z_0-9]+)", t):
            names[m] = names.get(m, 0) + 1
        for m in re.findall(r"\btile<[^>]*?([a-z]+[0-9]+(?:x[0-9a-z]+)*|ptr<[^>]*>)>", t):
            types[m] = types.get(m, 0) + 1
    return files, names, types


def main(root):
    ops, types, attrs, opcodes = td_inventory(root)
    print("ops (Ops.td defs):", len(ops), "| types:", len(types), "| attrs:", len(attrs), "| public opcodes:", len(opcodes))
    ok = bad = 0
    classes = {}
    for f in sorted(glob.glob(os.path.join(root, "test", "Bytecode", "**", "*"), recursive=True)):
        if not f.endswith((".tileirbc", ".bc")):
            continue
        rel = os.path.relpath(f, os.path.join(root, "test", "Bytecode")).replace("\\", "/")
        b = open(f, "rb").read()
        try:
            ver, secs, strings = probe(b)
            ok += 1
            print("PARSE  %-62s v%d.%d.%d %s strings=%s" % (rel, *ver, [s for s, _, _ in secs], None if strings is None else len(strings)))
        except Reject as e:
            bad += 1
            classes[e.cls] = classes.get(e.cls, 0) + 1
            print("REJECT %-62s [%s] %s" % (rel, e.cls, e))
    print("parsed:", ok, "rejected:", bad, classes)
    files, names, ttypes = mlir_census(root)
    mnem = {m for _, m, _ in ops}
    used = sorted(set(names) & mnem)
    unused = sorted(mnem - set(names))
    since = {}
    for _, m, v in ops:
        since.setdefault(v, []).append(m)
    print("ops by first version:", {v: len(x) for v, x in sorted(since.items())})
    print(".mlir tests scanned:", files, "| dialect ops used by the tests:", len(used), "of", len(mnem), "| never used:", unused)
    print("types seen in tile<...>:", dict(sorted(ttypes.items(), key=lambda kv: -kv[1])[:25]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) == 2 else print(__doc__) or 2)
