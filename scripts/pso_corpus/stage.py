"""Stage local XCLPJ001 journals for offline pipeline preparation.

python scripts/pso_corpus/stage.py --title FOLDER JOURNAL [JOURNAL ...]
Use the same window configuration for all journals. The resulting .xcl/pso-corpus
contains local shader bytes. An existing different corpus is preserved. Compute
node masks 0 and 1 select the first node on the supported single-node adapter.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

OFFSET, PRIME, MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1
HEADER = struct.Struct("<cBHIQ")
GFX_HEADER = b"XCLGFXP\0" + struct.pack("<I", 1)
IDENTITY_SCHEMA = b"xcl-pso-corpus-v1\0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fnv(data, seed=OFFSET):
    for value in data:
        seed = ((seed ^ value) * PRIME) & MASK
    return seed


def corpus_identity(dest):
    """SHA-256 of the published corpus schema, paths and file contents."""
    digest = hashlib.sha256(IDENTITY_SCHEMA)
    for path in sorted(p for p in dest.rglob("*") if p.is_file() and p.name != "identity"):
        relative = path.relative_to(dest).as_posix().encode("utf-8")
        digest.update(struct.pack("<I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack("<Q", path.stat().st_size))
        contents = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                contents.update(block)
        digest.update(contents.digest())
    return digest.hexdigest()


def shader_set(keys):
    value = OFFSET
    for key in keys:
        if key:
            value = ((value ^ key) * PRIME) & MASK
    return value | 1


def varints(*values):
    result = bytearray()
    for value in values:
        while value >= 128:
            result.append(128 | (value & 127))
            value >>= 7
        result.append(value)
    return result


class Reader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    def take(self, size):
        end = self.pos + size
        require(end <= len(self.data), "short descriptor")
        part = self.data[self.pos:end]
        self.pos = end
        return part

    def ints(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def u32(self):
        return self.ints("<I")[0]


def state(d):
    def words(start, count):
        return struct.unpack_from(f"<{count}I", d, start)
    require(words(576, 1)[0] <= 8, "too many render targets")
    require(not any(d[514:516]) and all(not any(d[165+i*40:168+i*40]) for i in range(8)),
            "noncanonical state padding")
    return varints(*words(120, 98), d[512], d[513], *words(516, 8),
                   *words(568, 15), *words(648, 1))


def gfx_header(tag, keys):
    return bytearray([tag]) + struct.pack(f"<{len(keys)+1}Q", shader_set(keys),
                                         *(shader_set([k]) if k else 0 for k in keys))


def graphics(r):
    keys = r.ints("<5Q")
    require(keys[0] and r.u32() == 656, "missing vertex shader or invalid descriptor size")
    d = r.take(656)
    require(not any(d[:120]) and not any(d[548:560]) and not any(d[564:568])
            and not any(d[628:648]) and not any(d[652:]), "graphics pointer or padding")
    count = r.u32()
    require(count == struct.unpack_from("<I", d, 560)[0] and count <= 64, "input count mismatch")
    fields = state(d) + varints(count)
    for _ in range(count):
        name = bytes(r.take(r.u32()))
        require(b"\0" not in name, "NUL in semantic name")
        fields += varints(len(name)) + name + varints(*r.ints("<6I"))
    tail = bytes(r.data[r.pos:])
    require(tail in (b"", b"DB01"), "invalid graphics trailer")
    return gfx_header(4 if tail else 1, keys) + fields, keys


def mesh(r):
    d = bytearray(656)
    layout = []
    keys = [0, 0]
    bounds = 0
    spans = {8: (120, 328), 9: (448, 4), 10: (452, 44), 11: (496, 52),
             13: (568, 4), 14: (572, 4), 16: (612, 4), 17: (616, 8),
             18: (624, 4), 20: (648, 4), 21: (496, 52)}
    while r.pos < len(r.data):
        kind = r.u32()
        require(kind not in layout, "duplicate mesh subobject")
        layout.append(kind)
        if kind in (0, 12):
            continue
        if kind in (1, 2, 3, 4, 5, 24, 25):
            key, = r.ints("<Q")
            if kind == 2:
                keys[1] = key
            elif kind in (24, 25):
                keys[0] = key
            else:
                require(key == 0, "active classic stage in mesh stream")
        elif kind == 15:
            d[580:612] = r.take(32)
            d[576:580] = r.take(4)
        elif kind in spans:
            offset, size = spans[kind]
            d[offset:offset+size] = r.take(size)
            if kind == 21:
                bounds = r.u32()
                require(bounds <= 1, "invalid depth bounds")
        else:
            raise ValueError(f"unsupported mesh subobject {kind}")
    require(keys[0] and ((24 in layout) != (25 in layout)) and len(layout) <= 19
            and not (11 in layout and 21 in layout), "invalid mesh layout")
    return (gfx_header(3, keys) + bytes([len(layout), *layout]) + state(d)
            + (varints(bounds) if 21 in layout else b"")), keys


def compute(r):
    key, = r.ints("<Q")
    require(key and r.u32() == 56, "invalid compute descriptor")
    d = r.take(56)
    require(r.pos == len(r.data) and not any(d[:24]) and not any(d[28:])
            and struct.unpack_from("<I", d, 24)[0] in (0, 1),
            "unsupported compute pointers, flags or node mask")
    return key


def gfx_frame(file, data):
    file.write(struct.pack("<II", len(data), fnv(data) & 0xffffffff))
    file.write(data)


def frames(path):
    with path.open("rb") as file:
        require(file.read(8) == b"XCLPJ001", f"{path}: invalid journal header")
        size = path.stat().st_size
        while file.tell() < size:
            offset = file.tell()
            header = file.read(HEADER.size)
            require(len(header) == HEADER.size, f"{path}@{offset}: short header")
            tag, win, reserved, n, key = HEADER.unpack(header)
            require(not reserved and n <= size-file.tell(), f"{path}@{offset}: invalid frame")
            yield tag.decode("ascii"), win, key, file.read(n)


def build(journals, dest, spool, records):
    sources, seen, normalized = {}, {}, set()
    gfx_sources, compute_sources = set(), set()
    counts = Counter()
    window = None
    for path in journals:
        for tag, win, key, data in frames(path):
            if tag in ("U", "N"):
                require(not data and win == 0, "invalid usage marker")
                continue
            require(tag in ("S", "G", "C", "M", "X", "R"), f"unknown frame {tag}")
            if tag == "S":
                require(win == 0 and data, "invalid shader frame")
            else:
                count = win & 127
                require(count in ((2, 3, 4, 8, 16) if win & 128 else (1, 2, 3, 4)), "invalid window count")
            digest = hashlib.sha256(data).digest()
            identity = tag, key
            if identity in seen:
                require(seen[identity] == (win, digest), "conflicting frame identity")
                counts["duplicate_frames"] += 1
                continue
            check = fnv(data)
            if tag in ("G", "C", "M", "X"):
                check = ((check ^ win) * PRIME) & MASK
            require(key == check, f"{path}: invalid {tag} hash {key:016x}")
            seen[identity] = win, digest
            if tag == "R":
                require(len(data) == 12, "invalid creation marker")
                continue
            if tag == "X":
                counts["refused_mesh_diagnostics"] += 1
                continue
            if tag == "S":
                sources[key] = spool.tell(), len(data)
                spool.write(data)
                continue
            require(window is None or win == window, "mixed window configurations")
            window = win
            counts[tag] += 1
            if tag == "C":
                compute_sources.add(compute(Reader(data)))
            else:
                payload, keys = (graphics if tag == "G" else mesh)(Reader(data))
                gfx_sources.update(k for k in keys if k)
                signature = hashlib.sha256(payload).digest()
                if signature not in normalized:
                    normalized.add(signature)
                    gfx_frame(records, payload)
    require(window is not None, "no supported recipes")
    (dest / "compute").mkdir()
    used = gfx_sources | compute_sources
    require(used <= sources.keys(), f"{len(used - sources.keys())} missing shader sources")
    require(len({shader_set([k]) for k in gfx_sources}) == len(gfx_sources), "shader key collision")
    for key in sorted(compute_sources):
        offset, length = sources[key]
        spool.seek(offset)
        (dest / "compute" / f"{key:016x}.dxbc").write_bytes(spool.read(length))
    if normalized:
        with (dest / "graphics.bin").open("wb") as out:
            out.write(GFX_HEADER)
            for key in sorted(gfx_sources):
                offset, length = sources[key]
                spool.seek(offset)
                gfx_frame(out, b"\2" + struct.pack("<Q", shader_set([key])) + spool.read(length))
            records.seek(0)
            shutil.copyfileobj(records, out)
    (dest / "windows").write_bytes(f"{window}\n".encode("ascii"))
    (dest / "complete").write_bytes(f"{len(normalized)} {len(compute_sources)}\n".encode("ascii"))
    (dest / "identity").write_bytes(f"{corpus_identity(dest)}\n".encode("ascii"))
    counts.update(graphics_mesh_records=len(normalized), graphics_sources=len(gfx_sources),
                  compute_files=len(compute_sources), compute_node_aliases=counts["C"]-len(compute_sources))
    return dict(counts)


def stage(title, journals):
    require(title.is_dir(), "title directory does not exist")
    parent = title / ".xcl"
    parent.mkdir(exist_ok=True)
    target = parent / "pso-corpus"
    require(not target.exists(), f"existing corpus preserved: {target}")
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        temp = Path(temporary)
        out = temp / "pso-corpus"
        out.mkdir()
        with (temp / "sources").open("w+b") as spool, (temp / "records").open("w+b") as records:
            counts = build(journals, out, spool, records)
        out.rename(target)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", type=Path, required=True)
    parser.add_argument("journals", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(stage(args.title, args.journals), sort_keys=True))
    except (ValueError, OSError) as err:
        parser.exit(1, f"{err}\n")
