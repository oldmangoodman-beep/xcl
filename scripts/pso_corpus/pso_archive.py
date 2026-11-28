"""Expand archive selector tables into a self-contained XCL pipeline journal."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import zlib

from pso_directory import read_directories

FRAME = struct.Struct("<cB2xIQ")
LIMIT = 64 << 20


def fnv64(data):
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return value


def read_at(file, offset, size):
    if offset < 0 or not 0 < size <= LIMIT:
        raise ValueError("archive range outside supported bounds")
    file.seek(offset)
    data = file.read(size)
    if len(data) != size:
        raise ValueError(f"truncated archive range at {offset}")
    return data


def source_bytes(archive, entry):
    data = read_at(archive, entry["offset"], entry["size"])
    method = entry.get("encoding", "raw")
    if method != "raw":
        inflater = zlib.decompressobj({"zlib": 15, "deflate": -15}[method])
        size = entry["raw_size"]
        if not 0 < size <= LIMIT:
            raise ValueError("invalid uncompressed size")
        data = inflater.decompress(data, size + 1)
        if len(data) != size or not inflater.eof or inflater.unused_data:
            raise ValueError("invalid compressed source range")
    return data


def field(record, spec):
    offset, size = spec
    if offset < 0 or size not in (1, 2, 4, 8) or offset + size > len(record):
        raise ValueError(f"invalid little-endian selector field {spec}")
    return str(int.from_bytes(record[offset:offset + size], "little"))


def expand(profile, archive, records, shader_frames, recipe_frames):
    if profile["version"] != 1:
        raise ValueError("unknown profile version")
    read_directories(profile, records)
    sources = {}
    for table in profile["tables"]:
        stride = table["stride"]
        if not 0 < stride <= LIMIT or table["count"] < 0 or table["offset"] < 0:
            raise ValueError("invalid record table")
        for index in range(table["count"]):
            record = read_at(records, table["offset"] + stride * index, stride)
            ids = {name: field(record, spec) for name, spec in table["fields"].items()}
            variants = table["variants"]
            if "variant_field" in table:
                variants = variants[ids[table["variant_field"]]]
            for variant in variants:
                pieces = []
                stage_ids = []
                for part in variant["parts"]:
                    if len(part) != 1:
                        raise ValueError("wire fragment requires exactly one operation")
                    op, arg = next(iter(part.items()))
                    if op == "hex":
                        value = bytes.fromhex(arg)
                    elif op == "field":
                        name, state_map = arg
                        value = bytes.fromhex(profile["states"][state_map][ids[name]])
                    elif op == "stage":
                        name, stage_map = arg
                        source = profile["stage_maps"][stage_map][ids[name]]
                        if source is None:
                            value = bytes(8)
                        else:
                            entry = profile["sources"][source]
                            region = (entry["offset"], entry["size"], entry.get("encoding", "raw"), entry.get("raw_size"))
                            if region not in sources:
                                blob = source_bytes(archive, entry)
                                key = f"{fnv64(blob):016x}"
                                shader_frames.add(key, "S", 0, blob)
                                sources[region] = key
                            key = sources[region]
                            value = struct.pack("<Q", int(key, 16))
                            stage_ids.append(key)
                    else:
                        raise ValueError(f"unknown fragment {op}")
                    pieces.append(value)
                payload = b"".join(pieces)
                kind, windows = variant["kind"], variant["windows"]
                if kind not in ("C", "G", "M") or not stage_ids or not 0 < len(payload) <= LIMIT:
                    raise ValueError("invalid recipe")
                key = f"{fnv64(payload + bytes([windows])):016x}"
                recipe_frames.add(key, kind, windows, payload)


class FrameWriter:
    def __init__(self, stream):
        self.stream = stream
        self.offsets = {}

    def add(self, key, kind, windows, payload):
        header = FRAME.pack(kind.encode("ascii"), windows, len(payload), int(key, 16))
        if key in self.offsets:
            end = self.stream.tell()
            self.stream.seek(self.offsets[key])
            same = self.stream.read(FRAME.size) == header and self.stream.read(len(payload)) == payload
            self.stream.seek(end)
            if not same:
                raise ValueError(f"frame hash collision for {key}")
            return
        self.offsets[key] = self.stream.tell()
        self.stream.write(header)
        self.stream.write(payload)


@contextmanager
def record_file(profile, archive, archive_path, records_path, scratch):
    chunks = profile.get("record_chunks")
    if chunks is None:
        with Path(records_path or archive_path).open("rb") as records:
            yield records
        return
    if records_path is not None:
        raise ValueError("choose either record_chunks or a separate table file")
    with (Path(scratch) / "records").open("w+b") as records:
        for chunk in chunks:
            records.write(source_bytes(archive, chunk))
        yield records


def generate(profile_path, archive_path, output, records_path=None):
    with Path(profile_path).open(encoding="utf-8") as file:
        profile = json.load(file)
    output = Path(output)
    if output.resolve() in (Path(archive_path).resolve(), Path(profile_path).resolve(), Path(records_path or archive_path).resolve()):
        raise ValueError("output aliases an input")
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        journal = Path(scratch) / "journal"
        with Path(archive_path).open("rb") as archive, journal.open("w+b") as file:
            if "archive_sha256" in profile:
                digest = hashlib.sha256()
                for chunk in iter(lambda: archive.read(1 << 20), b""):
                    digest.update(chunk)
                if digest.hexdigest() != profile["archive_sha256"]:
                    raise ValueError("archive SHA-256 disagrees with profile")
            file.write(b"XCLPJ001")
            shaders = FrameWriter(file)
            with (Path(scratch) / "recipes").open("w+b") as spool:
                recipes = FrameWriter(spool)
                with record_file(profile, archive, archive_path, records_path, scratch) as records:
                    expand(profile, archive, records, shaders, recipes)
                spool.seek(0)
                shutil.copyfileobj(spool, file, 1 << 20)
        os.replace(journal, output)
        return {"source": len(shaders.offsets), "recipe": len(recipes.offsets)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--records", type=Path, help="separate file containing the fixed-stride tables")
    args = parser.parse_args()
    print(json.dumps(generate(args.profile, args.archive, args.journal, args.records)))
