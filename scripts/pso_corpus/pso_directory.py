"""Read shader and state dictionaries from fixed-stride archive directories."""


def integer(record, field):
    offset, width = field
    if width not in (1, 2, 4, 8) or offset < 0 or offset + width > len(record):
        raise ValueError("invalid directory integer field")
    return int.from_bytes(record[offset:offset + width], "little")


def directory_entries(stream, directory):
    offset, count, stride = (directory[k] for k in ("offset", "count", "stride"))
    if offset < 0 or count < 0 or not 0 < stride <= 64 << 20:
        raise ValueError("invalid directory range")
    stream.seek(offset)
    for row in range(count):
        record = stream.read(stride)
        if len(record) != stride:
            raise ValueError(f"truncated directory at row {row}")
        key = integer(record, directory["id"]) if "id" in directory else row
        yield str(key), record


def add_entry(mapping, key, value):
    if key in mapping and mapping[key] != value:
        raise ValueError(f"ambiguous directory entry {key}")
    mapping[key] = value


def read_directories(profile, records):
    """Populate the maps for the ordinary profile expander from the record file."""
    for name, directory in profile.get("shader_directories", {}).items():
        stages = profile.setdefault("stage_maps", {}).setdefault(name, {})
        sources = profile.setdefault("sources", {})
        for key, record in directory_entries(records, directory):
            offset = directory.get("data_base", 0) + integer(record, directory["location"])
            size = integer(record, directory["size"])
            source = f"{name}:{key}"
            entry = {"offset": offset, "size": size, "encoding": directory.get("encoding", "raw")}
            if "raw_size" in directory:
                entry["raw_size"] = integer(record, directory["raw_size"])
            add_entry(sources, source, entry)
            add_entry(stages, key, source)
    for name, directory in profile.get("state_directories", {}).items():
        states = profile.setdefault("states", {}).setdefault(name, {})
        start, size = directory["fragment"]
        if start < 0 or size <= 0 or start + size > directory["stride"]:
            raise ValueError("invalid directory state fragment")
        for key, record in directory_entries(records, directory):
            add_entry(states, key, record[start:start + size].hex())
