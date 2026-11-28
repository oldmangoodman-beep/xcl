# Preparing a pipeline journal from an archive

Run `python pso_archive.py profile.json archive.bin pipelines.bin` with Python 3.10 or newer. Use `--records tables.bin` when a locally extracted archive member contains the tables and the shader data remains in the original archive. The command reports the number of distinct sources and recipes. It replaces the destination after every record resolves and the journal has been written.

A profile describes the local archive layout and how its selectors map to XCL wire records. The profile author must establish these mappings for the archive revision. Include optional `archive_sha256` to check the source archive. When the tables occupy compressed blocks, list their archive ranges in order under optional `record_chunks`, using the same range format as `sources`. Table offsets then address the concatenated decoded blocks. An absent selector raises an error. Use an explicitly empty variant list for an intentionally excluded record type.

| Version 1 field | Meaning |
|---|---|
| `sources` | Named shader byte ranges with `offset` and `size`. For `encoding` equal to `deflate` (raw) or `zlib`, supply the exact compressed stream length and `raw_size`. |
| `stage_maps` | Named dictionaries mapping a decimal selector string to a named source, or `null` for an absent optional shader stage. |
| `states` | Named dictionaries mapping decimal selector strings to hex-encoded state fragments in XCL wire order. |
| `tables` | A list of fixed-stride tables with `offset`, `count`, `stride`, and named `fields` given as `[byte_offset, width]` within a record. Integers use 1, 2, 4 or 8 little-endian bytes. |
| `variants` | A table's list of wire templates. Optionally use `variant_field` to select between lists keyed by decimal field value. Each variant supplies `kind` (C, G or M), the XCL journal `windows` byte, and an ordered `parts` list. |

A part is one of `{"hex":"..."}` for a literal, `{"field":["field name","state dictionary"]}` for a selected fragment, or `{"stage":["field name","stage dictionary"]}` for a shader identity computed from the selected local source. Templates must follow XCL's journal wire format, including any required zeroed fields and attachment layouts. The synthetic compute profile in `test_pso_archive.py` shows these parts in descriptor order.

The generated XCLPJ001 journal carries each required shader source once, followed by unique recipes. The tool enumerates the record tables and variants in the supplied profile. Coverage of gameplay paths depends on what that profile models.

If the archive contains its own dictionaries, define optional `shader_directories` and `state_directories` instead of listing entries by hand. Both take named directories with `offset`, `count`, `stride`, and an optional integer `id` field (defaults to the row ordinal). Offsets refer to the record file or decoded record chunks. A shader directory requires integer fields `location` and `size`, with optional `data_base`, `encoding` and integer `raw_size` field. Shader locations resolve against the original archive. A state directory requires `fragment: [offset, byte_length]` within the row. The named directory becomes a stage or state dictionary for the existing template operations. Conflicting entries raise an error.

Run `python -m unittest -v test_pso_archive test_pso_directory` in the script directory.
