import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from pso_archive import FRAME, FrameWriter, fnv64, generate, source_bytes


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.archive, self.profile, self.journal = (root / s for s in ("archive", "profile", "output"))
        self.shader = b"synthetic shader fixture"
        packed = zlib.compress(self.shader)
        self.archive.write_bytes(bytes([3, 7, 3, 7, 3, 9]) + packed)
        self.data = {
            "version": 1,
            "sources": {"shader-a": {"offset": 6, "size": len(packed), "encoding": "zlib", "raw_size": len(self.shader)}},
            "stage_maps": {"compute": {"3": "shader-a"}},
            "states": {"node": {"7": "00000000", "9": "01000000"}},
            "tables": [{"offset": 0, "count": 3, "stride": 2, "fields": {"stage": [0, 1], "node": [1, 1]},
                        "variants": [{"kind": "C", "windows": 4, "parts": [
                            {"stage": ["stage", "compute"]}, {"hex": "38000000" + "00" * 24},
                            {"field": ["node", "node"]}, {"hex": "00" * 28}]}]}],
        }

    def run_profile(self):
        self.profile.write_text(json.dumps(self.data), encoding="utf-8")
        return generate(self.profile, self.archive, self.journal)

    def test_wire_frames_and_deduplication(self):
        self.assertEqual(fnv64(b"hello"), 0xa430d84680aabd0b)
        self.assertEqual(self.run_profile(), {"source": 1, "recipe": 2})
        with self.journal.open("rb") as file:
            self.assertEqual(file.read(8), b"XCLPJ001")
            def read_frame():
                kind, windows, size, key = FRAME.unpack(file.read(16))
                return kind, windows, key, file.read(size)
            kind, windows, source_key, blob = read_frame()
            self.assertEqual((kind, windows, source_key, blob), (b"S", 0, fnv64(self.shader), self.shader))
            nodes = []
            for _ in range(2):
                kind, windows, key, payload = read_frame()
                self.assertEqual((kind, windows, key), (b"C", 4, fnv64(payload + bytes([4]))))
                self.assertEqual(struct.unpack_from("<QI", payload), (source_key, 56))
                nodes.append(struct.unpack_from("<I", payload, 36)[0])
                self.assertEqual(len(payload), 68)
            self.assertEqual(sorted(nodes), [0, 1])
            self.assertEqual(file.read(), b"")

    def test_frame_collision_rejects_different_payload_or_kind(self):
        frame = FrameWriter(io.BytesIO())
        frame.add("1", "G", 4, b"first")
        frame.add("1", "G", 4, b"first")
        with self.assertRaisesRegex(ValueError, "collision"):
            frame.add("1", "G", 4, b"other")
        with self.assertRaisesRegex(ValueError, "collision"):
            frame.add("1", "M", 4, b"first")

    def test_shader_range_aliases_are_read_once(self):
        alternate = dict(self.data["tables"][0]["variants"][0])
        alternate["parts"] = [{"stage": ["stage", "alias"]}] + alternate["parts"][1:]
        self.data["tables"][0]["variants"].append(alternate)
        self.data["stage_maps"]["alias"] = {"3": "shader-alias"}
        self.data["sources"]["shader-alias"] = dict(self.data["sources"]["shader-a"])
        with patch("pso_archive.source_bytes", wraps=source_bytes) as read:
            self.assertEqual(self.run_profile(), {"source": 1, "recipe": 2})
            self.assertEqual(read.call_count, 1)

    def test_missing_selector_preserves_output(self):
        self.run_profile()
        original = self.journal.read_bytes()
        del self.data["states"]["node"]["9"]
        with self.assertRaises(KeyError):
            self.run_profile()
        self.assertEqual(self.journal.read_bytes(), original)

    def test_archive_revision_and_separate_table_file(self):
        records = self.archive.with_name("records")
        data = self.archive.read_bytes()
        records.write_bytes(data[:6])
        self.archive.write_bytes(data[6:])
        self.data["sources"]["shader-a"]["offset"] = 0
        self.data["archive_sha256"] = hashlib.sha256(data[6:]).hexdigest()
        self.profile.write_text(json.dumps(self.data), encoding="utf-8")
        self.assertEqual(generate(self.profile, self.archive, self.journal, records)["recipe"], 2)
        self.archive.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            generate(self.profile, self.archive, self.journal, records)

    def test_chunked_records_cross_block_boundary(self):
        raw = self.archive.read_bytes()
        first, second = zlib.compress(raw[:3]), zlib.compress(raw[3:6])
        self.archive.write_bytes(first + second + raw[6:])
        self.data["sources"]["shader-a"]["offset"] = len(first) + len(second)
        self.data["record_chunks"] = [
            {"offset": 0, "size": len(first), "encoding": "zlib", "raw_size": 3},
            {"offset": len(first), "size": len(second), "encoding": "zlib", "raw_size": 3},
        ]
        self.assertEqual(self.run_profile(), {"source": 1, "recipe": 2})

    def test_invalid_compressed_length(self):
        self.data["sources"]["shader-a"]["raw_size"] -= 1
        with self.assertRaises(ValueError):
            self.run_profile()
        self.assertFalse(self.journal.exists())

    def test_discovers_sources_and_states_from_archive_directories(self):
        records = bytes([3, 7, 3, 9])
        shader_offset = len(records) + 5 + 10
        shader_index = struct.pack("<BHH", 3, shader_offset, len(self.shader))
        node_index = struct.pack("<BIBI", 7, 0, 9, 1)
        self.archive.write_bytes(records + shader_index + node_index + self.shader)
        self.data["tables"][0]["count"] = 2
        for name in ("sources", "stage_maps", "states"):
            del self.data[name]
        self.data["shader_directories"] = {"compute": {
            "offset": 4, "count": 1, "stride": 5, "id": [0, 1], "location": [1, 2], "size": [3, 2],
        }}
        self.data["state_directories"] = {"node": {
            "offset": 9, "count": 2, "stride": 5, "id": [0, 1], "fragment": [1, 4],
        }}
        self.assertEqual(self.run_profile(), {"source": 1, "recipe": 2})

    def test_unknown_record_or_bad_range(self):
        table = self.data["tables"][0]
        table["variant_field"] = "node"
        table["variants"] = {"7": table["variants"]}
        with self.assertRaises(KeyError):
            self.run_profile()
        table["fields"]["node"] = [2, 1]
        with self.assertRaises(ValueError):
            self.run_profile()


if __name__ == "__main__":
    unittest.main()
