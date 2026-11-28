"""Synthetic wire-format tests for stage.py. Run python -m unittest discover -s scripts/pso_corpus."""
from pathlib import Path
import struct
import tempfile
import unittest

import stage


def frame(tag, data, window=132):
    if tag == "S":
        window = 0
    key = stage.fnv(data if tag == "S" else data + bytes([window]))
    return stage.HEADER.pack(tag.encode(), window, 0, len(data), key) + data


def graphic(key, bounds=False):
    d = bytearray(656)
    struct.pack_into("<I", d, 448, 0xffffffff)
    struct.pack_into("<I", d, 560, 1)
    return (struct.pack("<5QI", key, 0, 0, 0, 0, 656) + d + struct.pack("<II", 1, 3)
            + b"POS" + struct.pack("<6I", 1, 2, 3, 4, 5, 6) + (b"DB01" if bounds else b""))


def compute(key, node):
    d = bytearray(56)
    struct.pack_into("<I", d, 24, node)
    return struct.pack("<QI", key, 56) + d


class Staging(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.title = Path(tmp.name)
        self.shader = b"local synthetic shader"
        self.key = stage.fnv(self.shader)
        self.src = frame("S", self.shader)
        self.dest = self.title / ".xcl/pso-corpus"

    def journal(self, name, frames):
        path = self.title / name
        path.write_bytes(b"XCLPJ001" + b"".join(frames))
        return path

    def test_stage_union_with_forward_sources_and_node_aliases(self):
        a = self.journal("first", [frame("G", graphic(self.key)), frame("C", compute(self.key, 0))])
        mesh = (struct.pack("<IIQIQII", 0, 1, 0, 25, self.key, 12, 21)
                + bytes(52) + struct.pack("<I", 1))
        b = self.journal("second", [self.src, frame("G", graphic(self.key, True)),
                                    frame("C", compute(self.key, 1)), frame("M", mesh)])
        counts = stage.stage(self.title, [a, b, a])
        self.assertEqual((counts["C"], counts["compute_files"], counts["compute_node_aliases"]), (2, 1, 1))
        self.assertEqual((self.dest / "compute" / f"{self.key:016x}.dxbc").read_bytes(), self.shader)
        self.assertEqual((self.dest / "windows").read_text().strip(), "132")
        self.assertEqual((self.dest / "complete").read_bytes(), b"3 1\n")
        identity = (self.dest / "identity").read_text().strip()
        self.assertEqual(len(identity), 64)
        self.assertEqual(identity, stage.corpus_identity(self.dest))
        original = identity
        (self.dest / "compute" / f"{self.key:016x}.dxbc").write_bytes(b"changed")
        self.assertNotEqual(original, stage.corpus_identity(self.dest))
        with (self.dest / "graphics.bin").open("rb") as f:
            self.assertEqual(f.read(12), stage.GFX_HEADER)
            entries = []
            while head := f.read(8):
                size, check = struct.unpack("<II", head)
                data = f.read(size)
                self.assertEqual(stage.fnv(data) & 0xffffffff, check)
                entries.append(data)
        self.assertEqual([e[0] for e in entries], [2, 1, 4, 3])
        self.assertEqual(entries[1][1:], entries[2][1:])
        self.assertEqual(entries[0][1:9], entries[1][1:9])
        self.assertEqual(entries[1][-11:], b"\1\3POS\1\2\3\4\5\6")
        self.assertEqual(entries[3][25:31], bytes([5, 0, 1, 25, 12, 21]))
        self.assertEqual(entries[3][-1], 1)
        with self.assertRaisesRegex(ValueError, "preserved"):
            stage.stage(self.title, [a])

    def test_compute_only_corpus_omits_graphics_job(self):
        stage.stage(self.title, [self.journal("compute", [self.src, frame("C", compute(self.key, 0))])])
        self.assertEqual((self.dest / "complete").read_bytes(), b"0 1\n")
        self.assertEqual((self.dest / "identity").read_text().strip(), stage.corpus_identity(self.dest))
        self.assertFalse((self.dest / "graphics.bin").exists())

    def test_invalid_frames_never_publish(self):
        good = frame("G", graphic(self.key))
        ptr = bytearray(graphic(self.key))
        ptr[44] = 1
        overflow = bytearray(graphic(self.key))
        struct.pack_into("<I", overflow, 44 + 576, 9)
        invalid = [
            [self.src, good[:-1]], [good], [self.src, frame("G", ptr)],
            [self.src, frame("G", overflow)], [self.src, good, frame("G", graphic(self.key), 4)],
            [self.src[:-1] + b"X", good], [self.src, good[:-1] + b"X"],
            [self.src, frame("C", compute(self.key, 2))],
            [self.src, frame("C", compute(self.key, 0), 133)],
            [self.src, frame("M", struct.pack("<IQII", 25, self.key, 12, 12))],
            [self.src, frame("M", struct.pack("<IQI", 25, self.key, 11) + bytes(52)
                             + struct.pack("<I", 21) + bytes(56))],
            [self.src, frame("M", struct.pack("<IQIQ", 25, self.key, 1, self.key))],
        ]
        for number, case in enumerate(invalid):
            with self.subTest(number=number), self.assertRaises(ValueError):
                stage.stage(self.title, [self.journal("bad", case)])
            self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
