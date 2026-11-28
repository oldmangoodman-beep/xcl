import io
import struct
import unittest

from pso_directory import read_directories


class DirectoryTests(unittest.TestCase):
    def test_shader_and_state_indices(self):
        contents = struct.pack("<BHHBHH", 9, 30, 4, 3, 34, 7) + bytes.fromhex("02aabb03ccdd")
        profile = {
            "shader_directories": {"vs": {"offset": 0, "count": 2, "stride": 5, "id": [0, 1],
                                           "location": [1, 2], "size": [3, 2], "data_base": 100}},
            "state_directories": {"raster": {"offset": 10, "count": 2, "stride": 3,
                                              "id": [0, 1], "fragment": [1, 2]}},
        }
        read_directories(profile, io.BytesIO(contents))
        self.assertEqual(profile["stage_maps"]["vs"], {"3": "vs:3", "9": "vs:9"})
        self.assertEqual(profile["sources"]["vs:3"], {"offset": 134, "size": 7, "encoding": "raw"})
        self.assertEqual(profile["states"]["raster"], {"2": "aabb", "3": "ccdd"})

    def test_conflicting_directory_entry(self):
        profile = {"state_directories": {"raster": {"offset": 0, "count": 2, "stride": 2,
                                                      "id": [0, 1], "fragment": [1, 1]}}}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            read_directories(profile, io.BytesIO(bytes([1, 2, 1, 3])))

    def test_truncated_directory(self):
        profile = {"state_directories": {"raster": {"offset": 0, "count": 3, "stride": 2,
                                                      "fragment": [1, 1]}}}
        with self.assertRaisesRegex(ValueError, "truncated"):
            read_directories(profile, io.BytesIO(bytes(4)))


if __name__ == "__main__":
    unittest.main()
