import argparse
import ctypes
import pathlib
import re
import unittest
import xml.etree.ElementTree as ET


class WindowsLongPathManifestTests(unittest.TestCase):
    def test_manifest_enables_long_paths_and_is_embedded(self):
        manifest = pathlib.Path(self.manifest)
        resource = pathlib.Path(self.resource)

        root = ET.parse(manifest).getroot()
        setting = root.find(
            ".//{http://schemas.microsoft.com/SMI/2016/WindowsSettings}"
            "longPathAware"
        )
        self.assertIsNotNone(setting)
        self.assertEqual("true", (setting.text or "").strip().lower())

        resource_text = resource.read_text(encoding="utf-8")
        self.assertRegex(
            resource_text,
            re.compile(
                r"CREATEPROCESS_MANIFEST_RESOURCE_ID\s+RT_MANIFEST\s+"
                r'"shadps4\.manifest"'
            ),
        )

    def test_built_executable_embeds_long_path_manifest(self):
        if not self.executable:
            self.skipTest("no executable was supplied")

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        load_library = kernel32.LoadLibraryExW
        load_library.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint]
        load_library.restype = ctypes.c_void_p
        module = load_library(self.executable, None, 0x00000002)
        self.assertTrue(module, ctypes.WinError(ctypes.get_last_error()))
        try:
            find_resource = kernel32.FindResourceW
            find_resource.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            find_resource.restype = ctypes.c_void_p
            resource = find_resource(module, ctypes.c_void_p(1), ctypes.c_void_p(24))
            self.assertTrue(resource, ctypes.WinError(ctypes.get_last_error()))

            size_resource = kernel32.SizeofResource
            size_resource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            size_resource.restype = ctypes.c_uint
            size = size_resource(module, resource)

            load_resource = kernel32.LoadResource
            load_resource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            load_resource.restype = ctypes.c_void_p
            loaded = load_resource(module, resource)

            lock_resource = kernel32.LockResource
            lock_resource.argtypes = [ctypes.c_void_p]
            lock_resource.restype = ctypes.c_void_p
            address = lock_resource(loaded)
            embedded = ctypes.string_at(address, size).decode("utf-8")
            root = ET.fromstring(embedded)
            setting = root.find(
                ".//{http://schemas.microsoft.com/SMI/2016/WindowsSettings}"
                "longPathAware"
            )
            self.assertIsNotNone(setting)
            self.assertEqual("true", (setting.text or "").strip().lower())
        finally:
            free_library = kernel32.FreeLibrary
            free_library.argtypes = [ctypes.c_void_p]
            free_library.restype = ctypes.c_int
            free_library(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--executable")
    arguments, remaining = parser.parse_known_args()
    WindowsLongPathManifestTests.manifest = arguments.manifest
    WindowsLongPathManifestTests.resource = arguments.resource
    WindowsLongPathManifestTests.executable = arguments.executable
    unittest.main(argv=[__file__, *remaining])
