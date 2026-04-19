"""Minimal local shim for iopath.common.file_io."""

from pathlib import Path


class _PathManager:
    @staticmethod
    def open(path, mode="r", *args, **kwargs):
        return open(path, mode, *args, **kwargs)

    @staticmethod
    def exists(path):
        return Path(path).exists()

    @staticmethod
    def isfile(path):
        return Path(path).is_file()


g_pathmgr = _PathManager()
