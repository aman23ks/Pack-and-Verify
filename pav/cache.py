import os, orjson, pathlib
from .utils import sha12

class DiskCache:
    def __init__(self, root: str = ".pav_cache"):
        self.root = root
        pathlib.Path(self.root).mkdir(parents=True, exist_ok=True)

    def _path(self, ns: str, key: str) -> str:
        d = os.path.join(self.root, ns)
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        return os.path.join(d, f"{sha12(key)}.json")

    def get(self, ns: str, key: str):
        p = self._path(ns, key)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return orjson.loads(f.read())
        return None

    def set(self, ns: str, key: str, val):
        p = self._path(ns, key)
        with open(p, "wb") as f:
            f.write(orjson.dumps(val))

CACHE = DiskCache()
