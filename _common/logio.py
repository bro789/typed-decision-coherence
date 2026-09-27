"""Append-only JSONL logging, safe across threads."""
import json
import threading
import time
from pathlib import Path

from filelock import FileLock


def utcnow() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Appends on Windows are not atomic across file handles (O_APPEND is emulated), so two writers on the
        # same file -- two JsonlLog objects, or two processes -- can clobber each other. A file lock next to the
        # log serialises all writers. Added 2026-09-26, see _infra/CHANGELOG.md.
        self._flock = FileLock(str(self.path) + ".lock")

    def write(self, rec: dict):
        data = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock, self._flock:
            with open(self.path, "ab") as f:
                f.write(data)

    def read(self):
        return read_jsonl(self.path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def done_keys(path, ok_field="ok"):
    """Keys of records already completed successfully (for resumable runs)."""
    return {r["key"] for r in read_jsonl(path) if r.get(ok_field) and "key" in r}


def latest_by_key(path, ok_field="ok"):
    """key -> last successful record. Analysis scripts should use this, so resumed runs are deduplicated."""
    out = {}
    for r in read_jsonl(path):
        if r.get(ok_field) and "key" in r:
            out[r["key"]] = r
    return out
