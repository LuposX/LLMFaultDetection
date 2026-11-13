"""Platform-tagged result cache.

Cross-method comparisons share per-fault, per-prompt quantities (random
detection vectors, healthy decodes, activity profiles). They are bitwise
reproducible only on the same platform: device, library versions, and CPU
thread count all change reduction order, and margin ties can flip argmax
verdicts between platforms. Every cache entry therefore stores the
platform tag it was computed under, and a lookup is a hit only when the
tag matches the current platform.

Storage is one append-only JSONL file per cache name; entries are keyed by
a sha1 of the key parts. Last write wins on load, so re-putting a key
simply appends a fresh entry.
"""

import hashlib
import json
import os

import torch
import transformers


def platform_tag():
    """Tag identifying the (hardware, software) combination that determines
    bitwise reproducibility. Deliberately excludes the model name: model
    identity belongs in the cache key when a value depends on it, not in
    the platform."""
    if torch.cuda.is_available():
        device = "cuda:" + str(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(0)
        device += f" {props.name}"
    else:
        device = "cpu"
    return {
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "default"),
    }


def _key_hash(parts):
    payload = json.dumps([str(p) for p in parts]).encode()
    return hashlib.sha1(payload).hexdigest()


class Cache:
    def __init__(self, path, strict=False):
        """strict=True raises on a platform-tag mismatch instead of
        silently treating it as a miss."""
        self.path = path
        self.strict = strict
        self.tag = platform_tag()
        self._entries = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    self._entries[e["key"]] = e
        self._f = None

    def get(self, *parts):
        """Returns the cached value or None. A hit requires the stored
        platform tag to equal the current tag."""
        e = self._entries.get(_key_hash(parts))
        if e is None:
            return None
        if e["platform"] != self.tag:
            if self.strict:
                raise RuntimeError(
                    f"cache entry {parts} was computed on a different "
                    f"platform: {e['platform']} != {self.tag}")
            return None
        return e["value"]

    def put(self, value, *parts):
        e = {"key": _key_hash(parts), "platform": self.tag,
             "value": value}
        self._entries[e["key"]] = e
        if self._f is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._f = open(self.path, "a")
        self._f.write(json.dumps(e) + "\n")
        self._f.flush()
