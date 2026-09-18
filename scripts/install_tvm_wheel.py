#!/usr/bin/env python3
import json
import subprocess
import sys
import urllib.request

VERSION = "0.14.dev264"
JSON_URL = f"https://pypi.org/pypi/apache-tvm/{VERSION}/json"

with urllib.request.urlopen(JSON_URL) as response:
    data = json.load(response)

candidates = [
    item for item in data["urls"]
    if item["filename"].startswith(f"apache_tvm-{VERSION}-cp311-cp311-")
    and "manylinux" in item["filename"]
    and "x86_64" in item["filename"]
]

if len(candidates) != 1:
    raise SystemExit(
        f"expected exactly one CPython 3.11 x86_64 wheel for TVM {VERSION}, "
        f"found {[x['filename'] for x in candidates]}"
    )

wheel = candidates[0]
print(f"[install] {wheel['filename']}")
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    wheel["url"],
])
