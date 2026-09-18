#!/usr/bin/env python3
from pathlib import Path


ARTIFACTS = Path("artifacts")
MLF_DIR = ARTIFACTS / "mlf"


def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1024 / 1024:.2f} MiB"


def main() -> None:
    if not MLF_DIR.exists():
        raise SystemExit("artifacts/mlf does not exist")

    files = [p for p in MLF_DIR.rglob("*") if p.is_file()]
    c_files = [p for p in files if p.suffix == ".c"]

    print()
    print("[artifact] generated C files")
    for p in sorted(c_files):
        print(f"  {fmt_size(p.stat().st_size):>10}  {p.relative_to(MLF_DIR)}")

    print()
    print("[artifact] largest files")
    for p in sorted(files, key=lambda x: x.stat().st_size, reverse=True)[:12]:
        print(f"  {fmt_size(p.stat().st_size):>10}  {p.relative_to(MLF_DIR)}")

    total = sum(p.stat().st_size for p in files)
    print()
    print(f"[artifact] extracted MLF total: {fmt_size(total)}")


if __name__ == "__main__":
    main()
