#!/usr/bin/env python3
"""show_bin_info.py — dump info about a self-play .bin training file.

Each record is a fixed-size LearningState (see src/naf.py:413).
For TwixT board size 24 the record size is 1789 bytes:

    4   bytes  'JTwx' magic header
    8   bytes  last 4 moves (2 bytes each)
    720 bytes  packed board features (10 channels × 24² bits)
    1056 bytes visit-count target (528 moves × uint16)
    1   byte   z value (offset by +1: 0=loss, 1=draw, 2=win)

Usage:
    python show_bin_info.py spdata/iter6_00.bin
    python show_bin_info.py spdata/iter6_*.bin              # multiple files
    python show_bin_info.py spdata/iter6_*.bin --validate   # also check magic per record
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import naf

RECORD_SIZE = naf.LearningState.NUM_BYTES   # 1789 for SIZE=24
MAGIC = b'JTwx'


def inspect(path, validate=False):
    size = os.path.getsize(path)
    full = size // RECORD_SIZE
    rem = size % RECORD_SIZE
    info = {
        "path": path,
        "size_bytes": size,
        "size_mb": size / (1024 * 1024),
        "full_records": full,
        "trailing_bytes": rem,
        "torn_record": rem != 0,
    }
    if validate:
        bad = []
        with open(path, "rb") as f:
            for i in range(full):
                f.seek(i * RECORD_SIZE)
                head = f.read(len(MAGIC))
                if head != MAGIC:
                    bad.append(i)
                    if len(bad) > 10:
                        break
        info["records_validated"] = full
        info["records_with_bad_magic"] = bad
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="One or more .bin files")
    p.add_argument("--validate", action="store_true",
                   help="Verify b'JTwx' magic at every record boundary (slow on big files)")
    p.add_argument("--avg-moves-per-game", type=int, default=None,
                   help="If given, print estimated game count = positions / avg")
    args = p.parse_args()

    print(f"record_size = {RECORD_SIZE} bytes  (LearningState.NUM_BYTES, board={24})")
    print()

    total_size = 0
    total_records = 0
    any_torn = False
    any_bad_magic = False

    for path in args.files:
        if not os.path.isfile(path):
            print(f"  {path}: not a file, skipping", file=sys.stderr)
            continue
        info = inspect(path, validate=args.validate)
        total_size += info["size_bytes"]
        total_records += info["full_records"]
        any_torn = any_torn or info["torn_record"]

        size_mb = info["size_mb"]
        torn_note = f"  TORN(+{info['trailing_bytes']}B)" if info["torn_record"] else ""
        print(f"  {path}: {info['full_records']:>8d} records  "
              f"({size_mb:6.2f} MB){torn_note}")
        if args.validate and info["records_with_bad_magic"]:
            any_bad_magic = True
            print(f"    BAD MAGIC at record indices: {info['records_with_bad_magic']}")

    print()
    print(f"TOTAL: {total_records:,} records  ({total_size / (1024*1024):.2f} MB)")
    if args.avg_moves_per_game:
        est_games = total_records / args.avg_moves_per_game
        print(f"  estimated games (positions / {args.avg_moves_per_game}) = {est_games:,.0f}")

    if any_torn:
        print()
        print("note: at least one file ended on a torn record. Integer-divide "
              "by 1789 for safe counting; the trailing bytes were never a "
              "complete LearningState and should be discarded on resume.")
    if args.validate and any_bad_magic:
        sys.exit(1)


if __name__ == "__main__":
    main()
