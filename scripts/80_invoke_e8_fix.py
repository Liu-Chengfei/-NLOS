"""Deprecated E8 patch entrypoint.

The original file was an incomplete one-off patch that referenced modules which
no longer exist. Keeping a small, explicit shim preserves the filename for old
automation while failing fast with an actionable message instead of breaking
repository-wide compilation or executing partial logic.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print("E8 patch is retired: use the maintained experiment pipeline instead.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
