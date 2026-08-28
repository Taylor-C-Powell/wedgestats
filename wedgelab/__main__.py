"""Run the workbench: ``python -m wedgelab``."""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the GUI, reporting a missing Tk installation clearly."""
    try:
        from wedgelab.gui import launch
    except ImportError as exc:
        print(
            "wedgelab needs tkinter and matplotlib to show its interface.\n"
            f"  import failed: {exc}\n"
            "The computational half (wedgelab.compute, wedgelab.render) works "
            "without a display.",
            file=sys.stderr,
        )
        return 1
    launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
