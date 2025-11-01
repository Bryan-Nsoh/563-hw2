"""Generate the STAT 563 HW2 LaTeX report and compile it to PDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stat563_hw2.reporting import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the STAT 563 HW2 report")
    parser.add_argument(
        "--author",
        type=str,
        default="STAT 563 Student",
        help="Name to display on the report title page.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    tex_path = build_report(project_root, author=args.author)
    pdf_path = tex_path.with_suffix(".pdf")
    print(f"Report generated: {tex_path.relative_to(project_root)}")
    if pdf_path.exists():
        print(f"Compiled PDF: {pdf_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
