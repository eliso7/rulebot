#!/usr/bin/env python3
"""Main CLI entry point."""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cli.__main__ import cli

def main():
    """Entry point for CLI."""
    cli()

if __name__ == "__main__":
    main()