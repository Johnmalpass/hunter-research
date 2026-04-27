"""`python -m quant ...` entry point. Dispatches to quant.cli.main."""
import sys

from quant.cli import main

if __name__ == "__main__":
    sys.exit(main())
