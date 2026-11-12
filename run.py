"""
run.py
Convenience entry point for Smart Email Reply Generator.
Can run the web server, execute tests, or generate comparison tables.
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Smart Email Reply Generator Launcher")
    parser.add_argument("--test", action="store_true", help="Run the unit and validation test suite.")
    parser.add_argument("--benchmark", action="store_true", help="Generate comparison table across all 10 test cases.")
    parser.add_argument("--port", type=int, default=5000, help="Web application port (default: 5000).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web host binding (default: 127.0.0.1).")

    args = parser.parse_args()

    if args.test:
        import unittest
        print("[*] Running automated test suite...")
        loader = unittest.TestLoader()
        suite = loader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    elif args.benchmark:
        from tests.generate_comparison_table import run_comparison
        print("[*] Generating benchmark comparison table...")
        run_comparison()
        sys.exit(0)

    else:
        from web.app import main as run_web_app
        run_web_app(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
