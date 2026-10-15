"""Command-line entry point: `rtt <command>`."""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rtt", description="Robot Tool Tournament")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print version")
    args = parser.parse_args(argv)
    if args.cmd == "version":
        from importlib.metadata import version

        print(version("robot-tool-tournament"))


if __name__ == "__main__":
    main()
