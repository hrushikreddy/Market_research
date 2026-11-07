from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .empirical import run_empirical_analysis
from .figures import build_figures
from .pipeline import run_demo, run_empirical
from .sources import prepare_empirical_data


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Market information-delay research pipeline")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("demo", "run", "prepare-data", "empirical", "figures"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    settings = load_settings(args.config)
    if args.command == "demo":
        output = run_demo(settings)
    elif args.command == "run":
        output = run_empirical(settings)
    elif args.command == "prepare-data":
        output = Path(next(iter(prepare_empirical_data(Path.cwd()).values()))).parent
    elif args.command == "empirical":
        output = run_empirical_analysis(settings)
    else:
        output = build_figures(settings)
    print(f"Research outputs written to {output}")


if __name__ == "__main__":
    main()
